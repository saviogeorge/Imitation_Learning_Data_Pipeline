
from pathlib import Path
from typing import List, Dict
import json

import polars as pl

from ..validate.validate_one import validate_one
from ..core.statuses import Status  # reuse your existing enum

# Which discover statuses should trigger validation
ACTIONABLE = {Status.NEW, Status.CHANGED, Status.MISSING_SIDE, Status.PENDING, Status.ERROR, Status.ORPHAN_VIDEO}

def _videos_root_from_row(row: dict) -> Path:
    """Derive videos root for the chunk from any available path in the row."""
    if row.get("video_front_uri"):
        return Path(row["video_front_uri"]).parents[1]
    if row.get("video_wrist_uri"):
        return Path(row["video_wrist_uri"]).parents[1]
    # derive from parquet path: .../data/chunk-XXX -> .../videos/chunk-XXX
    dataset_root = Path(row["parquet_uri"]).parents[2]
    return dataset_root / "videos" / ("chunk-%s" % row["chunk"])

def validate_from_manifest_refactored(
    manifest_fp: Path,
    meta_dir: Path,
    out_dir: Path,
    fps_expected: float = 30.0,
    frame_tolerance: int = 2,
    skip_video: bool = False,
) -> Dict[str, int]:
    """
    Reads the discover manifest and validates only actionable episodes.
    Writes:
      - episodes.parquet (all results)
      - failures.jsonl
      - validated_episodes.jsonl (ok-only, for downstream stages)
      - summary.yaml
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pl.read_parquet(str(manifest_fp))
    eps = manifest.filter(pl.col("status").is_in([s.value for s in ACTIONABLE]))

    meta_path = meta_dir / "episodes.jsonl"
    meta_eps = pl.read_ndjson(str(meta_path)) if meta_path.exists() else pl.DataFrame({"episode_index": [], "length": []})

    results: List[dict] = []
    failures: List[dict] = []

    for row in eps.iter_rows(named=True):
        pq = Path(row["parquet_uri"]) if row.get("parquet_uri") else None
        ep_idx = int(row["episode_index"])

        if pq is None or not pq.exists():
            r = {
                "episode_index": ep_idx,
                "rows": None,
                "frame_min": None,
                "frame_max": None,
                "expected_rows_meta": None,
                "ok": False,
                "issues": [{"parquet_missing": row.get("parquet_uri")}],
                "chunk": row.get("chunk"),
                "parquet_uri": row.get("parquet_uri"),
                "video_front_uri": row.get("video_front_uri"),
                "video_wrist_uri": row.get("video_wrist_uri"),
            }
            results.append(r); failures.append(r); continue

        videos_root = _videos_root_from_row(row)

        r = validate_one(
            pq,
            videos_root=videos_root,
            episodes_meta=meta_eps,
            fps_expected=fps_expected,
            frame_tolerance=frame_tolerance,
            skip_video=skip_video,
        )
        # add traceability
        r["chunk"] = row.get("chunk")
        r["parquet_uri"] = row.get("parquet_uri")
        r["video_front_uri"] = row.get("video_front_uri")
        r["video_wrist_uri"] = row.get("video_wrist_uri")

        results.append(r)
        if not r["ok"]:
            failures.append(r)

    # Write files
    pl.DataFrame(results).write_parquet(out_dir / "episodes.parquet")

    with open(out_dir / "failures.jsonl", "w") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")

    with open(out_dir / "validated_episodes.jsonl", "w") as f:
        for r in results:
            if r.get("ok"):
                f.write(json.dumps({
                    "episode_index": r.get("episode_index"),
                    "rows": r.get("rows"),
                    "chunk": r.get("chunk"),
                    "parquet_uri": r.get("parquet_uri"),
                    "video_front_uri": r.get("video_front_uri"),
                    "video_wrist_uri": r.get("video_wrist_uri"),
                }) + "\n")

    summary = {"total": len(results), "ok": sum(1 for r in results if r["ok"]), "fail": len(failures)}
    (out_dir / "summary.yaml").write_text("total: {}\nok: {}\nfail: {}\n".format(
        summary["total"], summary["ok"], summary["fail"]
    ))
    return summary
