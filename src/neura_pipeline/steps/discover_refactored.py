from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple
import concurrent.futures as cf
import json
from datetime import datetime, timezone

import polars as pl

from ..core.constants import CAMERAS, FINGERPRINT_ALGO
from ..core.models import EpisodeManifestRow
from ..core.statuses import Status
from ..fp.fingerprint import (
    quick_file_fingerprint,
    stable_sleep_check,
    combine_episode_fingerprint,
)
from ..io.fs_local import LocalFS

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def write_atomic_parquet(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.write_parquet(tmp)
    import os; os.replace(tmp, out)

def episode_index_from_name(path: Path) -> Optional[int]:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None

def fingerprint_episode(fs: LocalFS, chunk: str, pq: Path, *, full_hash: bool = False) -> EpisodeManifestRow | None:
    ep_idx = episode_index_from_name(pq)
    if ep_idx is None:
        return EpisodeManifestRow(
            episode_index=-1, chunk=chunk, parquet_uri=str(pq),
            video_front_uri=None, video_wrist_uri=None,
            exists_front=False, exists_wrist=False,
            bytes_total=0, fingerprint=None, fingerprint_algo=FINGERPRINT_ALGO,
            discovered_at=utc_now(), status=Status.ERROR,
            errors=json.dumps({"reason": "bad_episode_name"}),
        )
    v_front = fs.video_path(chunk, CAMERAS[0], ep_idx)
    v_wrist = fs.video_path(chunk, CAMERAS[1], ep_idx)

    pending = False
    for p in (pq, v_front, v_wrist):
        if p.exists() and not stable_sleep_check(p):
            pending = True; break

    try:
        parts: Dict[str, dict] = {"parquet": quick_file_fingerprint(pq, full_hash=full_hash)}
        if v_front.exists(): parts[CAMERAS[0]] = quick_file_fingerprint(v_front, full_hash=full_hash)
        if v_wrist.exists(): parts[CAMERAS[1]] = quick_file_fingerprint(v_wrist, full_hash=full_hash)
        fingerprint = combine_episode_fingerprint(parts)
        bytes_total = sum(d["size"] for d in parts.values())
        err = None
    except Exception as e:
        fingerprint = None; bytes_total = 0; pending = False
        err = json.dumps({"exception": type(e).__name__, "msg": str(e)})

    exists_front, exists_wrist = v_front.exists(), v_wrist.exists()
    status = Status.PENDING if pending else Status.NEW
    if not exists_front or not exists_wrist:
        if status == Status.NEW: status = Status.MISSING_SIDE

    return EpisodeManifestRow(
        episode_index=ep_idx, chunk=chunk, parquet_uri=str(pq),
        video_front_uri=str(v_front) if exists_front else None,
        video_wrist_uri=str(v_wrist) if exists_wrist else None,
        exists_front=exists_front, exists_wrist=exists_wrist,
        bytes_total=bytes_total, fingerprint=fingerprint, fingerprint_algo=FINGERPRINT_ALGO,
        discovered_at=utc_now(), status=status, errors=err,
    )

def discover_incremental(
    data_root: str | Path, manifest_out: str | Path, *,
    workers: int = 16, since_ns: Optional[int] = None,
    full_hash: bool = False, only_chunks: Optional[Iterable[str]] = None,
) -> pl.DataFrame:
    fs = LocalFS(data_root); manifest_out = Path(manifest_out)
    prev: Optional[pl.DataFrame] = pl.read_parquet(manifest_out) if manifest_out.exists() else None
    only_chunks_set = set(only_chunks) if only_chunks else None

    parquets: List[tuple[str, Path]] = []
    chunks = sorted(fs.list_chunks()) if not only_chunks_set else sorted(only_chunks_set)
    for chunk in chunks:
        for pq in fs.list_parquets(chunk):
            if since_ns is not None:
                try:
                    if pq.stat().st_mtime_ns < since_ns: continue
                except FileNotFoundError:
                    continue
            parquets.append((chunk, pq))

    rows: List[EpisodeManifestRow] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, 64))) as ex:
        futs = [ex.submit(fingerprint_episode, fs, c, p, full_hash=full_hash) for c, p in parquets]
        for f in cf.as_completed(futs):
            r = f.result()
            if r is not None: rows.append(r)

    df_cur = pl.from_dicts([r.__dict__ for r in rows]) if rows else pl.DataFrame(schema={
        "episode_index": pl.Int64, "chunk": pl.Utf8, "parquet_uri": pl.Utf8,
        "video_front_uri": pl.Utf8, "video_wrist_uri": pl.Utf8,
        "exists_front": pl.Boolean, "exists_wrist": pl.Boolean,
        "bytes_total": pl.Int64, "fingerprint": pl.Utf8,
        "fingerprint_algo": pl.Utf8, "discovered_at": pl.Utf8,
        "status": pl.Utf8, "errors": pl.Utf8,
    })

    if prev is not None and df_cur.height:
        key_cols = ["chunk", "episode_index"]
        joined = df_cur.join(prev.select(key_cols + ["fingerprint"]), on=key_cols, how="left", suffix="_prev")
        df_cur = joined.with_columns(
            pl.when(pl.col("fingerprint").is_null()).then(pl.lit(str(Status.ERROR)))
             .when(pl.col("fingerprint") == pl.col("fingerprint_prev")).then(pl.lit(str(Status.UNCHANGED)))
             .otherwise(pl.col("status")).alias("status")
        ).drop(["fingerprint_prev"])

    deleted_rows: List[dict] = []
    if prev is not None and prev.height:
        prev_keys = prev.select(["chunk", "episode_index"]).unique()
        cur_keys = df_cur.select(["chunk", "episode_index"]).unique()
        anti = prev_keys.join(cur_keys, on=["chunk", "episode_index"], how="anti")
        for rec in anti.iter_rows(named=True):
            deleted_rows.append({
                "episode_index": rec["episode_index"], "chunk": rec["chunk"],
                "parquet_uri": None, "video_front_uri": None, "video_wrist_uri": None,
                "exists_front": False, "exists_wrist": False, "bytes_total": 0,
                "fingerprint": None, "fingerprint_algo": FINGERPRINT_ALGO,
                "discovered_at": utc_now(), "status": str(Status.DELETED), "errors": None,
            })

    parquet_key_set = {(r["chunk"], int(r["episode_index"])) for r in df_cur.select(["chunk","episode_index"]).iter_rows(named=True)}
    orphan_rows: List[dict] = []
    for chunk in chunks:
        vids_root = Path(data_root) / "videos" / f"chunk-{chunk}"
        for cam in CAMERAS:
            cam_dir = vids_root / cam
            if not cam_dir.exists(): continue
            for mp4 in sorted(cam_dir.glob("episode_*.mp4")):
                try: ep_idx = int(mp4.stem.split("_")[-1])
                except Exception: continue
                if (chunk, ep_idx) not in parquet_key_set:
                    orphan_rows.append({
                        "episode_index": ep_idx, "chunk": chunk, "parquet_uri": None,
                        "video_front_uri": str(mp4) if cam == CAMERAS[0] else None,
                        "video_wrist_uri": str(mp4) if cam == CAMERAS[1] else None,
                        "exists_front": cam == CAMERAS[0], "exists_wrist": cam == CAMERAS[1],
                        "bytes_total": mp4.stat().st_size if mp4.exists() else 0,
                        "fingerprint": None, "fingerprint_algo": FINGERPRINT_ALGO,
                        "discovered_at": utc_now(), "status": str(Status.ORPHAN_VIDEO), "errors": None,
                    })

    parts = [df_cur]
    if deleted_rows: parts.append(pl.from_dicts(deleted_rows))
    if orphan_rows: parts.append(pl.from_dicts(orphan_rows))

    df_all = pl.concat(parts, how="vertical_relaxed").sort(["chunk", "episode_index"]) if parts else df_cur
    write_atomic_parquet(df_all, manifest_out)

    return df_all.filter(pl.col("status").is_in([
        str(Status.NEW), str(Status.CHANGED), str(Status.MISSING_SIDE),
        str(Status.DELETED), str(Status.ORPHAN_VIDEO), str(Status.PENDING),
        str(Status.ERROR)
    ]))
