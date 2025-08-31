
from pathlib import Path
from typing import Optional, Dict, List, Any
import json, subprocess, shlex

import polars as pl

# Required columns in the episode parquet
REQUIRED = [
    "action", "observation.state", "timestamp",
    "frame_index", "episode_index", "index", "task_index"
]
LIST_WIDTH = 8  # 7 DOF + gripper

def _scan_episode(fp: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(fp))

def _ffprobe_metadata(mp4: Path) -> Optional[Dict[str, Any]]:
    if not mp4.exists():
        return None
    cmd = (
        'ffprobe -v error -select_streams v:0 '
        '-show_entries stream=nb_frames,r_frame_rate,avg_frame_rate,duration '
        '-of json '
        f'"{mp4}"'
    )
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]

        def _rate_to_float(rate):
            if not rate or "/" not in rate:
                return None
            n, d = rate.split("/")
            d = float(d) if float(d) != 0 else 1.0
            return float(n) / d

        return {
            "nb_frames": int(stream.get("nb_frames")) if stream.get("nb_frames") else None,
            "r_fps": _rate_to_float(stream.get("r_frame_rate")),
            "avg_fps": _rate_to_float(stream.get("avg_frame_rate")),
            "duration": float(stream.get("duration")) if stream.get("duration") else None,
        }
    except Exception:
        return None

def _episode_expected_rows(meta_df: pl.DataFrame, ep_idx: int) -> Optional[int]:
    m = meta_df.filter(pl.col("episode_index") == ep_idx)
    return int(m["length"][0]) if m.height else None

def _vector_width_max(colname: str) -> pl.Expr:
    e = pl.col(colname)
    # If it's a fixed-size Array, convert to List then take length
    if hasattr(e, "arr") and hasattr(e.arr, "to_list"):
        return e.arr.to_list().list.len().max()
    # If it's already a List, just take length
    if hasattr(e, "list") and hasattr(e.list, "len"):
        return e.list.len().max()
    # Fallback: try a safe cast to List (works for array[f32,8] → List[f32])
    return e.cast(pl.List(pl.Float32)).list.len().max()

def validate_one(
    ep_fp: Path,
    videos_root: Path,
    episodes_meta: pl.DataFrame,
    fps_expected: float = 30.0,
    frame_tolerance: int = 2,
    skip_video: bool = False,
) -> Dict[str, Any]:
    """Validate a single episode parquet against basic schema and (optionally) video."""
    ep_idx = int(ep_fp.stem.split("_")[-1])
    lf = _scan_episode(ep_fp)

    issues = []  # type: List[Dict[str, Any]]
    ok = True

    # --- Schema / required columns
    try:
        cols = list(lf.collect_schema().names())
    except Exception as e:
        return {"episode_index": ep_idx, "ok": False, "issues": [{"schema_error": repr(e)}]}

    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        return {"episode_index": ep_idx, "ok": False, "issues": [{"missing_columns": missing}]}

    # --- Vectorized checks (no row iteration)
    df = lf.select(
        pl.len().alias("rows"),
        pl.col("frame_index").min().alias("fmin"),
        pl.col("frame_index").max().alias("fmax"),
        (pl.col("frame_index").diff().fill_null(1).gt(0).all()).alias("frame_sorted"),
        (pl.col("timestamp").diff().fill_null(0).ge(0).all()).alias("ts_sorted"),
        pl.any_horizontal(pl.col(REQUIRED).is_null()).alias("has_nulls"),
        pl.first("episode_index").alias("ep0"),
        pl.last("episode_index").alias("eplast"),
        _vector_width_max("action").alias("action_w_max"),
        _vector_width_max("observation.state").alias("state_w_max"),
    ).collect(streaming=True)

    rows = int(df["rows"][0])
    fmin, fmax = int(df["fmin"][0]), int(df["fmax"][0])

    if fmin != 0:
        ok = False; issues.append({"frame_index_start": fmin})
    if not bool(df["frame_sorted"][0]):
        ok = False; issues.append({"frame_index_not_sorted": True})
    if not bool(df["ts_sorted"][0]):
        ok = False; issues.append({"timestamp_not_sorted": True})
    if bool(df["has_nulls"][0]):
        ok = False; issues.append({"nulls_in_required_columns": True})
    if int(df["ep0"][0]) != ep_idx or int(df["eplast"][0]) != ep_idx:
        ok = False; issues.append({"episode_index_mismatch": [int(df["ep0"][0]), int(df["eplast"][0]), ep_idx]})
    if int(df["action_w_max"][0] or 0) != LIST_WIDTH:
        ok = False; issues.append({"action_width": int(df["action_w_max"][0] or -1)})
    if int(df["state_w_max"][0] or 0) != LIST_WIDTH:
        ok = False; issues.append({"state_width": int(df["state_w_max"][0] or -1)})

    exp_rows = _episode_expected_rows(episodes_meta, ep_idx)
    if exp_rows is not None and abs(exp_rows - rows) > frame_tolerance:
        ok = False; issues.append({"rows_vs_meta": {"meta": exp_rows, "table": rows}})

    # --- Video checks (optional)
    if not skip_video:
        for cam in ("front", "wrist"):
            vp = videos_root / ("observation.images.%s" % cam) / ("episode_%06d.mp4" % ep_idx)
            meta = _ffprobe_metadata(vp) if vp.exists() else None
            if not meta:
                ok = False; issues.append({("%s_video_missing" % cam): str(vp)})
                continue
            nb = meta["nb_frames"]
            fps = meta["avg_fps"] or meta["r_fps"]
            if fps and abs(fps - fps_expected) > 1.0:
                ok = False; issues.append({("%s_fps" % cam): fps})
            if nb is not None and abs(nb - rows) > frame_tolerance:
                ok = False; issues.append({("%s_frames_vs_rows" % cam): {"video": nb, "table": rows}})

    return {
        "episode_index": ep_idx,
        "rows": rows,
        "frame_min": fmin,
        "frame_max": fmax,
        "expected_rows_meta": exp_rows,
        "ok": ok,
        "issues": issues,
    }
