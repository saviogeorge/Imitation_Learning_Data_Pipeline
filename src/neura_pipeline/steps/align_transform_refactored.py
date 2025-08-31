
from pathlib import Path
from typing import Dict, List, Optional
import json
from json import JSONDecodeError

import polars as pl

from ..core.constants import (
    RAW_OBS_COL,
    OBS_SAFE_COL,
    ACTION_COL,
    REQUIRED_EPISODE_COLS,
    LIST_WIDTH,
)


# ---------- helpers ----------
def _load_stats(stats_path: Optional[Path]) -> Optional[Dict]:
    if not stats_path:
        return None
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, JSONDecodeError):
        return None


def _ensure_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast scalar columns to stable types and ensure vector columns are Lists."""
    return lf.with_columns(
        [
            pl.col("timestamp").cast(pl.Float64),
            pl.col("frame_index").cast(pl.Int64),
            pl.col("episode_index").cast(pl.Int64),
            pl.col("index").cast(pl.Int64),
            pl.col("task_index").cast(pl.Int64),
            pl.col(ACTION_COL).cast(pl.List(pl.Float32)),
            pl.col(OBS_SAFE_COL).cast(pl.List(pl.Float32)),
        ]
    )


def _enforce_list_width(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep only rows where both vectors match expected width."""
    return lf.filter(
        (pl.col(ACTION_COL).list.len() == LIST_WIDTH)
        & (pl.col(OBS_SAFE_COL).list.len() == LIST_WIDTH)
    )


def _normalize_list_with_stats(
    lf: pl.LazyFrame,
    col: str,
    mean: List[float],
    std: List[float],
) -> pl.LazyFrame:
    """
    Z-score normalize a list<float> column using global stats.
    If std == 0 at any dimension, divide by 1 (no scaling) but still center.
    """
    if not (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == LIST_WIDTH
        and len(std) == LIST_WIDTH
    ):
        # Stats malformed or wrong dimensionality — leave column unchanged.
        return lf

    return lf.with_columns(
        pl.col(col).map_elements(
            lambda v: [
                (vi - mi) / (si if si != 0 else 1.0)
                for vi, mi, si in zip(v, mean, std)
            ],
            return_dtype=pl.List(pl.Float32),
        )
    )


# ---------- public API ----------
def align_and_transform_episode(
    src_fp: Path,
    stats: Optional[Dict],
    normalize: bool = True,
) -> pl.DataFrame:
    """
    Align an episode parquet to a clean, normalized schema:
      - select required columns
      - rename dotted obs column to a safe internal name
      - cast scalars & vectors to stable dtypes
      - enforce expected vector width
      - sort by frame_index, de-dup frames (keep first), reindex 0..N-1
      - drop bad/null rows on key scalars
      - (optional) z-score normalize ACTION + OBS using global stats
      - rename back to dotted obs column for downstream compatibility
    """
    lf = (
        pl.scan_parquet(str(src_fp))
        .select(REQUIRED_EPISODE_COLS)
        .rename({RAW_OBS_COL: OBS_SAFE_COL})
    )

    lf = _ensure_schema(lf)
    lf = _enforce_list_width(lf)

    # Sort, drop duplicate frame_index keeping first occurrence, maintain order, and reindex 0..N-1
    lf = lf.sort("frame_index").unique(
        subset=["frame_index"], keep="first", maintain_order=True
    )
    lf = lf.with_columns(pl.int_range(0, pl.len()).alias("frame_index"))

    # Remove any NaNs/nulls in critical numeric scalars
    lf = lf.filter(
        pl.col("timestamp").is_not_null()
        & ~pl.col("timestamp").is_nan()
        & pl.col("frame_index").is_not_null()
        & pl.col("episode_index").is_not_null()
    )

    # (Optional) z-score normalize list columns using global_stats.json
    if normalize and (stats is not None):
        try:
            a_mu, a_sd = stats[ACTION_COL]["mean"], stats[ACTION_COL]["std"]
            s_mu, s_sd = stats[RAW_OBS_COL]["mean"], stats[RAW_OBS_COL]["std"]
            lf = _normalize_list_with_stats(lf, ACTION_COL, a_mu, a_sd)
            lf = _normalize_list_with_stats(lf, OBS_SAFE_COL, s_mu, s_sd)
        except Exception:
            # Stats not present or malformed — skip normalization gracefully
            pass

    # Rename internal column back to original dotted name for downstream
    lf = lf.rename({OBS_SAFE_COL: RAW_OBS_COL})

    # Materialize
    return lf.collect(streaming=True)


def process_episodes(
    data_dir: Path,
    out_dir: Path,
    stats_path: Optional[Path],
    normalize: bool = True,
) -> None:
    """
    Process all 'episode_*.parquet' under <data_dir>/chunk-*/ into <out_dir>.
    Also supports a flat directory (no chunk-*).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _load_stats(stats_path) if stats_path else None

    # Discover all episodes across chunks
    episode_files = sorted((data_dir).glob("chunk-*/episode_*.parquet"))
    if not episode_files:
        # Also support a direct directory of episodes (no chunk-*), if present
        episode_files = sorted((data_dir).glob("episode_*.parquet"))

    for src_fp in episode_files:
        df = align_and_transform_episode(src_fp, stats, normalize)
        df.write_parquet(out_dir / src_fp.name)
