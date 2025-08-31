
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple
import os, json, shutil

import polars as pl

from ..core.constants import CAMERAS  # e.g., ["observation.images.front","observation.images.wrist"]
from ..utils import stable_hash_int, hash_to_unit_interval


# Linking strategy (keep it simple for py3.8)
LinkMethod = str  # "symlink" | "hardlink" | "copy" | "manifest-only"


def _episode_split(
    ep_idx: int,
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> str:
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError("train+val+test must equal 1.0")
    h = hash_to_unit_interval(stable_hash_int(ep_idx, seed))
    return "train" if h < train else ("val" if h < (train + val) else "test")


def _link(src: Path, dst: Path, method: LinkMethod) -> None:
    """Create link/copy for videos (or skip if 'manifest-only')."""
    if method == "manifest-only":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if method == "symlink":
        # Use relative symlink for portability
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif method == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    elif method == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("Unknown link method: {}".format(method))


def _iter_episodes(norm_dir: Path) -> List[Path]:
    files = sorted(norm_dir.rglob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError("No episode_*.parquet found under {}".format(norm_dir))
    return files


def materialize_partitioned(
    norm_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    chunk_id: str = "000",
    # video handling
    videos_root: Optional[Path] = None,                # e.g. Path("robot_data/videos")
    views: Optional[Iterable[str]] = None,             # default to core.constants.CAMERAS
    video_source_chunk_id: str = "000",               # usually videos/chunk-000
    link_videos: LinkMethod = "symlink",               # "symlink" | "hardlink" | "copy" | "manifest-only"
) -> Dict[str, object]:
    """
    Build a simple, trainer-friendly dataset layout:

      out_dir/
        split={train|val|test}/chunk=<chunk_id>/
          episode_XXXXXX.parquet
          videos/<view>/episode_XXXXXX.mp4     (linked/copied or omitted if manifest-only)

      out_dir/dataset_index.parquet            (index of all items + paths)
      out_dir/_manifest.json                   (bookkeeping: seed/fractions/counts/config)

    'norm_dir' is the output of align-transform stage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = list(views) if views is not None else list(CAMERAS)

    counts = {"train": 0, "val": 0, "test": 0}
    index_rows: List[Dict[str, object]] = []

    for fp in _iter_episodes(Path(norm_dir)):
        # episode index from file name
        try:
            ep_idx = int(fp.stem.split("_")[-1])
        except Exception:
            raise ValueError("Cannot parse episode index from {}".format(fp.name))

        split = _episode_split(ep_idx, seed, train, val, test)
        counts[split] += 1

        # write parquet with split/chunk columns in the path
        part_dir = out_dir / "split={}".format(split) / "chunk={}".format(chunk_id)
        part_dir.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(fp).with_columns([
            pl.lit(split).alias("split"),
            pl.lit(chunk_id).alias("chunk"),
        ])
        out_parquet = part_dir / "episode_{:06d}.parquet".format(ep_idx)
        df.write_parquet(out_parquet, compression="zstd")

        # videos: resolve src and create link/copy as requested
        video_paths_out: Dict[str, Optional[str]] = {}
        for view in views:
            src = None
            if videos_root is not None:
                # videos/chunk-<video_source_chunk_id>/<view>/episode_xxxxxx.mp4
                candidate = Path(videos_root) / "chunk-{}".format(video_source_chunk_id) / view / "episode_{:06d}.mp4".format(ep_idx)
                if candidate.exists():
                    src = candidate

            dst = out_dir / "split={}".format(split) / "chunk={}".format(chunk_id) / "videos" / view / "episode_{:06d}.mp4".format(ep_idx)
            if src is not None:
                _link(src, dst, link_videos)
                video_paths_out[view] = str(dst.relative_to(out_dir))
            else:
                video_paths_out[view] = None  # absent

        index_rows.append({
            "episode_index": ep_idx,
            "split": split,
            "chunk": chunk_id,
            "parquet_path": str(out_parquet.relative_to(out_dir)),
            **{"{}.path".format(v): video_paths_out.get(v) for v in views},
            "num_rows": df.height,
        })

    # write dataset index
    pl.DataFrame(index_rows).write_parquet(out_dir / "dataset_index.parquet")

    manifest = {
        "source_parquet": str(Path(norm_dir)),
        "source_videos": str(videos_root) if videos_root else None,
        "output": str(out_dir),
        "seed": seed,
        "fractions": {"train": train, "val": val, "test": test},
        "counts": counts,
        "chunk": chunk_id,
        "views": views,
        "link_videos": link_videos,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
