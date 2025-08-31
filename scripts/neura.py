from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

import click
import polars as pl

# Allow running the CLI without installing the package (PYTHONPATH shim)
try:
    import neura_pipeline  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import pathlib as _pathlib, sys as _sys
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))

@click.group()
def cli():
    """Neura tooling CLI"""
    pass

# -------- discover --------
@cli.command()
@click.option("--data-root", type=click.Path(exists=True), required=True,
              help="Root folder containing data/ and videos/")
@click.option("--manifest", type=click.Path(), default="./output/manifest/episodes.parquet",
              help="Path to manifest parquet to write")
@click.option("--workers", type=int, default=16, show_default=True,
              help="Max worker threads for I/O-bound fingerprinting")
@click.option("--since", type=str, default=None,
              help="Only consider episodes whose parquet mtime >= this ISO8601 timestamp")
@click.option("--stdout", "stdout_jsonl", is_flag=True, default=False,
              help="Emit incremental set to stdout as JSONL")
@click.option("--full-hash", is_flag=True, default=False,
              help="Hash entire files instead of head/tail")
@click.option("--only-chunks", type=str, default=None,
              help="Comma-separated list of chunks to scan (e.g., 'chunk-000,chunk-003' or '000,003')")
@click.option("--all", "print_all", is_flag=True, default=False,
              help="Print full manifest snapshot (not just deltas)")
def discover(data_root, manifest, workers, since, stdout_jsonl, full_hash, only_chunks, print_all):
    """Incremental discovery: build/refresh manifest and print only the delta by default."""
    from neura_pipeline.steps.discover_refactored import discover_incremental

    since_ns = None
    if since:
        try:
            dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            since_ns = int(dt.timestamp() * 1e9)
        except Exception as e:
            raise click.BadParameter("--since must be ISO8601, e.g. 2024-05-17T12:34:56Z") from e

    chunks = None
    if only_chunks:
        chunks = set([x.strip() for x in only_chunks.split(",") if x.strip()])

    out = discover_incremental(
        data_root, manifest,
        workers=workers, since_ns=since_ns, full_hash=full_hash, only_chunks=chunks,
    )

    if print_all:
        full = pl.read_parquet(manifest)
        print(full.select(["chunk", "episode_index", "status", "parquet_uri"])
              .sort(["chunk", "episode_index"]))
        return

    if stdout_jsonl:
        cols = [
            "episode_index","chunk","parquet_uri","video_front_uri","video_wrist_uri",
            "exists_front","exists_wrist","bytes_total","fingerprint","fingerprint_algo",
            "discovered_at","status","errors"
        ]
        out = out.select([c for c in cols if c in out.columns])
        for row in out.iter_rows(named=True):
            sys.stdout.write(json.dumps(row, default=str) + "\n")
    else:
        print(out.select(["chunk","episode_index","status","parquet_uri"])
              .sort(["chunk","episode_index"]))

# -------- validate --------
@cli.command()
@click.option("--manifest", type=click.Path(exists=True), required=True,
              help="Path to manifest parquet from discover")
@click.option("--meta-dir", type=click.Path(), default="./robot_data/meta", show_default=True,
              help="Directory containing episodes.jsonl")
@click.option("--out", "out_dir", type=click.Path(), required=True,
              help="Output directory for validation reports")
@click.option("--fps", "fps_expected", type=float, default=30.0, show_default=True)
@click.option("--tolerance", "frame_tolerance", type=int, default=2, show_default=True)
@click.option("--skip-video", is_flag=True, default=False,
              help="Skip ffprobe checks (no video metadata checks)")
def validate(manifest, meta_dir, out_dir, fps_expected, frame_tolerance, skip_video):
    """Validate only the actionable episodes from the manifest (NEW/CHANGED/etc.)."""
    from neura_pipeline.steps.validate_from_manifest_refactored import (
        validate_from_manifest_refactored
    )

    summary = validate_from_manifest_refactored(
        Path(manifest),
        Path(meta_dir),
        Path(out_dir),
        fps_expected=fps_expected,
        frame_tolerance=frame_tolerance,
        skip_video=skip_video,
    )
    click.echo(f"total={summary['total']} ok={summary['ok']} fail={summary['fail']}")



# -------- Stats --------

@cli.command()
@click.option("--data-root", type=click.Path(exists=True), required=True,
              help="Root containing meta/episodes_stats.jsonl")
@click.option("--episodes-stats", type=click.Path(), default=None,
              help="Override path to episodes_stats.jsonl (defaults to <data-root>/meta/episodes_stats.jsonl)")
@click.option("--validated-ids", type=click.Path(), default=None,
              help="Optional JSON/JSONL/CSV list of episode_index produced by validate step")
@click.option("--features", type=str, default="action,observation.state", show_default=True,
              help="Comma-separated feature keys to reduce")
@click.option("--out", "out_path", type=click.Path(), default="./output/stats/global_stats.json",
              show_default=True)
def stats(data_root, episodes_stats, validated_ids, features, out_path):
    #Reduce per-episode stats -> global_stats.json for normalization.
    from neura_pipeline.steps.stats_refactored import reduce_global_stats

    data_root = Path(data_root)
    eps = Path(episodes_stats) if episodes_stats else (data_root / "meta" / "episodes_stats.jsonl")
    feature_keys = [k.strip() for k in features.split(",") if k.strip()]
    result = reduce_global_stats(str(eps), feature_keys, validated_ids)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    click.echo(f"[neura stats] wrote {out}")


# -------- align_transform --------

@cli.command("align-transform")
@click.option("--data-root", type=click.Path(exists=True), required=True,
              help="Root folder containing data/chunk-*/episode_*.parquet")
@click.option("--out", "out_dir", type=click.Path(), required=True,
              help="Output directory for cleaned/normalized episodes")
@click.option("--stats", "stats_path", type=click.Path(), default="./output/stats/global_stats.json",
              show_default=True, help="Path to global_stats.json (from 'stats' stage)")
@click.option("--no-normalize", is_flag=True, default=False, help="Disable z-score normalization")
def align_transform_cmd(data_root, out_dir, stats_path, no_normalize):
    # \"\"\"Align schema, reindex frames, (optionally) normalize vectors, and write cleaned parquets.\"\"\"
    from neura_pipeline.steps.align_transform_refactored import process_episodes
    from pathlib import Path as _P

    process_episodes(
        data_dir=_P(data_root) / "data",
        out_dir=_P(out_dir),
        stats_path=_P(stats_path) if stats_path else None,
        normalize=not no_normalize,
    )
    click.echo(f"[neura align-transform] wrote cleaned episodes to {out_dir}")


# -------- MAterialize --------

@cli.command("materialize")
@click.option("--norm-dir", type=click.Path(exists=True), required=True,
              help="Folder with normalized episodes (output of align-transform)")
@click.option("--out", "out_dir", type=click.Path(), required=True,
              help="Output dataset directory")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Deterministic seed for split assignment")
@click.option("--train", type=float, default=0.8, show_default=True)
@click.option("--val", type=float, default=0.1, show_default=True)
@click.option("--test", type=float, default=0.1, show_default=True)
@click.option("--chunk-id", type=str, default="000", show_default=True,
              help="Chunk id to embed under split=.../chunk=<id>")
@click.option("--videos-root", type=click.Path(exists=True), default=None,
              help="Root of videos/ (e.g., robot_data/videos)")
@click.option("--views", type=str, default=None,
              help="Comma-separated list of camera views (defaults to core.constants.CAMERAS)")
@click.option("--video-source-chunk-id", type=str, default="000", show_default=True,
              help="Which videos/chunk-<id> to read from")
@click.option("--link-videos", type=click.Choice(["symlink","hardlink","copy","manifest-only"]),
              default="symlink", show_default=True,
              help="How to place videos in the materialized dataset")
def materialize_cmd(norm_dir, out_dir, seed, train, val, test, chunk_id,
                    videos_root, views, video_source_chunk_id, link_videos):
    # \"\"\"Materialize a train/val/test dataset with optional video linking/copy.\"\"\"
    from neura_pipeline.steps.materialize_refactored import materialize_partitioned
    from pathlib import Path as _P

    views_list = None
    if views:
        views_list = [v.strip() for v in views.split(",") if v.strip()]

    manifest = materialize_partitioned(
        norm_dir=_P(norm_dir),
        out_dir=_P(out_dir),
        seed=seed,
        train=train, val=val, test=test,
        chunk_id=chunk_id,
        videos_root=_P(videos_root) if videos_root else None,
        views=views_list,
        video_source_chunk_id=video_source_chunk_id,
        link_videos=link_videos,
    )
    click.echo(f"[neura materialize] wrote dataset to {out_dir}")
    click.echo(json.dumps(manifest, indent=2))




if __name__ == "__main__":
    cli()



# #TO DO
# 1) The pipeline should only trigger if new data is added to the S3 bucket
# 2) Preemptive error handling and report generation at stages of the pipeline
# 3) Consider Docker