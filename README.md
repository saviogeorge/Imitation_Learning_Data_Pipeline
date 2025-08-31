# Imitation Learning Data Pipeline

This repository implements a reproducible, containerized **data pipeline** to transform the provided robot dataset (cube → box pick-and-place) into a clean, training-ready dataset for imitation learning.

The pipeline has **five stages**, each exposed as a CLI subcommand in `scripts/neura.py`:

1. **Discover** – build/update an episode manifest with fingerprints.  
2. **Validate** – check schema, vector widths, nulls, ordering, video metadata.  
3. **Stats** – reduce per-episode stats into global mean/std/min/max.  
4. **Align & Transform** – clean schema, enforce widths, reindex, normalize.  
5. **Materialize** – package into train/val/test splits with index + manifest.

Execution is reproducible via **Docker** and chainable via **Makefile**.

---

---

## Quick Start

### 1. Build the pipeline images
```bash
docker build -t neura:base -f Dockerfile .
docker build -t neura:media -f Dockerfile.media .
```

neura:base → used for most stages

neura:media → includes ffprobe for video checks in validate

### 2. Run the full pipeline
```bash
make pipeline
```


### 3. Run stages individually
```bash
make discover
make validate
make stats
make align
make materialize
```

## Pipeline Stages
### 1. Discover

Incrementally scans the dataset (data/ + videos/) and builds/updates a manifest of episodes with fingerprints.

* Input:

     * robot_data/data/chunk-*/episode_*.parquet

     * robot_data/videos/chunk-*/observation.images.
       {front,wrist}/episode_*.mp4

* Output:

    *   output/manifest/episodes.parquet
        
        Columns: episode_index, chunk, parquet_uri, video_front_uri, video_wrist_uri, fingerprint, status


### 2. Validate

Checks schema, nulls, frame order, vector widths, and optionally video FPS/frame counts.

* Input:

    *   output/manifest/episodes.parquet

    *   robot_data/meta/episodes.jsonl

* Output:

    *   output/validation/episodes.parquet — all validation results

    *   output/validation/failures.jsonl — failing episodes with    issues

    *   output/validation/validated_episodes.jsonl — passing episodes only

    *   output/validation/summary.yaml — summary counts

### 3. Stats

Reduces per-episode statistics into global stats for normalization.
Tolerates heterogeneous shapes (count: int vs count: [int,…]) and missing features.

* Input:

    *   robot_data/meta/episodes_stats.jsonl

    *   output/validation/validated_episodes.jsonl

* Output:

    *   output/stats/global_stats.json

### 4. Align & Transform

Cleans and normalizes each episode parquet:

Enforces schema & vector widths

Sorts/deduplicates frames

Reindexes frame_index

Drops nulls

Z-score normalizes action and observation.state using global stats

* Input:

    *   robot_data/data/chunk-*/episode_*.parquet

    *   output/stats/global_stats.json

* Output:

    *   output/normalized/episode_*.parquet

### 5. Materialize

Packages the cleaned dataset into train/val/test splits with optional video symlinks or copies.

* Input:

    *   output/normalized/episode_*.parquet

    *   robot_data/videos/...

* Output:
```bash
output/dataset/
├── split=train/chunk=000/episode_000000.parquet
├── split=val/chunk=000/...
├── split=test/chunk=000/...
├── dataset_index.parquet
└── _manifest.json
```

## Outputs Overview

 After running the pipeline:

```bash
output/
├── manifest/episodes.parquet
├── validation/{episodes.parquet,failures.jsonl,validated_episodes.jsonl,summary.yaml}
├── stats/global_stats.json
├── normalized/episode_*.parquet
└── dataset/
    ├── split={train,val,test}/chunk=000/episode_*.parquet
    ├── dataset_index.parquet
    └── _manifest.json
```


## TODO / Future Work


Airflow orchestration: DAG to orchestrate all stages (discover → validate → stats → align → materialize) with parallelism & cloud triggers. (ongoing)

CI/CD integration: build & push images, run tests on toy dataset, run full pipeline in CI.

Extended modality support: add new camera views or features via core/constants.py.