# Robot Dataset: Cube to Box Pick-and-Place Task

## Overview

This dataset contains 102 episodes of robot demonstrations performing a pick-and-place task: **"Grab cube and place into box"**. The data was collected using a `7 DoF robot` robot with synchronized visual observations and robot state/action recordings at 30 FPS.

## Dataset Statistics

- **Total Episodes**: 102
- **Robot Type**: 7-DOF manipulator + gripper
- **Task**: Single task - "Grab cube and place into box"
- **Camera Views**: Dual cameras (front view + wrist view)

## Directory Structure

```
robot-dataset/
├── meta/                          # Metadata and episode information
│   ├── info.json                  # Dataset configuration and feature schemas
│   ├── episodes.jsonl             # Episode metadata (index, tasks, length)
│   ├── episodes_stats.jsonl       # Statistical analysis per episode
│   ├── tasks.jsonl                # Task definitions
│   ├── stats.json                 # Overall dataset statistics
│   └── modality.json              # Data modality information
├── data/                          # Robot sensor data (parquet format)
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ... (102 episodes total)
└── videos/                        # Synchronized video observations
    └── chunk-000/
        ├── observation.images.front/
        │   ├── episode_000000.mp4
        │   ├── episode_000001.mp4
        │   └── ... (102 episodes total)
        └── observation.images.wrist/
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            └── ... (102 episodes total)
```

## Metadata Files (`meta/`)

### `episodes.jsonl` 
Per-episode metadata (one JSON object per line):
- **episode_index**: Episode ID (0 based index)
- **tasks**: Task description list (always ["Grab cube and place into box"])
- **length**: Number of frames in episode

**Example:**
```json
{"episode_index": 0, "tasks": ["Grab cube and place into box"], "length": 599}
```

### `episodes_stats.jsonl`
Comprehensive statistical analysis for each episode (one JSON per line):
- **Per-field statistics**: min, max, mean, std, count for all data columns
- **Joint-level analysis**: Statistics for each of the 6 robot joints
- **Image statistics**: RGB channel statistics (normalized 0-1 values)
- **Temporal data**: Frame timing and indexing statistics

**Structure:**
```json
{
  "episode_index": 0,
  "stats": {
    "action": {
      "min": [-6.16, -98.12, -12.46, 44.90, -6.76, 0.0],
      "max": [54.88, 2.86, 99.91, 98.87, 4.32, 51.58],
      "mean": [12.58, -69.76, 67.50, 64.17, 0.62, 12.44],
      "std": [21.91, 34.56, 34.33, 12.57, 2.34, 17.54],
      "q01": [-40.89, -30.73, 0.0, 70.99, 0.0, 5.28, -8.83, 0.54],
      "q99": [ 14.09, 31.13, 0.0, 140.76, 0.0, 108.43, 27.92, 44.88],
    },
    "observation.state": {...},
    "observation.images.front": {...}
  }
}
```

### `tasks.jsonl`
Task definitions (currently single task):
- **task_index**: Task identifier (0)
- **task**: Natural language task description

**Content:**
```json
{"task_index": 0, "task": "Grab cube and place into box"}
```

## Data Format

### Robot Data (Parquet Files)

Each episode is stored as a parquet file containing timestamped robot observations and actions:

**Columns (7 total):**
- `action` - Robot joint commands. ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]
- `observation.state` - Current robot joint positions. ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]
- `timestamp` - Time elapsed in seconds
- `frame_index` - Frame number within episode (0-indexed)
- `episode_index` - Episode identifier (0-indexed)
- `index` - Global frame index across all episodes
- `task_index` - Task identifier (always 0 for this single-task dataset)

**Joint Configuration (7-DOF):**
1. `joint1.pos` 
2. `joint2.pos` 
3. `joint3.pos` 
4. `joint4.pos` 
5. `joint5.pos` 
6. `joint6.pos` 
7. `joint7.pos` 
8. `finger_joint.pos`

### Video Data (MP4 Files)

- **Format**: AV1 codec, YUV420P pixel format
- **Resolution**: 640×480 pixels
- **Channels**: 3 (RGB)
- **Frame Rate**: 30 FPS
- **View**: Front-facing camera perspective

## Quick Start

### Loading Episode Data

```python
import pandas as pd

# Load robot data for episode 0
df = pd.read_parquet('data/chunk-000/episode_000000.parquet')
print(f"Episode shape: {df.shape}")
print(f"Duration: {df['timestamp'].max():.2f} seconds")

# Extract robot actions and states
actions = df['action'].tolist()  # List of 6D arrays
states = df['observation.state'].tolist()  # List of 6D arrays
```

### Loading Video Data

```python
import cv2

# Load synchronized video
cap = cv2.VideoCapture('videos/chunk-000/observation.images.front/episode_000000.mp4')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
```

## Data Quality Notes

- All episodes demonstrate the same task with natural variation in execution
- Camera provides consistent front-view perspective of the workspace
- Synchronized timing between robot data and video frames (reference)

## File Access Patterns

- **Individual Episodes**: `data/chunk-000/episode_{episode_index:06d}.parquet`
- **Videos**: `videos/chunk-000/observation.images.front/episode_{episode_index:06d}.mp4`
- **Metadata**: `meta/*.jsonl` files contain episode-level information

## Dependencies

```bash
pip install pandas pyarrow opencv-python
```