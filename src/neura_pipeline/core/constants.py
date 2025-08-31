SAMPLE_BYTES = 65536  # head/tail for quick fingerprint
STABILITY_MIN_BYTES = 50 * 1024 * 1024  # run stability check for files >= 50MB
STABILITY_PAUSE_S = 0.15  # seconds
FINGERPRINT_ALGO = "size+mtime+sha(head|tail)-v1"
CAMERAS = ["observation.images.front", "observation.images.wrist"]

# Column names (keep dotted raw name to match parquet schema)
RAW_OBS_COL = "observation.state"
OBS_SAFE_COL = "observation_state"  # internal safe name when working in Polars
ACTION_COL = "action"

# Minimum required columns for an episode parquet (validated earlier, re-checked here)
REQUIRED_EPISODE_COLS = [
    ACTION_COL, RAW_OBS_COL,
    "timestamp", "frame_index", "episode_index", "index", "task_index",
]

# Robot state/action dimension (7 DOF + gripper)
LIST_WIDTH = 8

# # --- Camera views (short & dotted) ---
# CAMERA_SHORT = ["front", "wrist"]
# CAMERA_FULL  = ["observation.images.front", "observation.images.wrist"]

# # Back-compat alias used around the codebase
# CAMERAS = CAMERA_FULL

# CAMERA_MAP_SHORT2FULL = dict(zip(CAMERA_SHORT, CAMERA_FULL))
# CAMERA_MAP_FULL2SHORT = {v: k for k, v in CAMERA_MAP_SHORT2FULL.items()}
