from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .statuses import Status

@dataclass(frozen=True)
class EpisodeManifestRow:
    episode_index: int
    chunk: str
    parquet_uri: Optional[str]
    video_front_uri: Optional[str]
    video_wrist_uri: Optional[str]
    exists_front: bool
    exists_wrist: bool
    bytes_total: int
    fingerprint: Optional[str]
    fingerprint_algo: str
    discovered_at: str
    status: Status
    errors: Optional[str]



@dataclass(frozen=True)
class ValidationResult:
    episode_index: int
    rows: Optional[int]
    frame_min: Optional[int]
    frame_max: Optional[int]
    expected_rows_meta: Optional[int]
    ok: bool
    issues: List[Dict[str, Any]]
    # Traceability (filled by caller)
    chunk: Optional[str] = None
    parquet_uri: Optional[str] = None
    video_front_uri: Optional[str] = None
    video_wrist_uri: Optional[str] = None
