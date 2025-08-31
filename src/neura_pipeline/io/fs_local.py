from pathlib import Path
from typing import Iterable, Union
from .fs_base import FSBase

class LocalFS(FSBase):
    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)

    def list_chunks(self) -> Iterable[str]:
        data_dir = self.root / "data"
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            yield chunk_dir.name.split("-")[-1]

    def list_parquets(self, chunk: str):
        base = self.root / "data" / f"chunk-{chunk}"
        for p in sorted(base.glob("episode_*.parquet")):
            yield p

    def video_path(self, chunk: str, cam: str, ep_idx: int) -> Path:
        return self.root / "videos" / f"chunk-{chunk}" / cam / f"episode_{ep_idx:06d}.mp4"