from pathlib import Path
from typing import Iterable

class FSBase:
    def list_chunks(self) -> Iterable[str]:
        raise NotImplementedError
    def list_parquets(self, chunk: str):
        raise NotImplementedError
    def video_path(self, chunk: str, cam: str, ep_idx: int) -> Path:
        raise NotImplementedError
    def stat(self, p: Path):
        return p.stat()