from pathlib import Path
from hashlib import sha256
import json, time
from ..core.constants import SAMPLE_BYTES, STABILITY_MIN_BYTES, STABILITY_PAUSE_S, FINGERPRINT_ALGO

def stable_sleep_check(fp: Path, min_bytes: int = STABILITY_MIN_BYTES, pause_s: float = STABILITY_PAUSE_S) -> bool:
    try:
        st1 = fp.stat()
    except FileNotFoundError:
        return False
    if st1.st_size < min_bytes:
        return True
    time.sleep(pause_s)
    try:
        st2 = fp.stat()
    except FileNotFoundError:
        return False
    return (st1.st_size == st2.st_size) and (st1.st_mtime_ns == st2.st_mtime_ns)

def quick_file_fingerprint(fp: Path, sample_bytes: int = SAMPLE_BYTES, full_hash: bool = False) -> dict:
    st = fp.stat()
    size, mtime_ns = st.st_size, st.st_mtime_ns
    h = sha256()
    with open(fp, "rb") as f:
        if full_hash:
            while True:
                b = f.read(1024 * 1024)
                if not b: break
                h.update(b)
        else:
            head = f.read(sample_bytes); h.update(head)
            if size > sample_bytes:
                f.seek(max(0, size - sample_bytes))
                tail = f.read(sample_bytes); h.update(tail)
    return {"size": size, "mtime_ns": mtime_ns, "sha": h.hexdigest()}

def combine_episode_fingerprint(parts: dict) -> str:
    return sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()