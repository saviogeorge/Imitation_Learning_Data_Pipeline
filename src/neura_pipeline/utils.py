
from __future__ import annotations
from pathlib import Path
from typing import Any, Union
from hashlib import blake2b
import json

BytesLike = Union[bytes, bytearray, memoryview]

def _to_bytes(x: Any) -> bytes:
    """Deterministically serialize common Python objects to bytes."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    if isinstance(x, (str, Path)):
        return str(x).encode("utf-8")
    if isinstance(x, (int, float, bool)) or x is None:
        return json.dumps(x, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # Fallback: JSON (stable with sort_keys and compact separators)
    return json.dumps(x, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")

def stable_hash_int(obj: Any, seed: int = 0, bits: int = 64) -> int:
    """
    Return a stable, cross-platform integer hash of `obj` in [0, 2^bits - 1].
    Uses BLAKE2b keyed hashing (seeded) so results are reproducible across runs.

    Why not built-in `hash()`? Because it’s randomi zed per process (PYTHONHASHSEED),
    so it’s not stable across runs or machines.
    """
    if bits <= 0 or bits % 8 != 0 or bits > 512:
        raise ValueError("bits must be a positive multiple of 8, <= 512")
    digest_size = bits // 8
    # Use the seed as the BLAKE2 key (up to 64 bytes). Mask to 64 bits then encode.
    seed_key = (seed & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)
    h = blake2b(digest_size=digest_size, key=seed_key)
    h.update(_to_bytes(obj))
    return int.from_bytes(h.digest(), "big", signed=False)

def hash_to_unit_interval(h: int, bits: int = 64) -> float:
    """Map an integer hash to [0, 1]."""
    return h / ((1 << bits) - 1)
