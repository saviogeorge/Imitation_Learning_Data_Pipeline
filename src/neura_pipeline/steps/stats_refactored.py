
from pathlib import Path
from typing import Dict, List, Optional, Set, Iterable, Any
import json
import math


# --------- numeric reducers (dimension-wise, weighted by frame count) ---------
def _init_acc() -> Dict[str, Any]:
    return {"S": 0, "sum_mu": None, "sum_m2": None, "min": None, "max": None}


def _acc(
    acc: Dict[str, Any],
    n: int,
    mean: List[float],
    std: List[float],
    mi: List[float],
    ma: List[float],
) -> None:
    D = len(mean)
    if acc["sum_mu"] is None:
        acc["sum_mu"] = [0.0] * D
        acc["sum_m2"] = [0.0] * D
        acc["min"] = [float("inf")] * D
        acc["max"] = [float("-inf")] * D
    acc["S"] += n
    for d in range(D):
        mu = float(mean[d])
        sd = float(std[d])
        acc["sum_mu"][d] += n * mu
        acc["sum_m2"][d] += n * (sd * sd + mu * mu)
        acc["min"][d] = min(acc["min"][d], float(mi[d]))
        acc["max"][d] = max(acc["max"][d], float(ma[d]))


def _finalize(acc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if acc["S"] == 0:
        return None
    S = float(acc["S"])
    mean = [s / S for s in acc["sum_mu"]]
    var = [max(m2 / S - m * m, 0.0) for m2, m in zip(acc["sum_m2"], mean)]
    std = [math.sqrt(v) for v in var]
    return {"count": int(S), "mean": mean, "std": std, "min": acc["min"], "max": acc["max"]}


# --------- helpers to ingest validated ids & flexible per-episode stats ---------
def load_valid_ids(path: Optional[str]) -> Optional[Set[int]]:
    """Load a set of episode indices from JSON/JSONL/CSV/lines; tolerant to shapes."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None

    ids: Set[int] = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                # JSON object per line
                try:
                    obj = json.loads(line)
                    if "episode_index" in obj:
                        ids.add(int(obj["episode_index"]))
                    elif "episode" in obj:
                        ids.add(int(obj["episode"]))
                except Exception:
                    pass
            else:
                # CSV or plain number at last position
                try:
                    ids.add(int(line.split(",")[-1]))
                except Exception:
                    pass
    return ids


def _as_float_list(x: Any) -> Optional[List[float]]:
    """Coerce a numeric or list-like value into a float list; return None if impossible."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _extract_count(feature_stats: Dict[str, Any]) -> Optional[int]:
    """
    Robustly extract the frame count for a feature.
    Common shapes seen:
      - {"count": 1000}
      - {"count": [1000, 1000, ...]}
      - {"frames": 1000} or {"frame_count": 1000}
    """
    if not isinstance(feature_stats, dict):
        return None
    c = feature_stats.get("count")
    if isinstance(c, (int, float)):
        return int(c)
    if isinstance(c, (list, tuple)) and len(c) > 0:
        try:
            return int(c[0])
        except Exception:
            pass
    # fallbacks occasionally used by other toolchains
    for k in ("frame_count", "frames", "count_total"):
        v = feature_stats.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _extract_vectors(feature_stats: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """Extract mean/std/min/max lists from a feature stats dict; return None if any missing."""
    if not isinstance(feature_stats, dict):
        return None
    mean = _as_float_list(feature_stats.get("mean"))
    std = _as_float_list(feature_stats.get("std"))
    mi = _as_float_list(feature_stats.get("min"))
    ma = _as_float_list(feature_stats.get("max"))
    if any(v is None for v in (mean, std, mi, ma)):
        return None
    # dimension guard: all vectors must align
    D = len(mean)  # type: ignore[arg-type]
    if not all(len(vec) == D for vec in (std, mi, ma)):  # type: ignore[arg-type]
        return None
    return {"mean": mean, "std": std, "min": mi, "max": ma}  # type: ignore[return-value]


# --------- public API ---------
def reduce_global_stats(
    episodes_stats_path: str,
    features: Iterable[str],
    validated_ids_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reduce per-episode feature stats into global stats (mean/std/min/max),
    weighted by per-episode frame counts.

    Input:
      - episodes_stats.jsonl: one JSON per line with shape like:
        {"episode_index": 123, "stats": {"action": {"count":[1000,...], "mean":[...], "std":[...], "min":[...], "max":[...]}, ...}}
      - validated_ids_path (optional): a JSONL/CSV/lines list containing episode_index values to include.

    Output:
      - {"meta": {...}, "action": {...}, "observation.state": {...}, ...}
    """
    eps = Path(episodes_stats_path)
    if not eps.exists():
        raise FileNotFoundError(f"episodes_stats.jsonl not found: {eps}")

    feature_keys = list(features)
    accs: Dict[str, Dict[str, Any]] = {k: _init_acc() for k in feature_keys}
    valid_ids = load_valid_ids(validated_ids_path)

    episodes_used = 0
    total_frames = 0

    with eps.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ep = int(row.get("episode_index"))
            if valid_ids is not None and ep not in valid_ids:
                continue

            st = row.get("stats") or {}
            # the reference feature used to get 'n' (frame count)
            ref = st.get("action") or st.get("observation.state") or {}

            n = _extract_count(ref)
            if not n or n <= 0:
                # if ref missing, try any feature present
                for fk in feature_keys:
                    cand = _extract_count(st.get(fk, {}))
                    if cand and cand > 0:
                        n = cand
                        break
            if not n or n <= 0:
                # skip episodes with zero/unknown frames
                continue

            episodes_used += 1
            total_frames += int(n)

            for key in feature_keys:
                s = st.get(key)
                vecs = _extract_vectors(s) if s else None
                if not vecs:
                    # feature missing for this episode: skip accumulating it
                    continue
                _acc(accs[key], int(n), vecs["mean"], vecs["std"], vecs["min"], vecs["max"])

    out: Dict[str, Any] = {
        "meta": {
            "episodes_used": episodes_used,
            "total_frames": total_frames,
            "source": str(eps),
            "features": feature_keys,
            "note": "Weighted reduction over per-episode means/stds (streaming).",
        }
    }
    for k, a in accs.items():
        fin = _finalize(a)
        if fin:
            out[k] = fin
    return out
