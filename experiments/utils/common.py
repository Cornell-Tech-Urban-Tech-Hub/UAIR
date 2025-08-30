import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_structured(path: str) -> Any:
    try:
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {path}: {e}")


def start_wandb_heartbeat(wandb, ray, stage_name: str, interval_s: float = 15.0, dir_to_count: Optional[str] = None):
    if not (wandb is not None and getattr(wandb, "run", None) is not None):
        return None, None
    import threading, time
    stop_event = threading.Event()
    start_ts = time.time()

    def _poll():
        while not stop_event.is_set():
            payload: Dict[str, Any] = {
                "heartbeat/stage": stage_name,
                "heartbeat/elapsed_s": time.time() - start_ts,
            }
            try:
                avail = ray.available_resources()
                total = ray.cluster_resources()
                payload["resources/available/CPU"] = float(avail.get("CPU", 0.0))
                payload["resources/available/GPU"] = float(avail.get("GPU", 0.0))
                payload["resources/total/CPU"] = float(total.get("CPU", 0.0))
                payload["resources/total/GPU"] = float(total.get("GPU", 0.0))
            except Exception:
                pass
            if dir_to_count:
                try:
                    files = list(Path(dir_to_count).glob("**/*.parquet"))
                    payload["progress/dir"] = dir_to_count
                    payload["progress/parquet_files"] = len(files)
                except Exception:
                    pass
            try:
                wandb.log(payload, commit=True)
            except Exception:
                pass
            stop_event.wait(interval_s)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    return stop_event, t
