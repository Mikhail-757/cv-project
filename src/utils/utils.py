import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path="configs/default.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(to_serializable(results), f, indent=2)


def print_metrics(metrics, model_name="model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    for key, val in metrics.items():
        if key == "per_class_ap":
            continue
        print(f"  {key:>15s}: {val:.4f}")
    if "per_class_ap" in metrics:
        print(f"\n  Per-class AP:")
        for cls, ap in metrics["per_class_ap"].items():
            print(f"    {str(cls):>15s}: {ap:.4f}")
    print(f"{'='*50}\n")


def get_gpu_memory():
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }
