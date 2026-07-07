import csv
import json
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    def __init__(self, output_dir, model_name, config):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        self.csv_path = self.output_dir / "training_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    def log_epoch(self, epoch, train_loss, val_loss):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])

    def log_metrics(self, metrics):
        metrics_path = self.output_dir / f"{self.model_name}_metrics.json"
        data = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=2, default=_serializable)

    def log_comparison(self, all_results):
        comp_path = self.output_dir / "comparison.csv"
        if not all_results:
            return

        keys = list(all_results[list(all_results.keys())[0]].keys())
        keys = [k for k in keys if k != "per_class"]

        with open(comp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + keys)
            for model_name, metrics in all_results.items():
                row = [model_name] + [f"{metrics.get(k, 0):.4f}" for k in keys]
                writer.writerow(row)


def _serializable(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
