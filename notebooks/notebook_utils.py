import sys
import os
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_project(project_root=None):
    if project_root is None:
        if "google.colab" in sys.modules:
            project_root = "/content/drive/MyDrive/coco-animal-detection"
        else:
            project_root = str(Path(__file__).parent.parent)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)
    return project_root


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


def prepare_data(config):
    from src.dataset.prepare_data import download_coco, filter_animal_annotations, prepare_yolo_data
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    animal_classes = config["data"]["animal_classes"]
    year = config["data"]["coco_year"]

    download_coco(raw_dir, year)
    filter_animal_annotations(raw_dir, processed_dir, animal_classes, year)

    yolo_dir = Path(processed_dir) / "yolo"
    data_yaml, num_classes, class_names = prepare_yolo_data(processed_dir, yolo_dir)
    return data_yaml, num_classes, class_names


def train_model(model_name, config, data_yaml=None):
    from src.training.train import train_yolo_pipeline, train_torchvision_model
    from src.evaluation.metrics import evaluate_model
    from src.dataset.dataset import create_dataloaders
    from src.dataset.prepare_data import get_dataset_info

    results_dir = Path(config["results"]["output_dir"]) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    num_classes, class_names = get_dataset_info(config["data"]["processed_dir"])

    if model_name == "yolov8":
        if data_yaml is None:
            data_yaml = str(Path(config["data"]["processed_dir"]) / "yolo" / "data.yaml")
        model, metrics = train_yolo_pipeline(config, data_yaml, results_dir)
        history = None
    else:
        model, history = train_torchvision_model(model_name, config, results_dir)
        device = get_device()
        _, val_loader = create_dataloaders(config, model_name)
        metrics, _, _ = evaluate_model(model, val_loader, num_classes, device, model_name)

    return model, metrics, history


def evaluate_and_visualize(model, model_name, config):
    from src.evaluation.metrics import evaluate_model
    from src.evaluation.visualize import generate_all_plots
    from src.dataset.dataset import create_dataloaders
    from src.dataset.prepare_data import get_dataset_info

    device = get_device()
    num_classes, class_names = get_dataset_info(config["data"]["processed_dir"])
    _, val_loader = create_dataloaders(config, model_name)

    metrics, predictions, ground_truths = evaluate_model(
        model, val_loader, num_classes, device, model_name
    )

    results_dir = Path(config["results"]["output_dir"]) / model_name
    generate_all_plots(results_dir, model_name, metrics, class_names)

    return metrics, predictions, ground_truths


def plot_training_curves(history, model_name="model"):
    if history is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4)
    ax1.set_title(f"{model_name} - Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_loss"], "r-o", markersize=4)
    ax2.set_title(f"{model_name} - Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comparison(all_results):
    models = list(all_results.keys())
    metrics = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    plt.figure(figsize=(14, 7))
    for i, model in enumerate(models):
        values = [all_results[model].get(m, 0) for m in metrics]
        offset = (i - len(models) / 2) * width + width / 2
        plt.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])

    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_metrics(metrics, model_name="model"):
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    for key, val in metrics.items():
        if key == "per_class_ap":
            continue
        print(f"  {key:>15s}: {val:.4f}")
    if "per_class_ap" in metrics:
        print(f"\n  Per-class AP:")
        for cls, ap in metrics["per_class_ap"].items():
            print(f"    {str(cls):>15s}: {ap:.4f}")
    print()


def save_results(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)
