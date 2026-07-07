import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(log_csv, output_path, model_name="model"):
    epochs, train_losses, val_losses = [], [], []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Train Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name} - Val Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined_loss(log_csv, output_path, model_name="model"):
    epochs, train_losses, val_losses = [], [], []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    plt.plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(metrics_dict, output_path, title="Model Metrics"):
    keys = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    values = [metrics_dict.get(k, 0) for k in keys]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 1.1)
    plt.title(title)
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_ap(per_class_ap, class_names, output_path, model_name="model"):
    classes = []
    aps = []
    for cls_id, ap_val in sorted(per_class_ap.items()):
        idx = int(cls_id) - 1 if isinstance(cls_id, (int, float)) else int(cls_id) - 1
        if idx < len(class_names):
            classes.append(class_names[idx])
        else:
            classes.append(f"Class {cls_id}")
        aps.append(ap_val)

    plt.figure(figsize=(12, 6))
    bars = plt.barh(classes, aps, color="#2196F3")
    for bar, val in zip(bars, aps):
        plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10)
    plt.xlim(0, 1.1)
    plt.xlabel("AP@0.5")
    plt.title(f"{model_name} - Per-Class AP")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(all_results, output_path):
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_table(all_results, output_path):
    models = list(all_results.keys())
    metrics = ["mAP50", "mAP50-95", "precision", "recall", "f1"]

    cell_text = []
    for model in models:
        row = [f"{all_results[model].get(m, 0):.4f}" for m in metrics]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(12, 2 + len(models) * 0.5))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=metrics,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    plt.title("Model Comparison Table", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_detection_examples(images, predictions, ground_truths, class_names, output_path, num_samples=4):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    n = min(num_samples, len(images))
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        for row, (data, title) in enumerate([(ground_truths[i], "Ground Truth"), (predictions[i], "Prediction")]):
            axes[row, i].imshow(img)
            axes[row, i].set_title(title)
            axes[row, i].axis("off")

            boxes = data["boxes"].cpu().numpy()
            labels = data["labels"].cpu().numpy()

            for box, lbl in zip(boxes, labels):
                x1, y1, x2, y2 = box
                idx = int(lbl) - 1
                name = class_names[idx] if 0 <= idx < len(class_names) else f"cls_{lbl}"
                color = plt.cm.tab10(idx % 10)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=color, facecolor="none")
                axes[row, i].add_patch(rect)
                axes[row, i].text(x1, y1 - 5, name, fontsize=8, color=color,
                                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_stats(processed_dir, class_names, output_path):
    processed_dir = Path(processed_dir)
    stats = {}

    for split in ["train", "val"]:
        ann_file = processed_dir / split / "annotations.json"
        if not ann_file.exists():
            continue
        with open(ann_file) as f:
            data = json.load(f)

        counts = {}
        for ann in data["annotations"]:
            cls_id = ann["category_id"]
            idx = cls_id - 1
            name = class_names[idx] if idx < len(class_names) else f"cls_{cls_id}"
            counts[name] = counts.get(name, 0) + 1
        stats[split] = counts

    if not stats:
        return

    all_classes = sorted(set().union(*[s.keys() for s in stats.values()]))
    x = np.arange(len(all_classes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if "train" in stats:
        vals = [stats["train"].get(c, 0) for c in all_classes]
        ax1.barh(all_classes, vals, color="#2196F3")
        ax1.set_title("Train Set Distribution")
        ax1.set_xlabel("Count")

    if "val" in stats:
        vals = [stats["val"].get(c, 0) for c in all_classes]
        ax2.barh(all_classes, vals, color="#4CAF50")
        ax2.set_title("Val Set Distribution")
        ax2.set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(model_dir, model_name, metrics, class_names):
    model_dir = Path(model_dir)

    log_csv = model_dir / "training_log.csv"
    if log_csv.exists():
        plot_training_loss(log_csv, model_dir / "train_val_loss.png", model_name)
        plot_combined_loss(log_csv, model_dir / "training_curves.png", model_name)

    plot_metrics_bar(metrics, model_dir / "metrics_bar.png", f"{model_name} Metrics")

    if "per_class_ap" in metrics:
        plot_per_class_ap(metrics["per_class_ap"], class_names,
                          model_dir / "per_class_ap.png", model_name)
