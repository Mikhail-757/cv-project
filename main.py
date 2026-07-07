import argparse
from pathlib import Path

from src.utils.utils import load_config, set_seed, cleanup_gpu, save_results, print_metrics
from src.dataset.prepare_data import download_coco, filter_animal_annotations, prepare_yolo_data, get_dataset_info
from src.training.train import train_yolo_pipeline, train_torchvision_model
from src.training.logger import TrainingLogger
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import (
    generate_all_plots, plot_comparison, plot_comparison_table, plot_dataset_stats,
)


ALL_MODELS = ["yolov8", "faster_rcnn", "ssd", "retinanet", "detr"]
TORCHVISION_MODELS = ["faster_rcnn", "ssd", "retinanet", "detr"]


def prepare_data(config):
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    animal_classes = config["data"]["animal_classes"]
    year = config["data"]["coco_year"]

    print("Downloading COCO dataset...")
    download_coco(raw_dir, year)

    print("Filtering animal annotations...")
    filter_animal_annotations(raw_dir, processed_dir, animal_classes, year)

    yolo_dir = Path(processed_dir) / "yolo"
    data_yaml, num_classes, class_names = prepare_yolo_data(processed_dir, yolo_dir)

    print(f"Dataset prepared: {num_classes} classes - {class_names}")
    return data_yaml


def train_single_model(model_name, config, data_yaml=None):
    results_dir = Path(config["results"]["output_dir"]) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    num_classes, class_names = get_dataset_info(config["data"]["processed_dir"])

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}\n")

    if model_name == "yolov8":
        if data_yaml is None:
            yolo_dir = Path(config["data"]["processed_dir"]) / "yolo"
            data_yaml = str(yolo_dir / "data.yaml")
        model, metrics = train_yolo_pipeline(config, data_yaml, results_dir)
    else:
        model, history = train_torchvision_model(model_name, config, results_dir)

        from src.utils.utils import get_device
        device = get_device()
        from src.dataset.dataset import create_dataloaders
        _, val_loader = create_dataloaders(config, model_name)

        metrics, predictions, ground_truths = evaluate_model(
            model, val_loader, num_classes, device, model_name
        )

        logger = TrainingLogger(results_dir, model_name, config)
        logger.log_metrics(metrics)

    print_metrics(metrics, model_name)
    generate_all_plots(results_dir, model_name, metrics, class_names)
    save_results({"model": model_name, "metrics": metrics}, results_dir / f"{model_name}_results.json")

    cleanup_gpu()
    return metrics


def train_all_models(config):
    yolo_dir = Path(config["data"]["processed_dir"]) / "yolo"
    data_yaml = str(yolo_dir / "data.yaml")

    all_results = {}
    for model_name in ALL_MODELS:
        metrics = train_single_model(model_name, config, data_yaml)
        all_results[model_name] = metrics
        cleanup_gpu()

    results_dir = Path(config["results"]["output_dir"])
    plot_comparison(all_results, results_dir / "plots" / "comparison.png")
    plot_comparison_table(all_results, results_dir / "plots" / "comparison_table.png")

    logger = TrainingLogger(results_dir / "logs", "all", config)
    logger.log_comparison(all_results)
    save_results(all_results, results_dir / "logs" / "all_results.json")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="all", choices=ALL_MODELS + ["all"])
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    if args.prepare_data:
        data_yaml = prepare_data(config)

    num_classes, class_names = get_dataset_info(config["data"]["processed_dir"])
    results_dir = Path(config["results"]["output_dir"])
    plot_dataset_stats(config["data"]["processed_dir"], class_names, results_dir / "plots" / "dataset_stats.png")

    if args.model == "all":
        all_results = train_all_models(config)
    else:
        yolo_dir = Path(config["data"]["processed_dir"]) / "yolo"
        data_yaml = str(yolo_dir / "data.yaml") if args.model == "yolov8" else None
        train_single_model(args.model, config, data_yaml)


if __name__ == "__main__":
    main()
