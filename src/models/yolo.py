from pathlib import Path
from ultralytics import YOLO


def create_yolo_model(config):
    variant = config["models"]["yolov8"]["variant"]
    model = YOLO(f"{variant}.pt")
    return model


def train_yolo(model, data_yaml, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config["training"]
    model_cfg = config["models"]["yolov8"]

    results = model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        imgsz=model_cfg["input_size"],
        optimizer="Adam",
        lr0=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        project=str(output_dir),
        name="yolov8_train",
        exist_ok=True,
        device=0,
        workers=cfg["num_workers"],
        patience=cfg["early_stopping_patience"],
        save=True,
        plots=True,
        verbose=True,
    )
    return results


def evaluate_yolo(model, data_yaml, config, output_dir):
    output_dir = Path(output_dir)
    model_cfg = config["models"]["yolov8"]

    metrics = model.val(
        data=data_yaml,
        imgsz=model_cfg["input_size"],
        batch=config["training"]["batch_size"],
        project=str(output_dir),
        name="yolov8_eval",
        exist_ok=True,
        device=0,
    )

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    per_class = {}
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = metrics.names[int(cls_idx)]
            per_class[cls_name] = {
                "ap50": float(metrics.box.ap50[i]),
                "ap": float(metrics.box.ap[i]),
            }
    results["per_class"] = per_class

    return results
