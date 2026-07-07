from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.yolo import create_yolo_model, train_yolo, evaluate_yolo
from src.models.faster_rcnn import create_faster_rcnn, get_faster_rcnn_params
from src.models.ssd import create_ssd, get_ssd_params
from src.models.retinanet import create_retinanet, get_retinanet_params
from src.models.detr import create_detr, get_detr_params
from src.dataset.dataset import create_dataloaders
from src.training.logger import TrainingLogger


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_optimizer(params, config):
    cfg = config["training"]
    opt_name = cfg["optimizer"].lower()
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    return torch.optim.SGD(params, lr=cfg["learning_rate"], momentum=0.9, weight_decay=cfg["weight_decay"])


def _create_scheduler(optimizer, config):
    return CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])


def train_yolo_pipeline(config, data_yaml, output_dir):
    output_dir = Path(output_dir)
    model = create_yolo_model(config)
    train_yolo(model, data_yaml, config, output_dir)

    best_path = output_dir / "yolov8_train" / "weights" / "best.pt"
    if best_path.exists():
        model = __import__("ultralytics").YOLO(str(best_path))

    results = evaluate_yolo(model, data_yaml, config, output_dir)
    return model, results


def train_torchvision_model(model_name, config, output_dir):
    from src.dataset.prepare_data import get_dataset_info

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    num_classes, class_names = get_dataset_info(config["data"]["processed_dir"])

    if model_name == "faster_rcnn":
        model = create_faster_rcnn(num_classes, config)
        params = get_faster_rcnn_params(model, config)
    elif model_name == "ssd":
        model = create_ssd(num_classes, config)
        params = get_ssd_params(model, config)
    elif model_name == "retinanet":
        model = create_retinanet(num_classes, config)
        params = get_retinanet_params(model, config)
    elif model_name == "detr":
        model = create_detr(num_classes, config)
        params = get_detr_params(model, config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    optimizer = _create_optimizer(params, config)
    scheduler = _create_scheduler(optimizer, config)

    train_loader, val_loader = create_dataloaders(config, model_name)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = TrainingLogger(output_dir, model_name, config)

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)

                if isinstance(loss_dict, dict):
                    losses = sum(loss_dict.values())
                else:
                    losses = loss_dict

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += losses.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        scheduler.step()
        avg_train = train_loss / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    if model_name == "detr":
                        model.train()
                        loss_dict = model(images, targets)
                        model.eval()
                    else:
                        loss_dict = model(images, targets)

                    if isinstance(loss_dict, dict):
                        losses = sum(loss_dict.values())
                    else:
                        losses = loss_dict

                val_loss += losses.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        logger.log_epoch(epoch + 1, avg_train, avg_val)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train:.4f}, val_loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    best_weights = output_dir / "best.pt"
    if best_weights.exists():
        model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))

    return model, history
