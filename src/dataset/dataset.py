import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors

from src.dataset.augmentations import TrainTransform, ValTransform


class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        with open(ann_file) as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.img_ids = list(self.images.keys())

        self.annotations = {}
        for ann in data["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        anns = self.annotations.get(img_id, [])

        boxes = []
        labels = []
        areas = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)

        w_img, h_img = img.size
        boxes = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(h_img, w_img)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataloaders(config, model_name=None):
    processed_dir = Path(config["data"]["processed_dir"])

    if model_name and model_name in config["models"]:
        model_cfg = config["models"][model_name]
        input_size = model_cfg.get("input_size", config["image"]["input_size"])
        cfg_override = dict(config)
        cfg_override["image"] = dict(config["image"])
        cfg_override["image"]["input_size"] = input_size
    else:
        cfg_override = config

    train_transform = TrainTransform(cfg_override)
    val_transform = ValTransform(cfg_override)

    train_ds = COCODetectionDataset(
        img_dir=processed_dir / "train" / "images",
        ann_file=processed_dir / "train" / "annotations.json",
        transform=train_transform,
    )

    val_ds = COCODetectionDataset(
        img_dir=processed_dir / "val" / "images",
        ann_file=processed_dir / "val" / "annotations.json",
        transform=val_transform,
    )

    bs = config["training"]["batch_size"]
    nw = config["training"]["num_workers"]
    pin = config["training"]["pin_memory"]

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin, collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin, collate_fn=collate_fn,
    )

    return train_loader, val_loader
