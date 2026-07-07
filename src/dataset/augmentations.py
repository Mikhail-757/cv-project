import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors


class TrainTransform:
    def __init__(self, config):
        img_cfg = config["image"]
        aug_cfg = config["augmentations"]
        size = img_cfg["input_size"]
        cj = aug_cfg["color_jitter"]
        crop = aug_cfg["random_crop"]

        self.transforms = T.Compose([
            T.ToImage(),
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip"]),
            T.RandomResizedCrop(
                size=(size, size),
                scale=crop["scale"],
                ratio=crop["ratio"],
                antialias=True,
            ),
            T.ColorJitter(
                brightness=cj["brightness"],
                contrast=cj["contrast"],
                saturation=cj["saturation"],
                hue=cj["hue"],
            ),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
            T.ClampBoundingBoxes(),
            T.SanitizeBoundingBoxes(min_size=1),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class ValTransform:
    def __init__(self, config):
        img_cfg = config["image"]
        size = img_cfg["input_size"]

        self.transforms = T.Compose([
            T.ToImage(),
            T.Resize((size, size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
            T.ClampBoundingBoxes(),
            T.SanitizeBoundingBoxes(min_size=1),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
