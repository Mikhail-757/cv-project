import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


def create_ssd(num_classes, config):
    weights = SSD300_VGG16_Weights.DEFAULT if config["models"]["ssd"]["pretrained"] else None
    model = ssd300_vgg16(weights=weights)

    num_classes_with_bg = num_classes + 1
    in_channels = [c.in_channels for c in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    cls_heads = torch.nn.ModuleList()
    for channels, anchors in zip(in_channels, num_anchors):
        cls_heads.append(
            torch.nn.Conv2d(channels, num_classes_with_bg * anchors, kernel_size=3, padding=1)
        )
    model.head.classification_head.module_list = cls_heads
    model.head.classification_head.num_columns = num_classes_with_bg
    model.head.classification_head.num_classes = num_classes_with_bg

    return model


def get_ssd_params(model, config):
    cfg = config["models"]["ssd"]
    lr = config["training"]["learning_rate"]
    factor = cfg["lr_backbone_factor"]

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {"params": backbone_params, "lr": lr * factor},
        {"params": head_params, "lr": lr},
    ]
