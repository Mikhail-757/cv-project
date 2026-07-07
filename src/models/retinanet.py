import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def create_retinanet(num_classes, config):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if config["models"]["retinanet"]["pretrained"] else None
    model = retinanet_resnet50_fpn_v2(weights=weights)

    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.conv[0][0].in_channels

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
    )

    return model


def get_retinanet_params(model, config):
    cfg = config["models"]["retinanet"]
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
