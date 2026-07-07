import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_faster_rcnn(num_classes, config):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if config["models"]["faster_rcnn"]["pretrained"] else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


def get_faster_rcnn_params(model, config):
    cfg = config["models"]["faster_rcnn"]
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
