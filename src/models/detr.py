import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment


class PositionalEncoding2D(nn.Module):
    def __init__(self, hidden_dim, temperature=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        half = self.hidden_dim // 2

        y_pos = torch.arange(h, device=device).unsqueeze(1).expand(h, w).float()
        x_pos = torch.arange(w, device=device).unsqueeze(0).expand(h, w).float()

        dim = torch.arange(half, device=device).float()
        dim = self.temperature ** (2 * (dim // 2) / half)

        pe_x = torch.zeros(half, h, w, device=device)
        pe_y = torch.zeros(half, h, w, device=device)

        pe_x[0::2] = torch.sin(x_pos.unsqueeze(0) / dim[0::2].unsqueeze(1).unsqueeze(2))
        pe_x[1::2] = torch.cos(x_pos.unsqueeze(0) / dim[1::2].unsqueeze(1).unsqueeze(2))
        pe_y[0::2] = torch.sin(y_pos.unsqueeze(0) / dim[0::2].unsqueeze(1).unsqueeze(2))
        pe_y[1::2] = torch.cos(y_pos.unsqueeze(0) / dim[1::2].unsqueeze(1).unsqueeze(2))

        pe = torch.cat([pe_x, pe_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return pe


class DETR(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        cfg = config["models"]["detr"]
        hidden_dim = cfg["hidden_dim"]
        num_heads = cfg["num_heads"]
        num_enc = cfg["num_encoder_layers"]
        num_dec = cfg["num_decoder_layers"]
        num_queries = cfg["num_queries"]

        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if cfg["pretrained_backbone"] else None
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.pos_enc = PositionalEncoding2D(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 2)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.num_classes = num_classes
        self.num_queries = num_queries

    def forward(self, x):
        features = self.backbone(x)
        h = self.conv(features)
        pos = self.pos_enc(h)

        b, c, fh, fw = h.shape
        h_flat = (h + pos).flatten(2).permute(0, 2, 1)
        memory = self.encoder(h_flat)

        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(queries, memory)

        class_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs)

        return {"pred_logits": class_logits, "pred_boxes": bbox_pred}


class DETRLoss(nn.Module):
    def __init__(self, num_classes, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.no_object_weight = 0.1

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        b = pred_logits.shape[0]

        total_loss = torch.tensor(0.0, device=pred_logits.device)
        num_boxes = 0

        for i in range(b):
            tgt_boxes = targets[i]["boxes"]
            tgt_labels = targets[i]["labels"]

            if len(tgt_labels) == 0:
                weight = torch.ones(self.num_classes + 2, device=pred_logits.device)
                no_obj_label = (self.num_classes + 1) * torch.ones(
                    pred_logits.shape[1], dtype=torch.long, device=pred_logits.device
                )
                total_loss += F.cross_entropy(pred_logits[i], no_obj_label, weight=weight) * 0.1
                continue

            num_boxes += len(tgt_labels)

            with torch.no_grad():
                out_prob = pred_logits[i].softmax(-1)
                cost_cls = -out_prob[:, tgt_labels]
                cost_box = torch.cdist(pred_boxes[i], tgt_boxes, p=1)
                C = self.cost_class * cost_cls + self.cost_bbox * cost_box
                row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())

            row_ind = torch.tensor(row_ind, device=pred_logits.device)
            col_ind = torch.tensor(col_ind, device=pred_logits.device)

            target_classes = torch.full(
                (pred_logits.shape[1],), self.num_classes + 1,
                dtype=torch.long, device=pred_logits.device,
            )
            target_classes[row_ind] = tgt_labels[col_ind]

            weight = torch.ones(self.num_classes + 2, device=pred_logits.device)
            weight[-1] = self.no_object_weight
            loss_ce = F.cross_entropy(pred_logits[i], target_classes, weight=weight)

            loss_bbox = F.l1_loss(pred_boxes[i][row_ind], tgt_boxes[col_ind], reduction="mean")

            total_loss += loss_ce + self.cost_bbox * loss_bbox

        if num_boxes > 0:
            total_loss = total_loss / max(num_boxes, 1)

        return total_loss


class DETRWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.loss_fn = DETRLoss(num_classes)
        self.num_classes = num_classes

    def forward(self, images, targets=None):
        x = torch.stack(images) if isinstance(images, (list, tuple)) else images
        outputs = self.model(x)

        if targets is not None and self.training:
            detr_targets = []
            for t in targets:
                boxes = t["boxes"].float()
                h, w = x.shape[-2], x.shape[-1]
                if boxes.numel() > 0:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
                    bw = (boxes[:, 2] - boxes[:, 0]) / w
                    bh = (boxes[:, 3] - boxes[:, 1]) / h
                    norm_boxes = torch.stack([cx, cy, bw, bh], dim=1)
                else:
                    norm_boxes = torch.zeros((0, 4), device=boxes.device)
                detr_targets.append({"boxes": norm_boxes, "labels": t["labels"]})

            loss = self.loss_fn(outputs, detr_targets)
            return {"loss_total": loss}

        return self._postprocess(outputs, x.shape[-2], x.shape[-1])

    def _postprocess(self, outputs, img_h, img_w):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        results = []

        for logits, boxes in zip(pred_logits, pred_boxes):
            probs = logits.softmax(-1)
            scores, labels = probs[:, :-1].max(-1)

            keep = scores > 0.3
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if boxes.numel() > 0:
                cx, cy, bw, bh = boxes.unbind(-1)
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                det_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            else:
                det_boxes = torch.zeros((0, 4), device=logits.device)

            results.append({
                "boxes": det_boxes,
                "labels": labels,
                "scores": scores,
            })

        return results


def create_detr(num_classes, config):
    model = DETR(num_classes, config)
    wrapper = DETRWrapper(model, num_classes)
    return wrapper


def get_detr_params(model, config):
    cfg = config["models"]["detr"]
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
