import numpy as np
import torch


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(precisions, recalls):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    recall_levels = np.linspace(0, 1, 101)
    precision_at_recall = np.zeros(101)

    for i, r in enumerate(recall_levels):
        above = mpre[mrec >= r]
        if len(above) > 0:
            precision_at_recall[i] = above.max()

    return precision_at_recall.mean()


def evaluate_detections(predictions, ground_truths, num_classes, iou_threshold=0.5):
    per_class_ap = {}
    all_precisions = []
    all_recalls = []

    for cls in range(1, num_classes + 1):
        cls_preds = []
        n_gt = 0

        for img_idx in range(len(predictions)):
            pred = predictions[img_idx]
            gt = ground_truths[img_idx]

            gt_mask = gt["labels"] == cls
            n_gt += gt_mask.sum().item()
            gt_boxes_cls = gt["boxes"][gt_mask]
            gt_matched = [False] * len(gt_boxes_cls)

            pred_mask = pred["labels"] == cls
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            for j in range(len(pred_boxes)):
                best_iou = 0.0
                best_gt = -1
                for k in range(len(gt_boxes_cls)):
                    iou = compute_iou(pred_boxes[j].tolist(), gt_boxes_cls[k].tolist())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = k
                cls_preds.append({
                    "score": pred_scores[j].item(),
                    "tp": best_iou >= iou_threshold and best_gt >= 0 and not gt_matched[best_gt],
                    "img_idx": img_idx,
                })
                if best_iou >= iou_threshold and best_gt >= 0 and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True

        if n_gt == 0:
            continue

        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precs = []
        recs = []

        for p in cls_preds:
            if p["tp"]:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precs.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recs.append(tp_cumsum / n_gt)

        if precs:
            ap = compute_ap(np.array(precs), np.array(recs))
            per_class_ap[cls] = ap
            all_precisions.append(precs[-1] if precs else 0)
            all_recalls.append(recs[-1] if recs else 0)

    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
    precision = np.mean(all_precisions) if all_precisions else 0.0
    recall = np.mean(all_recalls) if all_recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mAP50": float(mAP),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class_ap": {k: float(v) for k, v in per_class_ap.items()},
    }


def compute_map_at_range(predictions, ground_truths, num_classes, iou_range=(0.5, 0.95), step=0.05):
    thresholds = np.arange(iou_range[0], iou_range[1] + step, step)
    maps = []

    for thr in thresholds:
        result = evaluate_detections(predictions, ground_truths, num_classes, iou_threshold=thr)
        maps.append(result["mAP50"])

    return {
        "mAP50": maps[0] if maps else 0.0,
        "mAP50-95": float(np.mean(maps)) if maps else 0.0,
    }


@torch.no_grad()
def evaluate_model(model, dataloader, num_classes, device, model_name="model"):
    model.eval()
    predictions = []
    ground_truths = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        for out in outputs:
            predictions.append({k: v.cpu() for k, v in out.items()})
        for t in targets:
            ground_truths.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()})

    metrics = evaluate_detections(predictions, ground_truths, num_classes)
    range_metrics = compute_map_at_range(predictions, ground_truths, num_classes)
    metrics["mAP50-95"] = range_metrics["mAP50-95"]

    return metrics, predictions, ground_truths
