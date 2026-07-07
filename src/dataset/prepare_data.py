import os
import json
import shutil
from pathlib import Path
from pycocotools.coco import COCO


def download_coco(raw_dir, year="2017"):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "train_images": f"http://images.cocodataset.org/zips/train{year}.zip",
        "val_images": f"http://images.cocodataset.org/zips/val{year}.zip",
        "annotations": f"http://images.cocodataset.org/annotations/annotations_trainval{year}.zip",
    }

    import urllib.request
    import zipfile

    for name, url in urls.items():
        zip_path = raw_dir / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)
        extract_dir = raw_dir
        if not (raw_dir / f"train{year}").exists() or not (raw_dir / f"val{year}").exists():
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

    return raw_dir


def filter_animal_annotations(raw_dir, processed_dir, animal_classes, year="2017"):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        ann_file = raw_dir / "annotations" / f"instances_{split}{year}.json"
        coco = COCO(str(ann_file))

        cat_ids = coco.getCatIds(catNms=animal_classes)
        cat_id_to_new = {old: i + 1 for i, old in enumerate(sorted(cat_ids))}
        cat_info = []
        for cid in sorted(cat_ids):
            info = coco.loadCats(cid)[0]
            cat_info.append({
                "id": cat_id_to_new[cid],
                "name": info["name"],
                "supercategory": info["supercategory"],
            })

        img_ids = set()
        for cid in cat_ids:
            img_ids.update(coco.getImgIds(catIds=[cid]))
        img_ids = sorted(img_ids)

        images = coco.loadImgs(img_ids)
        ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        new_anns = []
        for ann in anns:
            if ann["category_id"] in cat_id_to_new:
                new_ann = ann.copy()
                new_ann["category_id"] = cat_id_to_new[ann["category_id"]]
                new_anns.append(new_ann)

        split_dir = processed_dir / split
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        valid_images = []
        for img_info in images:
            src = raw_dir / f"{split}{year}" / img_info["file_name"]
            dst = img_dir / img_info["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            if src.exists() or dst.exists():
                valid_images.append(img_info)

        coco_out = {
            "images": valid_images,
            "annotations": new_anns,
            "categories": cat_info,
        }

        ann_out = split_dir / "annotations.json"
        with open(ann_out, "w") as f:
            json.dump(coco_out, f)

        print(f"{split}: {len(valid_images)} images, {len(new_anns)} annotations, {len(cat_info)} classes")

    return processed_dir


def prepare_yolo_data(processed_dir, output_dir):
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ann = processed_dir / "train" / "annotations.json"
    with open(train_ann) as f:
        data = json.load(f)

    class_names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]
    num_classes = len(class_names)

    for split in ["train", "val"]:
        ann_file = processed_dir / split / "annotations.json"
        with open(ann_file) as f:
            split_data = json.load(f)

        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_id_to_info = {img["id"]: img for img in split_data["images"]}
        img_id_to_anns = {}
        for ann in split_data["annotations"]:
            img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        for img_id, img_info in img_id_to_info.items():
            src = processed_dir / split / "images" / img_info["file_name"]
            dst = img_dir / img_info["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

            w, h = img_info["width"], img_info["height"]
            label_file = lbl_dir / (Path(img_info["file_name"]).stem + ".txt")
            lines = []
            for ann in img_id_to_anns.get(img_id, []):
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cls_idx = ann["category_id"] - 1
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            with open(label_file, "w") as f:
                f.write("\n".join(lines))

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": class_names,
    }

    yaml_path = output_dir / "data.yaml"
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return str(yaml_path), num_classes, class_names


def get_dataset_info(processed_dir):
    processed_dir = Path(processed_dir)
    ann_file = processed_dir / "train" / "annotations.json"
    with open(ann_file) as f:
        data = json.load(f)
    categories = sorted(data["categories"], key=lambda x: x["id"])
    class_names = [c["name"] for c in categories]
    num_classes = len(class_names)
    return num_classes, class_names
