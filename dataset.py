import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


class ImageParser:
    def __init__(self, anns_file, imgs_dir=None):
        with open(anns_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        self.imgs_dir = Path(imgs_dir) if imgs_dir else None
        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in data.get("annotations", []):
            self.annIm_dict[ann["image_id"]].append(ann)
            self.annId_dict[ann["id"]] = ann
        for img in data.get("images", []):
            self.im_dict[img["id"]] = img
        for cat in data.get("categories", []):
            self.cat_dict[cat["id"]] = cat
        for license_info in data.get("licenses", []):
            self.licenses_dict[license_info["id"]] = license_info

    def get_imgIds(self):
        return list(self.im_dict)

    def get_allCatId(self):
        return list(self.cat_dict)

    def get_annIds(self, im_ids):
        ids = im_ids if isinstance(im_ids, (list, tuple)) else [im_ids]
        return [ann["id"] for image_id in ids for ann in self.annIm_dict[image_id]]

    def load_anns(self, ann_ids):
        ids = ann_ids if isinstance(ann_ids, (list, tuple)) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ids]

    def load_cats(self, class_ids):
        ids = class_ids if isinstance(class_ids, (list, tuple)) else [class_ids]
        return [self.cat_dict[class_id] for class_id in ids]

    def get_imgLicenses(self, im_ids):
        ids = im_ids if isinstance(im_ids, (list, tuple)) else [im_ids]
        return [self.licenses_dict[self.im_dict[i]["license"]] for i in ids]

    def get_imgWidth(self, im_ids):
        ids = im_ids if isinstance(im_ids, (list, tuple)) else [im_ids]
        return [self.im_dict[i]["width"] for i in ids]

    def get_imgHeigth(self, im_ids):
        ids = im_ids if isinstance(im_ids, (list, tuple)) else [im_ids]
        return [self.im_dict[i]["height"] for i in ids]

    def get_imgFilename(self, im_ids):
        ids = im_ids if isinstance(im_ids, (list, tuple)) else [im_ids]
        return [self.im_dict[i]["file_name"] for i in ids]


def create_txt_file_val(strings, file_name, labels_dir="/content/kitti/labels"):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    output = labels_dir / (Path(file_name).stem + ".txt")
    output.write_text("\n".join(strings) + ("\n" if strings else ""), encoding="utf-8")
    return output


def create_yaml(strings, output_path="/content/yolov3/dataset.yaml"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(strings) + "\n", encoding="utf-8")
    return output_path


def build_dataset_yaml(annotation_file, output_path="/content/yolov3/dataset.yaml", root="/content", train_images="kitti/images", val_images="kitti/images"):
    parser = ImageParser(annotation_file)
    used_ids = []
    for image_id in parser.get_imgIds():
        for ann in parser.load_anns(parser.get_annIds(image_id)):
            used_ids.append(ann["category_id"])
    used_ids = np.sort(np.unique(used_ids)).tolist()
    lines = [f"path: {root}", f"train: {train_images}", f"val: {val_images}", "", "names:"]
    for index, category_id in enumerate(used_ids):
        category = parser.load_cats(category_id)[0]
        lines.append(f"  {index}: {category['name']}")
    return create_yaml(lines, output_path)
