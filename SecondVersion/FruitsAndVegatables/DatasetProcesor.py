import os
import re
import shutil

import cv2


def normalize_class_name(raw_name):
    name = raw_name.lower().replace("_", " ")
    return re.split(r"\s+", name.strip())[0].capitalize()


def get_yolo_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(255 - gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img_h, img_w = image.shape[:2]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return (cx, cy, w / img_w, h / img_h)


def collect_classes(base_dir):
    class_names = set()
    for split in ['train', 'test', 'validation']:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue
        for d in os.listdir(split_path):
            full_path = os.path.join(split_path, d)
            if os.path.isdir(full_path):
                class_names.add(normalize_class_name(d))
    class_names = sorted(class_names)
    return class_names, {name: idx for idx, name in enumerate(class_names)}


def copy_dataset(original_dir, copy_dir):
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    shutil.copytree(original_dir, copy_dir)
    print(f"📁 Copiado para {copy_dir}")


def create_txt_labels(base_dir, class_to_id):
    for split in ['train', 'test', 'validation']:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue

        for raw_class in os.listdir(split_path):
            class_path = os.path.join(split_path, raw_class)
            if not os.path.isdir(class_path):
                continue

            norm_class = normalize_class_name(raw_class)
            class_id = class_to_id[norm_class]

            for file in os.listdir(class_path):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(class_path, file)
                txt_path = os.path.splitext(img_path)[0] + ".txt"

                if os.path.exists(txt_path):
                    continue

                image = cv2.imread(img_path)
                if image is None:
                    continue

                bbox = get_yolo_bbox(image)
                if bbox is None:
                    continue

                cx, cy, bw, bh = bbox
                if all(0 <= v <= 1 for v in [cx, cy, bw, bh]):
                    with open(txt_path, "w") as f:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")


if __name__ == "__main__":
    original_dir = "fruits-and-vegetables"
    labeled_dir = "fruits-and-vegetables-labeled"

    class_names, class_to_id = collect_classes(original_dir)
    copy_dataset(original_dir, labeled_dir)
    create_txt_labels(labeled_dir, class_to_id)

