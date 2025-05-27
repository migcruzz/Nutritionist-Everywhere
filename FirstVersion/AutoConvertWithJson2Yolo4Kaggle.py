import shutil
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import yaml

from Kaggle.JSON2YOLO.utils import exif_size


def find_image_by_stem(img_dir, stem):
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        candidate = img_dir / f"{stem}.{ext}"
        if candidate.exists():
            return candidate
    return None

def process_single_json(args):
    json_file, img_dir, labels_path, images_split_path = args
    try:
        img_stem = Path(json_file.stem).stem
        img_path = find_image_by_stem(img_dir, img_stem)
        if not img_path:
            print(f"Image not found for {json_file.name}")
            return None

        img_path = img_path.resolve()
        try:
            wh = exif_size(Image.open(img_path))
        except UnidentifiedImageError:
            print(f"Corrupted image: {img_path}")
            return None

        label_name = img_path.stem + ".txt"
        label_path = labels_path / label_name
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with json_file.open('r') as f:
            data = json.load(f)

        classes_found = set()
        with label_path.open("w") as f_out:
            for obj in data.get("objects", []):
                cls = obj["classTitle"].lower()
                classes_found.add(cls)
                points = obj["points"]["exterior"]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_center = (x_min + x_max) / 2 / wh[0]
                y_center = (y_min + y_max) / 2 / wh[1]
                width = (x_max - x_min) / wh[0]
                height = (y_max - y_min) / wh[1]

                if width > 0 and height > 0:
                    f_out.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        target_img_path = images_split_path / img_path.name
        target_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, target_img_path)

        return str(target_img_path), classes_found  # <- sem .resolve()

    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")
        return None

def convert_food_recognition_json(name, base_dir):
    base_dir = Path(base_dir).resolve()
    dataset_path = Path("Kaggle/datasets") / name
    labels_base_path = dataset_path / "labels"
    images_base_path = dataset_path / "images"
    labels_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path.mkdir(parents=True, exist_ok=True)

    sets = ['training', 'validation', 'test']
    file_list = []
    all_classes = set()
    split_files_map = {}

    for split in tqdm(sets, desc="Processing splits"):
        img_dir = base_dir / split / 'img'
        ann_dir = base_dir / split / 'ann'
        split_name = split.replace('training', 'train').replace('validation', 'val')
        labels_path = labels_base_path / split_name
        images_split_path = images_base_path / split_name
        labels_path.mkdir(parents=True, exist_ok=True)
        images_split_path.mkdir(parents=True, exist_ok=True)

        json_files = list(ann_dir.glob('*.json'))
        tasks = [(json_file, img_dir, labels_path, images_split_path) for json_file in json_files]

        split_img_paths = []
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            for result in tqdm(executor.map(process_single_json, tasks), total=len(tasks), desc=f"{split_name}", leave=False):
                if result:
                    img_path, classes = result
                    split_img_paths.append(img_path)
                    all_classes.update(classes)

        split_files_map[split_name] = split_img_paths
        file_list.extend(split_img_paths)

    # Salvar classes
    class_list = sorted(all_classes)
    with (dataset_path / f"{name}.names").open("w") as f:
        for c in class_list:
            f.write(f"{c}\n")

    # Mapear nomes para índices
    class_to_id = {cls: i for i, cls in enumerate(class_list)}
    for label_file in labels_base_path.rglob("*.txt"):
        with label_file.open("r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5 and parts[0] in class_to_id:
                new_lines.append(f"{class_to_id[parts[0]]} {' '.join(parts[1:])}\n")
        with label_file.open("w") as f:
            f.writelines(new_lines)

    # Gerar .train.txt, .val.txt, .test.txt com caminhos relativos
    for split, paths in split_files_map.items():
        split_txt = dataset_path / f"{name}.{split}.txt"
        with split_txt.open("w") as f:
            for p in paths:
                rel_path = Path(p).relative_to(dataset_path)
                f.write(f"{rel_path.as_posix()}\n")

    # Gerar .yaml
    yaml_path = dataset_path / f"{name}.yaml"
    data_yaml = {
        "path": ".",  # <- relativo para compatibilidade no zip ou no Kaggle
        "train": f"{name}.train.txt",
        "val": f"{name}.val.txt",
        "test": f"{name}.test.txt",
        "names": {i: c for i, c in enumerate(class_list)}
    }
    with yaml_path.open("w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\n✅ Conversão completa: {len(file_list)} imagens")
    print(f"📂 Dataset YOLO gerado em: {dataset_path.resolve()}")
    print(f"📄 Arquivo YAML: {yaml_path.resolve()}")

if __name__ == "__main__":
    convert_food_recognition_json(
        name="DatasetProcessedFromCocoToYoloFoodDetection2022",
        base_dir="./FoodRecognition2022"
    )
