import os
import fitz
import json
import pybboxes as pbx
from PIL import Image, ImageDraw, ImageFont
from src.config import pii_entities_colors_names, pii_to_id, id_to_pii, layoutlm_ner_classes
from src.process import convert_yolo_to_predictions
from sklearn.model_selection import train_test_split
from src.augmentation import Augmentation
import cv2
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel
from datasets import Image as datasetsImage



def return_image_with_bounding_boxes(img_path, bboxes_path, zoom=1.0, b_type="txt"):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if b_type == "txt":
        with open(bboxes_path, "r") as f:
            bboxes = f.readlines()
            bboxes = [list(map(float, bbox.strip().split())) for bbox in bboxes]
            bboxes = [[int(bbox[0])] + bbox[1:] for bbox in bboxes]
    elif b_type == "layoutlm":
        with open(bboxes_path, "r") as f:
            labels_file = json.load(f)
            bboxes_coord = labels_file["bboxes"]
            class_labels = [
                i if i == "O" else i.split("-")[1] for i in labels_file["ner_tags"]
            ]
            bboxes = [
                [pii_to_id.get(class_labels[i], -1)] + bboxes_coord[i] for i in range(len(bboxes_coord))
            ]
    else:
        raise ValueError("Invalid bounding boxes type. Choose between 'txt' and 'layoutlm'.")

    for bbox in bboxes:
        class_label, coords = bbox[0], bbox[1:]
        x0, y0, x1, y1 = [coord * zoom for coord in coords]
        class_name = id_to_pii.get(class_label, -1)
        color = pii_entities_colors_names.get(class_name, "black")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    return img


def launch_augmentation(
        path_to_original_pdfs,
        path_to_bounding_boxes,
        path_to_layoutlm_boxes,
        path_to_augmented_images,
        num_of_images=4,
        zoom_range=(1.25, 1.8)
):
    augmentation = Augmentation()
    os.makedirs(path_to_augmented_images, exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "images"), exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "labels"), exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "layoutlm_labels"), exist_ok=True)
    labels = {}
    mapping = defaultdict(list)
    all_images = []

    for pdf_filename in tqdm(os.listdir(path_to_original_pdfs)):
        if not pdf_filename.endswith(".pdf"):
            continue
        pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
        entity_json_filename = f"{pdf_name_without_ext}_bounding_boxes.json"
        entity_layoutlm_filename = f"{pdf_name_without_ext}_layoutlm_labels.json"
        entity_json_path = os.path.join(path_to_bounding_boxes, entity_json_filename)
        entity_layoutlm_path = os.path.join(path_to_layoutlm_boxes, entity_layoutlm_filename)

        if not os.path.exists(entity_json_path):
            # print(f"Skipping {pdf_filename}: No corresponding entity JSON found.")
            continue

        zoom = random.uniform(*zoom_range)
        pdf_path = os.path.join(path_to_original_pdfs, pdf_filename)
        doc = fitz.open(pdf_path)

        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        page_width, page_height = page.rect.width, page.rect.height
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        bboxes = []
        layoutlm_bboxes = []

        with open(entity_json_path, "r") as f:
            entities = json.load(f)

        with open(entity_layoutlm_path, "r") as f:
            layoutlm_labels = json.load(f)
            layoutlm_entities = layoutlm_labels['bboxes']
            assert layoutlm_labels is not None, f"layoutlm_labels is None for {pdf_filename}"
            assert "B-middle_name" not in layoutlm_labels["ner_tags"], f"middle_name is present in {pdf_filename}"
            assert "I-middle_name" not in layoutlm_labels["ner_tags"], f"middle_name is present in {pdf_filename}"

        assert sorted(list(pii_to_id.values())) == list(range(0, len(pii_to_id)))

        for entity in entities:
            entity_type, entity_value, bbox = entity
            zoomed_coord = [coord * zoom for coord in bbox]
            if entity_type not in pii_to_id:
                continue
            entity_type_code = pii_to_id[entity_type]
            bboxes.append([entity_type_code] + zoomed_coord)

        for bbox in layoutlm_entities:
            zoomed_coord = [coord * zoom for coord in bbox]
            layoutlm_bboxes.append(zoomed_coord)

        augmented_images, new_tuple_bboxes = augmentation.create_augmented_images(
            img_array, bboxes=[bboxes, layoutlm_bboxes], num_of_images=num_of_images
        )

        for i, augmented_image in enumerate(augmented_images):
            all_images.append((f"{pdf_name_without_ext}_{i}.png", augmented_image))
            mapping[pdf_name_without_ext].append(f"{pdf_name_without_ext}_{i}.png")

        for i in range(len(augmented_images)):
            new_bboxes = new_tuple_bboxes[i][0]
            new_layoutlm_bboxes = new_tuple_bboxes[i][1]
            new_layoutlm_labels = {
                "tokens": layoutlm_labels["tokens"],
                "bboxes": new_layoutlm_bboxes,
                "ner_tags": layoutlm_labels["ner_tags"]

            }

            labels[f"{pdf_name_without_ext}_{i}"] = {
                "image": augmented_images[i],
                "entities": new_bboxes,
                "layoutlm_labels": new_layoutlm_labels
            }

        for key in labels:
            labels[key]["entities"] = list(
                set(tuple(x) for x in labels[key]["entities"]))

    mapping = dict(mapping)

    for path, image in all_images:
        cv2.imwrite(os.path.join(path_to_augmented_images, "images", path), image)

    with open(os.path.join(path_to_augmented_images, "mapping.json"), "w") as f:
        json.dump(mapping, f, indent=4)

    for key in labels:
        label_path = os.path.join(path_to_augmented_images, "labels", f"{key}.txt")
        with open(label_path, "w") as f:
            for label in labels[key]["entities"]:
                f.write(" ".join(map(str, label)) + "\n")

        layoutlm_label_path = os.path.join(
            path_to_augmented_images, "layoutlm_labels", f"{key}.json"
        )
        with open(layoutlm_label_path, "w") as f:
            json.dump(labels[key]["layoutlm_labels"], f, indent=4)


def split_by_layout(mapping, train_val_ratio, val_test_ratio):
    docs = ["_".join(im.split("_")[:-1]) for im in mapping]
    layouts = set(docs)

    train, val = train_test_split(list(layouts), train_size=train_val_ratio, random_state=42)

    train_images = []
    val_images = []


    for image in mapping:
        image_layout = "_".join(image.split("_")[:-1])
        if image_layout in train:
            train_images.append(mapping[image])
        elif image_layout in val:
            val_images.append([random.choice(mapping[image])])

    val_images, test_images = train_test_split(val_images, train_size=val_test_ratio, random_state=42)

    return train_images, val_images, test_images


def split_layoutlm_dataset(
        path_to_folder,
        output_path,
        train_val_ratio=0.8,
        val_test_ratio=0.9,
        split_strategy="random"
):
    images_dir = os.path.join(path_to_folder, "images")
    labels_dir = os.path.join(path_to_folder, "layoutlm_labels")

    ner_classes = layoutlm_ner_classes
    ner_feature = ClassLabel(names=ner_classes)

    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "bboxes": Sequence(Sequence(Value("int64"))),
        "ner_tags": Sequence(ner_feature),
        "image": datasetsImage(decode=True),
    })

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    json_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".json")])
    assert len(image_files) == len(json_files), "Количество изображений и меток не совпадает"

    data = []
    for img_file, json_file in zip(image_files, json_files):
        img_path = os.path.join(images_dir, img_file)
        json_path = os.path.join(labels_dir, json_file)

        with open(json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        data.append({
            "id": img_file.replace(".png", ""),
            "tokens": label_data["tokens"],
            "bboxes": label_data["bboxes"],
            "ner_tags": label_data["ner_tags"],
            "image": img_path,
        })
    random.shuffle(data)

    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Создаем DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data, features=features),
        "val": Dataset.from_list(val_data, features=features),
        "test": Dataset.from_list(test_data, features=features),
    })

    dataset.save_to_disk(output_path)


def split_yolo_dataset(
        path_to_folder,
        output_folder,
        train_val_ratio = 0.8,
        val_test_ratio = 0.9,
        split_strategy="random"
):
    os.makedirs(output_folder, exist_ok=True)
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_folder, folder, split))

    with open(os.path.join(path_to_folder, "mapping.json"), "r") as f:
        mapping = json.load(f)

    if split_strategy == "random":
        train, test = train_test_split(list(mapping.values()), train_size=train_val_ratio, random_state=42)
        validation, test = train_test_split(test, train_size=val_test_ratio, random_state=42)
    elif split_strategy == "layout":
        train, validation, test = split_by_layout(mapping, train_val_ratio, val_test_ratio)
    else:
        raise ValueError("Invalid split strategy. Choose between 'random' and 'layout'.")

    """
    dataset/  # This is the 'path' in your custom.yaml
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── val/
    │       ├── image101.jpg
    │       ├── image102.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        └── val/
            ├── image101.txt
            ├── image102.txt
            └── ...
    """
    for split, images_split in zip(["train", "val", "test"], [train, validation, test]):
        for images in tqdm(images_split):
            if split == "train":
                images_to_copy = images
            else:
                images_to_copy = [random.choice(images)]

            for image in images_to_copy:
                try:
                    image_path = os.path.join(path_to_folder, "images", image)
                    label_path = os.path.join(path_to_folder, "labels", image.replace(".png", ".txt"))

                    img = Image.open(image_path)
                    img_width, img_height = img.size

                    with open(label_path, 'r') as f:
                        label = f.readlines()
                        bboxes = [list(map(float, line.strip().split())) for line in label]
                        bboxes = [[int(cls_id)] + coords for cls_id, *coords in bboxes]

                    converted_bboxes = []
                    for bbox in bboxes:
                        converted_bboxes.append([bbox[0]] + [c for c in pbx.convert_bbox(
                            (bbox[1], bbox[2], bbox[3], bbox[4]),
                            from_type="voc",
                            to_type="yolo",
                            image_size=(img_width, img_height),
                        )])

                    with open(os.path.join(output_folder, "labels", split, f"{image.replace('.png', '.txt')}"), "w") as f:
                        for bbox in converted_bboxes:
                            f.write(" ".join(map(str, bbox)) + "\n")

                    os.system(f"cp {image_path} {os.path.join(output_folder, 'images', split, image)}")
                except Exception as e:
                    print(f"Error processing {image}: {e}")
                    continue

    convert_yolo_to_predictions(
        output_folder,
        raw_labels_path="labels/test",
        new_labels_dir=f"{output_folder}/converted_test_labels",
        images_dir=f"{output_folder}/images/test"
    )

    print("Dataset creation complete.")


def save_image_with_bounding_boxes_pillow(
        image_path, label_txt_path, output_path=None, reference_labels=None, bbox_format="yolo"
):
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    with open(label_txt_path, "r") as f:
        lines = f.readlines()

    found_classes = []
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        found_classes.append(class_id)

        box_center_x = float(values[1])
        box_center_y = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])
        if bbox_format == "yolo":
            bbox_cor = pbx.convert_bbox(
                (box_center_x, box_center_y, box_width, box_height),
                from_type="yolo",
                to_type="voc",
                image_size=(img_width, img_height),
            )
        else:
            bbox_cor = (box_center_x, box_center_y, box_width, box_height)
        x0, y0, x1, y1 = bbox_cor

        # Draw the bounding box
        class_name = id_to_pii[class_id]
        color = pii_entities_colors_names[class_name]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    # Check for missing classes
    if reference_labels is not None:
        with open(reference_labels, "r") as ref_file:
            reference_lines = ref_file.readlines()
        reference_classes = sorted([int(line.strip().split()[0]) for line in reference_lines])
        found_classes = sorted(found_classes)
        for c in found_classes:
            if c in reference_classes:
                reference_classes.remove(c)
        missing_classes = reference_classes
        # Prepare the text for missing classes
        if missing_classes:
            text = "\n".join([f"Missing class: {id_to_pii[class_id]}" for class_id in missing_classes])

            # Load font
            font = ImageFont.load_default(size=16)

            # Calculate text size using the font
            text_width, text_height = 250, 250

            # Position the text at the bottom-right corner
            text_x = img_width - text_width - 10
            text_y = img_height - text_height - 10

            # Draw the text
            padding = 5
            draw.rectangle(
                [
                    text_x - padding, text_y - padding,
                    text_x + text_width + padding, text_y + text_height + padding
                ],
                fill="white", outline="black"
            )

            # Draw the text
            draw.text((text_x, text_y), text, fill="red", font=font)

    # Save the modified image
    if output_path:
        img.save(output_path)
    else:
        return img
