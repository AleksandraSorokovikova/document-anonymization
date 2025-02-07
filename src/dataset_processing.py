import os
import fitz
import json
import pybboxes as pbx
from PIL import Image, ImageDraw, ImageFont
from src.config import pii_entities_colors_names, pii_to_id, id_to_pii
from sklearn.model_selection import train_test_split
from src.augmentation import Augmentation
import cv2
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm


def return_image_with_bounding_boxes(img_path, bboxes_path, zoom=1.0):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    with open(bboxes_path, "r") as f:
        bboxes = f.readlines()
        bboxes = [list(map(float, bbox.strip().split())) for bbox in bboxes]
        bboxes = [[int(bbox[0])] + bbox[1:] for bbox in bboxes]

    for bbox in bboxes:
        class_label, coords = bbox[0], bbox[1:]
        x0, y0, x1, y1 = [coord * zoom for coord in coords]
        class_name = id_to_pii[class_label]
        color = pii_entities_colors_names.get(class_name, "black")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    return img


def launch_augmentation(
        path_to_original_pdfs, path_to_bounding_boxes,
        path_to_augmented_images, num_of_images=4, zoom_range=(1.15, 1.75)
):
    augmentation = Augmentation()
    os.makedirs(path_to_augmented_images, exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "images"), exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "labels"), exist_ok=True)
    labels = {}
    mapping = defaultdict(list)
    all_images = []
    for pdf_filename in tqdm(os.listdir(path_to_original_pdfs)):
        if not pdf_filename.endswith(".pdf"):
            continue
        pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
        entity_json_filename = f"{pdf_name_without_ext}_bounding_boxes.json"
        entity_json_path = os.path.join(path_to_bounding_boxes, entity_json_filename)

        if not os.path.exists(entity_json_path):
            print(f"Skipping {pdf_filename}: No corresponding entity JSON found.")
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

        with open(entity_json_path, "r") as f:
            entities = json.load(f)

        assert sorted(list(pii_to_id.values())) == list(range(0, len(pii_to_id)))

        for entity in entities:
            entity_type, entity_value, bbox = entity
            zoomed_coord = [coord * zoom for coord in bbox]
            if entity_type not in pii_to_id:
                continue
            entity_type_code = pii_to_id[entity_type]
            bboxes.append([entity_type_code] + zoomed_coord)

        augmented_images, new_bboxes = augmentation.create_augmented_images(
            img_array, bboxes=bboxes, num_of_images=num_of_images
        )

        for i, augmented_image in enumerate(augmented_images):
            all_images.append((f"{pdf_name_without_ext}_{i}.png", augmented_image))
            mapping[pdf_name_without_ext].append(f"{pdf_name_without_ext}_{i}.png")

        for i in range(len(augmented_images)):
            labels[f"{pdf_name_without_ext}_{i}"] = {
                "image": augmented_images[i],
                "entities": new_bboxes[i]
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


def split_yolo_dataset(
        path_to_folder,
        output_folder,
        train_val_ratio = 0.8,
        val_test_ratio = 0.9,
):
    os.makedirs(output_folder, exist_ok=True)
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_folder, folder, split))

    with open(os.path.join(path_to_folder, "mapping.json"), "r") as f:
        mapping = json.load(f)

    train, test = train_test_split(list(mapping.values()), train_size=train_val_ratio, random_state=42)
    validation, test = train_test_split(test, train_size=val_test_ratio, random_state=42)

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
