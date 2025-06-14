import os
import fitz
import json
import pybboxes as pbx
from PIL import Image, ImageDraw, ImageFont
from sympy.core.random import shuffle

from src.config import pii_entities_colors_names, pii_to_id, id_to_pii, layoutlm_ner_classes, pii_entities_colors_rgba
from sklearn.model_selection import train_test_split
from src.augmentation import Augmentation
import cv2
import shutil
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel
from datasets import Image as datasetsImage



def return_image_with_bounding_boxes(img_path, bboxes_path, zoom=1.0, b_type="txt"):
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

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
        if class_name == -1:
            continue
        color = pii_entities_colors_rgba.get(class_name, "black")
        draw.rectangle([x0, y0, x1, y1], fill=color)
    combined = Image.alpha_composite(img, overlay)
    return combined


def launch_augmentation(
        path_to_original_pdfs,
        path_to_bounding_boxes,
        path_to_layoutlm_boxes,
        path_to_augmented_images,
        num_of_images=1,
        zoom_range=(1.35, 1.8)
):
    augmentation = Augmentation()
    os.makedirs(path_to_augmented_images, exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "images"), exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "labels"), exist_ok=True)
    os.makedirs(os.path.join(path_to_augmented_images, "layoutlm_labels"), exist_ok=True)
    labels = {}
    mapping = defaultdict(list)
    all_images = []
    original_pdfs = os.listdir(path_to_original_pdfs)

    for pdf_filename in tqdm(original_pdfs):
        if not pdf_filename.endswith(".pdf"):
            continue
        pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
        entity_json_filename = f"{pdf_name_without_ext}_bounding_boxes.json"
        entity_layoutlm_filename = f"{pdf_name_without_ext}_layoutlm_labels.json"
        entity_json_path = os.path.join(path_to_bounding_boxes, entity_json_filename)
        entity_layoutlm_path = os.path.join(path_to_layoutlm_boxes, entity_layoutlm_filename)

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
        layoutlm_bboxes = []

        with open(entity_json_path, "r") as f:
            entities = json.load(f)

        with open(entity_layoutlm_path, "r") as f:
            layoutlm_labels = json.load(f)
            layoutlm_entities = layoutlm_labels['bboxes']
            assert layoutlm_labels is not None, f"layoutlm_labels is None for {pdf_filename}"
            assert "B-middle_name" not in layoutlm_labels["ner_tags"], f"middle_name is present in {pdf_filename}"
            assert "I-middle_name" not in layoutlm_labels["ner_tags"], f"middle_name is present in {pdf_filename}"

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


def delete_signatures(labels):
    while "B-signature" in labels["ner_tags"]:
        signature_index = labels["ner_tags"].index("B-signature")
        labels["tokens"].pop(signature_index)
        labels["bboxes"].pop(signature_index)
        labels["ner_tags"].pop(signature_index)
    return labels


def add_payment_information(labels):
    for i, ner_tag in enumerate(labels["ner_tags"]):
        if ner_tag == "B-iban" or ner_tag == "B-credit_card_number":
            labels["ner_tags"][i] = "B-payment_information"
        elif ner_tag == "I-iban" or ner_tag == "I-credit_card_number":
            labels["ner_tags"][i] = "I-payment_information"
    return labels


def delete_dates(labels):
    for i, ner_tag in enumerate(labels["ner_tags"]):
        if ner_tag == "B-dates" or ner_tag == "I-dates":
            labels["ner_tags"][i] = "O"
    return labels


def split_layoutlm_dataset(
        path_to_folder,
        output_path,
        doc_path,
        new_ner_tags=None,
):
    images_dir = os.path.join(path_to_folder, "images")
    labels_dir = os.path.join(path_to_folder, "layoutlm_labels")
    unique_labels = set()

    with open(doc_path, "r") as f:
        documents = json.load(f)
    documents = {
        doc["file_name"].replace(".pdf", ""): doc for doc in documents
    }

    ner_classes = layoutlm_ner_classes
    if new_ner_tags:
        ner_classes += new_ner_tags
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
    assert len(image_files) == len(json_files), "Number of images and labels do not match"

    train_data = []
    val_data = []
    test_data = []
    for img_file, json_file in zip(image_files, json_files):
        img_path = os.path.join(images_dir, img_file)
        json_path = os.path.join(labels_dir, json_file)

        with open(json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        label_data = delete_signatures(label_data)
        label_data = add_payment_information(label_data)
        label_data = delete_dates(label_data)

        unique_labels.update(label_data["ner_tags"])

        assert len(label_data["tokens"]) == len(label_data["bboxes"]) == len(label_data["ner_tags"]), \
            f"Lenghth mismatch in {img_file}: "

        assert "B-signature" not in label_data["ner_tags"], f"Signature found in {img_file}"

        file_id = "_".join(img_file.split("_")[:-1])
        if documents[file_id]["split"] == "train":
            train_data.append({
                "id": img_file.replace(".png", ""),
                "tokens": label_data["tokens"],
                "bboxes": label_data["bboxes"],
                "ner_tags": label_data["ner_tags"],
                "image": img_path,
            })
        elif documents[file_id]["split"] == "val":
            val_data.append({
                "id": img_file.replace(".png", ""),
                "tokens": label_data["tokens"],
                "bboxes": label_data["bboxes"],
                "ner_tags": label_data["ner_tags"],
                "image": img_path,
            })
        else:
            test_data.append({
                "id": img_file.replace(".png", ""),
                "tokens": label_data["tokens"],
                "bboxes": label_data["bboxes"],
                "ner_tags": label_data["ner_tags"],
                "image": img_path,
            })

    print(f"Test size: {len(test_data)}")

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data, features=features),
        "val": Dataset.from_list(val_data, features=features),
        "test": Dataset.from_list(test_data, features=features),
    })

    dataset.save_to_disk(output_path)

    os.makedirs(f"{output_path}/test_images", exist_ok=True)
    os.makedirs(f"{output_path}/test_labeled_images", exist_ok=True)
    os.makedirs(f"{output_path}/test_layoutlm_labels", exist_ok=True)

    for image in dataset["test"]["id"]:
        json_label = image + ".json"
        with open(f"{path_to_folder}/layoutlm_labels/{json_label}", "r") as f:
            label_data = json.load(f)
        label_data = delete_signatures(label_data)
        label_data = add_payment_information(label_data)
        label_data = delete_dates(label_data)
        with open(f"{output_path}/test_layoutlm_labels/{json_label}", "w") as f:
            json.dump(label_data, f, indent=4)


    for image in dataset["test"]["id"]:
        labeled_image = return_image_with_bounding_boxes(
            f"{path_to_folder}/images/{image}.png",
            f"{output_path}/test_layoutlm_labels/{image}.json",
            b_type="layoutlm"
        )
        shutil.copy(
            f"{path_to_folder}/images/{image}.png",
            f"{output_path}/test_images/{image}.png",
        )
        labeled_image.save(f"{output_path}/test_labeled_images/{image}.png")


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

        class_name = id_to_pii[class_id]
        color = pii_entities_colors_names[class_name]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    if reference_labels is not None:
        with open(reference_labels, "r") as ref_file:
            reference_lines = ref_file.readlines()
        reference_classes = sorted([int(line.strip().split()[0]) for line in reference_lines])
        found_classes = sorted(found_classes)
        for c in found_classes:
            if c in reference_classes:
                reference_classes.remove(c)
        missing_classes = reference_classes
        if missing_classes:
            text = "\n".join([f"Missing class: {id_to_pii[class_id]}" for class_id in missing_classes])

            font = ImageFont.load_default(size=16)

            text_width, text_height = 250, 250

            text_x = img_width - text_width - 10
            text_y = img_height - text_height - 10

            padding = 5
            draw.rectangle(
                [
                    text_x - padding, text_y - padding,
                    text_x + text_width + padding, text_y + text_height + padding
                ],
                fill="white", outline="black"
            )

            draw.text((text_x, text_y), text, fill="red", font=font)

    if output_path:
        img.save(output_path)
    else:
        return img
