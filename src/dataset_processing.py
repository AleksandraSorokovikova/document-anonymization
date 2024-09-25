import os
import fitz
import json
from src.config import pii_classes
import pybboxes as pbx
from PIL import Image, ImageDraw
from src.config import pii_entities_colors_names, pii_to_id, id_to_pii
from sklearn.model_selection import train_test_split


def create_yolov5_dataset_structure(
    original_pdf_folder,
    entities_json_folder,
    output_folder="dataset",
    root_folder="yolo-detection",
    zoom=1.5,
):
    """
    Function to convert PDFs to images and their corresponding bounding boxes to YOLO format.
    """

    custom_data_folder = os.path.join(root_folder, output_folder)
    os.makedirs(custom_data_folder, exist_ok=True)
    os.makedirs(os.path.join(custom_data_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(custom_data_folder, "labels"), exist_ok=True)
    # create train, val and test folders
    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(custom_data_folder, "images", folder), exist_ok=True)
        os.makedirs(os.path.join(custom_data_folder, "labels", folder), exist_ok=True)
    labels = {}

    for pdf_filename in os.listdir(original_pdf_folder):
        if pdf_filename.endswith(".pdf"):
            pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
            entity_json_filename = f"{pdf_name_without_ext}_bounding_boxes.json"
            entity_json_path = os.path.join(entities_json_folder, entity_json_filename)

            if not os.path.exists(entity_json_path):
                print(f"Skipping {pdf_filename}: No corresponding entity JSON found.")
                continue

            pdf_path = os.path.join(original_pdf_folder, pdf_filename)
            doc = fitz.open(pdf_path)

            page = doc.load_page(0)
            page_width, page_height = page.rect.width, page.rect.height

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            labels[pdf_name_without_ext] = {
                "image": pix,
                "entities": [],
            }

            with open(entity_json_path, "r") as f:
                entities = json.load(f)

            for entity in entities:
                entity_type, entity_value, bbox = entity

                class_index = pii_to_id[entity_type]

                yolo_cor = pbx.convert_bbox(
                    bbox,
                    from_type="voc",
                    to_type="yolo",
                    image_size=(page_width, page_height),
                )
                box_center_x, box_center_y, box_width, box_height = yolo_cor

                labels[pdf_name_without_ext]["entities"].append(
                    [class_index, box_center_x, box_center_y, box_width, box_height]
                )

    train, test = train_test_split(list(labels.keys()), test_size=0.3, random_state=42)
    validation, test = train_test_split(test, test_size=0.5, random_state=42)

    # write train, val and test labels
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
    for split, pdfs in zip(["train", "val", "test"], [train, validation, test]):
        for pdf_name in pdfs:
            image_path = os.path.join(custom_data_folder, "images", split, f"{pdf_name}.png")
            labels_path = os.path.join(custom_data_folder, "labels", split, f"{pdf_name}.txt")

            labels_list = labels[pdf_name]["entities"]
            image = labels[pdf_name]["image"]

            image.save(image_path)
            with open(labels_path, "w") as f:
                for label in labels_list:
                    f.write(" ".join(map(str, label)) + "\n")

    print("Dataset creation complete.")


def save_image_with_bounding_boxes_pillow(image_path, label_txt_path, output_path):
    # Open the image file using PIL
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Read the label txt file (YOLO format)
    with open(label_txt_path, "r") as f:
        lines = f.readlines()

    # Convert YOLO bounding boxes to pixel coordinates and draw them
    for line in lines:
        values = line.strip().split()
        class_name = id_to_pii[int(values[0])]
        color = pii_entities_colors_names[class_name]
        box_center_x = float(values[1])
        box_center_y = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert YOLO format to Pascal VOC (x0, y0, x1, y1)
        bbox_cor = pbx.convert_bbox(
            (box_center_x, box_center_y, box_width, box_height),
            from_type="yolo",
            to_type="voc",
            image_size=(img_width, img_height),
        )

        x0, y0, x1, y1 = bbox_cor

        # Draw rectangle on the image using Pillow
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    # Save the modified image
    img.save(output_path)
