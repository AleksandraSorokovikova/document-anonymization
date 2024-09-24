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
    output_folder,
    root_folder="yolov5",
    zoom=1.5,
):
    """
    Function to convert PDFs to images and their corresponding bounding boxes to YOLO format.
    """

    custom_data_folder = os.path.join(root_folder, output_folder)
    os.makedirs(custom_data_folder, exist_ok=True)
    all_images = []

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
            img_filename = f"{pdf_name_without_ext}.png"
            img_path = os.path.join(custom_data_folder, img_filename)
            all_images.append(os.path.join(output_folder, img_filename))
            pix.save(img_path)

            with open(entity_json_path, "r") as f:
                entities = json.load(f)

            label_filename = f"{pdf_name_without_ext}.txt"
            label_path = os.path.join(custom_data_folder, label_filename)

            with open(label_path, "w") as label_file:
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

                    label_file.write(
                        f"{class_index} {box_center_x} {box_center_y} {box_width} {box_height}\n"
                    )

    train, test = train_test_split(all_images, test_size=0.4, random_state=42)
    validation, test = train_test_split(test, test_size=0.5, random_state=42)
    with open(os.path.join(root_folder, "dataset", "custom_train.txt"), "w") as f:
        for item in train:
            f.write("%s\n" % item)
    with open(os.path.join(root_folder, "dataset", "custom_validation.txt"), "w") as f:
        for item in validation:
            f.write("%s\n" % item)
    with open(os.path.join(root_folder, "dataset", "custom_test.txt"), "w") as f:
        for item in test:
            f.write("%s\n" % item)

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
