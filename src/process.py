import json
from src.config import *
import cv2
import numpy as np
import math
import os
from PIL import Image
from scipy.ndimage import interpolation as inter


def rotate_bbox(x_min, y_min, x_max, y_max, angle, img_width, img_height):
    angle = np.radians(angle)

    cx, cy = img_width / 2, img_height / 2

    points = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    rotated_points = np.dot(points - [cx, cy], rotation_matrix) + [cx, cy]

    x_min_new = np.min(rotated_points[:, 0])
    y_min_new = np.min(rotated_points[:, 1])
    x_max_new = np.max(rotated_points[:, 0])
    y_max_new = np.max(rotated_points[:, 1])

    return x_min_new, y_min_new, x_max_new, y_max_new


def process_labels(path_to_labels):
    with open(path_to_labels, "r") as f:
        labels = json.load(f)

    all_labels = {}
    for label in labels:
        image_name = label["image"].split(".")[0]
        label_name = image_name + ".txt"
        txt_labels = []
        for bbox in label["label"]:
            label_id = pii_to_id[bbox["rectanglelabels"][0]]
            x_percent = bbox["x"]
            y_percent = bbox["y"]
            width_percent = bbox["width"]
            height_percent = bbox["height"]

            original_width = bbox["original_width"]
            original_height = bbox["original_height"]

            rotation = bbox["rotation"]

            x_min = (x_percent / 100) * original_width
            y_min = (y_percent / 100) * original_height
            x_max = x_min + (width_percent / 100) * original_width
            y_max = y_min + (height_percent / 100) * original_height

            if rotation != 0:
                x_min, y_min, x_max, y_max = rotate_bbox(
                    x_min, y_min, x_max, y_max, rotation, original_width, original_height
                )
                print(image_name)

            txt_labels.append(f"{label_id} {x_min} {y_min} {x_max} {y_max}")
        all_labels[label_name] = '\n'.join(txt_labels)

    for label in all_labels:
        with open(f"benchmark_dataset/labels/{label}", "w") as f:
            f.write(all_labels[label])


# def correct_skew(image, delta=1, limit=5):
#     def determine_score(arr, angle):
#         data = inter.rotate(arr, angle, reshape=False, order=0)
#         histogram = np.sum(data, axis=1, dtype=float)
#         score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
#         return histogram, score
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
#     scores = []
#     angles = np.arange(-limit, limit + delta, delta)
#     for angle in angles:
#         histogram, score = determine_score(thresh, angle)
#         scores.append(score)
#
#     best_angle = angles[scores.index(max(scores))]
#
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
#     corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
#             borderMode=cv2.BORDER_REPLICATE)
#
#     return best_angle, corrected


def deskew_hough(input_path, output_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=100, minLineLength=100, maxLineGap=20)

    if lines is None:
        print("No lines detected, skipping rotation.")
        cv2.imwrite(output_path, img)
        return

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) < 30 or abs(angle) > 150:
            angles.append(angle)

    if len(angles) == 0:
        print("No suitable horizontal lines found, skipping rotation.")
        cv2.imwrite(output_path, img)
        return

    print(f"Rotating image {input_path} by {np.median(angles)} degrees.")

    median_angle = np.median(angles)
    rotate_angle = median_angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(output_path, rotated)


def rotate_benchmark(path_to_images):
    images = os.listdir(path_to_images)
    images = [img for img in images if img.endswith(".jpg") or img.endswith(".png")]

    for img in images:
        input_path = os.path.join(path_to_images, img)
        output_path = os.path.join(path_to_images, img)
        deskew_hough(input_path, output_path)

        if output_path.endswith(".jpg"):
            img = Image.open(output_path)
            png_path = output_path.replace(".jpg", ".png")
            img.save(png_path, "PNG")
            os.remove(output_path)


def convert_yolo_to_predictions(root_dir: str, raw_labels_path: str, new_labels_dir: str = "labels", images_dir: str = None):
    if not images_dir:
        images_dir = os.path.join(root_dir, "images")

    os.makedirs(new_labels_dir, exist_ok=True)
    labels = os.listdir(os.path.join(root_dir, raw_labels_path))
    labels = [label for label in labels if label.endswith(".txt")]
    labels = sorted(labels)

    images = os.listdir(images_dir)
    images = [img for img in images if img.endswith(".jpg") or img.endswith(".png")]
    images = sorted(images)


    all_labels = {}
    for label, image in zip(labels, images):
        with open(os.path.join(root_dir, raw_labels_path, label), "r") as f:
            lines = f.readlines()
            lines = [list(map(float, bbox.strip().split())) for bbox in lines]
            lines = [[int(bbox[0])] + bbox[1:] for bbox in lines]


        txt_labels = []
        for line in lines:
            image_opened = Image.open(os.path.join(images_dir, image))
            img_width, img_height = image_opened.size
            class_id = line[0]
            box_center_x = line[1]
            box_center_y = line[2]
            box_width = line[3]
            box_height = line[4]

            x_min = (box_center_x - box_width / 2) * img_width
            y_min = (box_center_y - box_height / 2) * img_height
            x_max = (box_center_x + box_width / 2) * img_width
            y_max = (box_center_y + box_height / 2) * img_height

            txt_labels.append(f"{class_id} {x_min} {y_min} {x_max} {y_max}")
        all_labels[label] = '\n'.join(txt_labels)


    for label in all_labels:
        with open(os.path.join(new_labels_dir, label), "w") as f:
            f.write(all_labels[label])


if __name__ == "__main__":
    # rotate_benchmark("benchmark_dataset/images")
    convert_yolo_to_predictions("../evaluation/predictions/yolo", "labels_raw")