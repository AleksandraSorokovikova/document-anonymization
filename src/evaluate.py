from src.config import *
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from src.dataset_processing import return_image_with_bounding_boxes


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxA_area + boxB_area - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def match_predictions_to_ground_truths(predictions, ground_truths, iou_threshold=0.5):
    matched_gt = set()

    TP = 0
    FP = 0

    for pred_box in predictions:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            current_iou = iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN = len(ground_truths) - len(matched_gt)
    return TP, FP, FN


def compute_detection_metrics_single_threshold(all_predictions, all_ground_truths, iou_threshold=0.5):
    preds_by_class = defaultdict(list)
    gts_by_class = defaultdict(list)

    for p in all_predictions:
        cls_id, x1, y1, x2, y2 = p
        preds_by_class[int(cls_id)].append((x1, y1, x2, y2))

    for g in all_ground_truths:
        cls_id, x1, y1, x2, y2 = g
        gts_by_class[int(cls_id)].append((x1, y1, x2, y2))

    all_classes = set(list(preds_by_class.keys()) + list(gts_by_class.keys()))

    metrics_per_class = {}
    ap_values = []

    for cls_id in sorted(all_classes):
        preds = preds_by_class[cls_id]
        gts = gts_by_class[cls_id]

        TP, FP, FN = match_predictions_to_ground_truths(preds, gts, iou_threshold)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        ap = precision

        metrics_per_class[cls_id] = {
            'precision': precision,
            'recall': recall,
            'AP': ap,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }
        ap_values.append(ap)

    mAP = sum(ap_values) / len(ap_values) if len(ap_values) > 0 else 0.0
    return metrics_per_class, mAP


def read_files(groundtruth_file, predictions_file):
    with open(groundtruth_file, 'r') as f:
        lines = f.readlines()
        ground_truths = [list(map(float, line.strip().split())) for line in lines]
        ground_truths = [[int(cls_id)] + coords for cls_id, *coords in ground_truths]

    with open(predictions_file, 'r') as f:
        lines = f.readlines()
        predictions = [list(map(float, line.strip().split())) for line in lines]
        predictions = [[int(cls_id)] + coords for cls_id, *coords in predictions]

    return ground_truths, predictions


def create_image_view(image_path, ground_truth_path, predictions_path, output_path):
    img_with_gt_bboxes = return_image_with_bounding_boxes(image_path, ground_truth_path)
    img_with_pred_bboxes = return_image_with_bounding_boxes(image_path, predictions_path)

    width, height = img_with_gt_bboxes.size
    new_image = Image.new("RGB", (2 * width, height))

    new_image.paste(img_with_gt_bboxes, (0, 0))
    new_image.paste(img_with_pred_bboxes, (width, 0))

    new_image.save(output_path)


def evaluate_all_labels(
        ground_truth_dir: str,
        prediction_dir: str,
        iou_threshold: float = 0.5,
        class_names=None,
        create_image_views = False
):
    all_ground_truths = []
    all_predictions = []
    class_image_count = defaultdict(int)
    class_instance_count = defaultdict(int)

    if create_image_views:
        os.makedirs(os.path.join(prediction_dir, "image_views"), exist_ok=True)

    gt_files = sorted(os.listdir(os.path.join(ground_truth_dir, "labels")))
    gt_files = [f for f in gt_files if f.endswith(".txt")]
    n_images = 0

    for filename in gt_files:
        gt_path = os.path.join(ground_truth_dir, "labels", filename)
        pred_path = os.path.join(prediction_dir, "labels", filename)
        if not os.path.isfile(pred_path):
            continue

        if create_image_views:
            image_path = os.path.join(ground_truth_dir, "images", filename.replace(".txt", ".png"))
            output_path = os.path.join(prediction_dir, "image_views", filename.replace(".txt", ".png"))
            create_image_view(image_path, gt_path, pred_path, output_path)

        ground_truths, predictions = read_files(gt_path, pred_path)
        n_images += 1

        all_ground_truths.extend(ground_truths)
        all_predictions.extend(predictions)
        classes_in_this_image = set()
        for (cls_id, x1, y1, x2, y2) in ground_truths:
            classes_in_this_image.add(cls_id)
            class_instance_count[cls_id] += 1

        for cls_id in classes_in_this_image:
            class_image_count[cls_id] += 1

    metrics_per_class, mAP_50 = compute_detection_metrics_single_threshold(
        all_predictions,
        all_ground_truths,
        iou_threshold=iou_threshold
    )
    rows = {}
    class_ids = set(metrics_per_class.keys()) | set(class_instance_count.keys())
    sorted_class_ids = sorted(list(class_ids))

    for cls_id in sorted_class_ids:
        met = metrics_per_class.get(cls_id, {})
        precision = met.get('precision', 0.0)
        recall = met.get('recall', 0.0)
        ap = met.get('AP', 0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if class_names:
            class_label = class_names[cls_id] if cls_id in class_names else str(cls_id)
        else:
            class_label = str(cls_id)
        images_for_class = class_image_count.get(cls_id, 0)
        instances_for_class = class_instance_count.get(cls_id, 0)

        rows[class_label] = {
            "Images": images_for_class,
            "Instances": instances_for_class,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "mAP50": ap,
        }

    total_TP = sum(met.get("TP", 0) for met in metrics_per_class.values())
    total_FP = sum(met.get("FP", 0) for met in metrics_per_class.values())
    total_FN = sum(met.get("FN", 0) for met in metrics_per_class.values())

    global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0

    rows["all"] = {
        "Class": "all",
        "Images": n_images,
        "Instances": sum(class_instance_count.values()),
        "Precision": global_precision,
        "Recall": global_recall,
        "F1": global_f1,
        "mAP50": mAP_50,
    }
    return rows


def create_markdown_report(df_results, model_metadata, output_path):
    try:
        with open(output_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""
        text += f"# Evaluation Report\n\n"

    text += f"## Model and experiment information\n\n"
    text += f"{model_metadata}\n\n"
    text += f"### Results\n\n"
    text += df_results.to_markdown(index=False)
    text += "\n\n---\n\n"

    with open(output_path, "w") as f:
        f.write(text)


def measure_inference_time(model, test_images):
    """
    Измеряет среднее время инференса на одно изображение.

    :param model: Модель (например, YOLO, LayoutLM, DETR)
    :param test_images: Список изображений для инференса
    :return: Среднее время инференса (в секундах)
    """
    times = []

    for img in test_images:
        start_time = time.perf_counter()  # Засекаем время
        _ = model.predict(img)  # Запускаем инференс
        end_time = time.perf_counter()  # Фиксируем время

        times.append(end_time - start_time)

    avg_inference_time = np.mean(times)
    return avg_inference_time


# if __name__ == "__main__":
#     df_results = evaluate_all_labels(
#         "benchmark_dataset",
#         "predictions/yolo",
#         iou_threshold=0.5,
#         class_names=id_to_pii,
#         create_image_views=True
#     )
#     model_metadata = "Model: YOLOv10-Document-Layout-Analysis\n\nImage size: 960x960\n\nEpochs: 100\n\nDataset size: 2k images\n\n"
#     create_markdown_report(df_results, model_metadata, "benchmark_evaluation.md")
