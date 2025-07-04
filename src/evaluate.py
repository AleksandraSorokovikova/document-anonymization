from src.config import *
import os
import time
import numpy as np
import json
from collections import defaultdict
from PIL import Image
import pandas as pd
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


def match_predictions_to_ground_truths(predictions, ground_truths, iou_threshold=0.7):
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


def compute_detection_metrics_single_threshold(all_predictions, all_ground_truths, iou_threshold=0.7):
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
        ground_truth_labels_dir: str,
        prediction_labels_dir: str,
        images_dir: str,
        iou_threshold: float = 0.7,
        create_image_views = False,
        image_views_dir = None
):
    all_ground_truths = []
    all_predictions = []
    class_image_count = defaultdict(int)
    class_instance_count = defaultdict(int)

    class_names = id_to_pii

    if create_image_views:
        image_views_dir = image_views_dir or "image_views"
        os.makedirs(image_views_dir, exist_ok=True)

    gt_files = sorted(os.listdir(ground_truth_labels_dir))
    gt_files = [f for f in gt_files if f.endswith(".txt")]
    n_images = 0

    for filename in gt_files:
        gt_path = os.path.join(ground_truth_labels_dir, filename)
        pred_path = os.path.join(prediction_labels_dir, filename)
        if not os.path.isfile(pred_path):
            continue

        if create_image_views:
            image_path = os.path.join(images_dir, filename.replace(".txt", ".png"))
            output_path = os.path.join(image_views_dir, filename.replace(".txt", ".png"))
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
    times = []

    for img in test_images:
        start_time = time.perf_counter()  # Засекаем время
        _ = model.predict(img)  # Запускаем инференс
        end_time = time.perf_counter()  # Фиксируем время

        times.append(end_time - start_time)

    avg_inference_time = np.mean(times)
    return avg_inference_time


def compute_area(box):
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width * height


def compute_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height


def extract_token_entities(doc, entities_to_exclude=None):
    entities = []
    for token, box, tag in zip(doc["tokens"], doc["bboxes"], doc["ner_tags"]):
        if tag == "O":
            continue
        if tag.startswith("B-") or tag.startswith("I-"):
            entity_type = tag[2:]
        else:
            entity_type = tag
        if entities_to_exclude is not None and entity_type in entities_to_exclude:
            continue
        entities.append({"type": entity_type, "bbox": box})
    return entities


def count_std(file_paths):
    dfs = [pd.read_csv(path).set_index('class') for path in file_paths]

    metrics = ['precision', 'recall']
    std_df = pd.DataFrame(index=dfs[0].index)

    for metric in metrics:
        metric_values = pd.concat(
            [df[metric].rename(f'model_{i + 1}') for i, df in enumerate(dfs)],
            axis=1
        )
        std_df[f'{metric}_std'] = metric_values.std(axis=1)

    std_df = std_df.reset_index()
    std_df['precision_std'] = std_df['precision_std'] * 100
    std_df['recall_std'] = std_df['recall_std'] * 100

    print(std_df[['class', 'precision_std', 'recall_std']])


def calculate_layoutlm_metrics_single(gt_doc, pred_doc, coverage_threshold=0.7):
    gt_entities = extract_token_entities(gt_doc)
    pred_entities = extract_token_entities(pred_doc)

    categories = set([e["type"] for e in gt_entities] + [e["type"] for e in pred_entities])
    metrics = {}

    for cat in categories:
        gt_cat = [e for e in gt_entities if e["type"] == cat]
        pred_cat = [e for e in pred_entities if e["type"] == cat]

        TP = 0
        for gt in gt_cat:
            gt_area = compute_area(gt["bbox"])
            covered = False
            for pred in pred_cat:
                inter_area = compute_intersection(gt["bbox"], pred["bbox"])
                if gt_area > 0 and (inter_area / gt_area) >= coverage_threshold:
                    covered = True
                    break
            if covered:
                TP += 1
        FN = len(gt_cat) - TP

        correct_pred = 0
        for pred in pred_cat:
            covers = False
            for gt in gt_cat:
                gt_area = compute_area(gt["bbox"])
                inter_area = compute_intersection(gt["bbox"], pred["bbox"])
                if gt_area > 0 and (inter_area / gt_area) >= coverage_threshold:
                    covers = True
                    break
            if covers:
                correct_pred += 1
        FP = len(pred_cat) - correct_pred

        precision = correct_pred / len(pred_cat) if len(pred_cat) > 0 else 0.0
        recall = TP / len(gt_cat) if len(gt_cat) > 0 else 1.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FN": FN,
            "FP": FP,
        }

    return metrics


def calculate_layoutlm_metrics_batch(list_gt_docs, list_pred_docs, coverage_threshold=0.7):
    all_gt = []
    all_pred = []
    for gt_doc, pred_doc in zip(list_gt_docs, list_pred_docs):
        all_gt.extend(extract_token_entities(gt_doc))
        all_pred.extend(extract_token_entities(pred_doc))

    categories = set([e["type"] for e in all_gt] + [e["type"] for e in all_pred])
    counts = {cat: {"TP": 0, "FN": 0, "FP": 0, "total_gt": 0, "total_pred": 0} for cat in categories}

    for cat in categories:
        gt_cat = [e for e in all_gt if e["type"] == cat]
        pred_cat = [e for e in all_pred if e["type"] == cat]

        # Recall
        TP = 0
        for gt in gt_cat:
            gt_area = compute_area(gt["bbox"])
            covered = False
            for pred in pred_cat:
                inter_area = compute_intersection(gt["bbox"], pred["bbox"])
                if gt_area > 0 and (inter_area / gt_area) >= coverage_threshold:
                    covered = True
                    break
            if covered:
                TP += 1
        FN = len(gt_cat) - TP

        # Precision
        correct_pred = 0
        for pred in pred_cat:
            covers = False
            for gt in gt_cat:
                gt_area = compute_area(gt["bbox"])
                inter_area = compute_intersection(gt["bbox"], pred["bbox"])
                if gt_area > 0 and (inter_area / gt_area) >= coverage_threshold:
                    covers = True
                    break
            if covers:
                correct_pred += 1
        FP = len(pred_cat) - correct_pred

        counts[cat]["TP"] = TP
        counts[cat]["FN"] = FN
        counts[cat]["FP"] = FP
        counts[cat]["total_gt"] = len(gt_cat)
        counts[cat]["total_pred"] = len(pred_cat)

    metrics = {}
    for cat, c in counts.items():
        TP = c["TP"]
        FN = c["FN"]
        FP = c["FP"]
        total_gt = c["total_gt"]
        total_pred = c["total_pred"]

        precision = (total_pred - FP) / total_pred if total_pred > 0 else 0.0
        recall = TP / total_gt if total_gt > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FN": FN,
            "FP": FP,
        }

    return metrics


def calculate_overall_metrics_single(gt_doc, pred_doc, coverage_threshold=0.7, entities_to_exclude=None):

    gt_entities = extract_token_entities(gt_doc, entities_to_exclude=entities_to_exclude)
    pred_entities = extract_token_entities(pred_doc, entities_to_exclude=entities_to_exclude)

    TP = 0
    for gt in gt_entities:
        gt_area = compute_area(gt["bbox"])
        covered = False
        for pred in pred_entities:
            inter_area = compute_intersection(gt["bbox"], pred["bbox"])
            if gt_area > 0 and (inter_area / gt_area) >= coverage_threshold:
                covered = True
                break
        if covered:
            TP += 1
    FN = len(gt_entities) - TP

    correct_pred = 0
    for pred in pred_entities:
        valid = False
        for gt in gt_entities:
            gt_area = compute_area(gt["bbox"])
            inter_area = compute_intersection(gt["bbox"], pred["bbox"])
            if gt_area > 0 and (inter_area / gt_area) >= coverage_threshold:
                valid = True
                break
        if valid:
            correct_pred += 1
    FP = len(pred_entities) - correct_pred

    precision = correct_pred / len(pred_entities) if len(pred_entities) > 0 else 0.0
    recall = TP / len(gt_entities) if len(gt_entities) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    overall = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "total_gt": len(gt_entities),
    }
    return overall


def count_all_layoutlm_metrics(path_to_gt: str, labels_folder: str, class_names: list):
    all_predictions = os.listdir(labels_folder)

    list_gt_docs = []
    list_pred_docs = []
    predictions_dict = {}

    metrics = {
        cat: defaultdict(float)
        for cat in class_names
    }
    tp_count = {
        cat: 0
        for cat in class_names
    }
    overall_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "TP": 0,
        "FN": 0,
        "FP": 0
    }
    overall_tp_count = 0
    for pred in all_predictions:
        if not pred.endswith(".json"):
            continue
        with open(f"{labels_folder}/{pred}", "r") as f:
            predictions = json.load(f)
            list_pred_docs.append(predictions)
        with open(f"{path_to_gt}/{pred}", "r") as f:
            gt = json.load(f)
            list_gt_docs.append(gt)

        metrics_single = calculate_layoutlm_metrics_single(gt, predictions, coverage_threshold=0.7)
        overall_metrics_single = calculate_overall_metrics_single(gt, predictions, coverage_threshold=0.7)
        predictions_dict[pred] = metrics_single

        for cat in metrics_single:
            if cat not in metrics:
                continue
            tp = metrics_single[cat]["TP"]
            fn = metrics_single[cat]["FN"]
            if tp + fn > 0:
                tp_count[cat] += 1
                metrics[cat]["recall"] += metrics_single[cat]["recall"]
                metrics[cat]["precision"] += metrics_single[cat]["precision"]

        overall_metrics["TP"] += overall_metrics_single["TP"]
        overall_metrics["FN"] += overall_metrics_single["FN"]
        overall_metrics["FP"] += overall_metrics_single["FP"]
        overall_metrics["precision"] += overall_metrics_single["precision"]

        if overall_metrics_single["total_gt"] > 0:
            overall_tp_count += 1
            overall_metrics["recall"] += overall_metrics_single["recall"]


    for cat in metrics:
        if tp_count[cat] == 0:
            metrics[cat]["recall"] = 0
            metrics[cat]["precision"] = 0
        else:
            metrics[cat]["recall"] /= tp_count[cat]
            metrics[cat]["precision"] /= tp_count[cat]

    overall_metrics["precision"] /= overall_tp_count
    overall_metrics["recall"] /= overall_tp_count
    overall_metrics["f1"] = 2 * overall_metrics["precision"] * overall_metrics["recall"] / (overall_metrics["precision"] + overall_metrics["recall"])

    metrics_batch = calculate_layoutlm_metrics_batch(list_gt_docs, list_pred_docs, coverage_threshold=0.7)

    for cat in list(metrics_batch.keys()):
        if cat not in metrics:
            del metrics_batch[cat]

    return metrics, metrics_batch, overall_metrics
