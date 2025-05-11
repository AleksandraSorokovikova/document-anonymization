import wandb
import os
from src.evaluate import count_all_layoutlm_metrics


def init_wandb(model_name, dataset_name, params):
    run_name = f"{model_name}_{dataset_name}_{'_'.join([f'{k}{v}' for k, v in params.items()])}"
    wandb.init(
        project="PII_Detection",
        name=run_name,
        group=model_name,
        config={
            "model": model_name,
            "dataset": dataset_name,
            **params
        },
    )


def log_detection_metrics(
        metrics: dict,
        test_name: str,
        metric_columns: list,
        image_views_path: str = None,
):
    data = []
    for class_name, metric_dict in metrics.items():
        try:
            row = [class_name] + [metric_dict[col] for col in metric_columns]
            data.append(row)
        except KeyError as e:
            print(f"⚠️ Missing metric {e} for class {class_name}. Skipping this class.")

    table = wandb.Table(columns=["class"] + metric_columns, data=data)
    wandb.log({
        f"Metrics Table {test_name}": table,
    }, step=0)

    if image_views_path:
        image_paths = [
            os.path.join(image_views_path, file)
            for file in os.listdir(image_views_path)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if image_paths:
            table = wandb.Table(columns=["Comparison"])

            for img_path in image_paths:
                table.add_data(wandb.Image(img_path))

            wandb.log({f"Bboxes Comparisons {test_name}": table}, step=0)
        else:
            print("⚠️ No images found in the specified path.")

def log_lm_metrics(
        path_to_gt_benchmark_labels,
        predicted_benchmark_labels_folder,
        test_name,
        predicted_image_view_path,
        class_names,
):
    metrics_per_documents, metrics_batch, overall_metrics = count_all_layoutlm_metrics(
        path_to_gt_benchmark_labels, predicted_benchmark_labels_folder, class_names
    )

    log_detection_metrics(
        metrics_batch,
        test_name=f"batch metrics {test_name}",
        metric_columns=["recall", "precision", "f1", "TP", "FP", "FN"],
        image_views_path=predicted_image_view_path,
    )

    log_detection_metrics(
        metrics_per_documents,
        test_name=f"per document metrics {test_name}",
        metric_columns=["recall", "precision"],
    )

    log_detection_metrics(
        {
            "overall": overall_metrics
        },
        test_name=f"overall metrics {test_name}",
        metric_columns=["recall", "precision", "f1"],
    )

"""
[
        {   
            "test_name": "benchmark",
            "gt_labels": path_to_gt_benchmark_labels,
            "predicted_labels": predicted_benchmark_labels_folder,
            "image_views": predicted_image_view_path,
        },
        {
            "test_name": "synthetic",
            "gt_labels": path_to_gt_synthetic_labels,
            "predicted_labels": predicted_synthetic_labels_folder,
            "image_views": predicted_image_view_path,
        },
]
"""

def count_and_log_all_metrics(
        samples,
        lm_model_name,
        ocr_model_name,
        run_specification=None
):
    run_name = f"{lm_model_name}_{ocr_model_name}" 
    run_name += '' if run_specification is None else f"_{run_specification}"
    wandb.init(
        project="PII_Detection",
        name=run_name,
        group=lm_model_name,
        config={
            "model": lm_model_name,
        },
    )

    for sample in samples:
        log_lm_metrics(
            sample["gt_labels"],
            sample["predicted_labels"],
            sample["test_name"],
            sample["image_views"],
            sample["class_names"],
        )

    wandb.finish()
