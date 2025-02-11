import wandb
import random
import numpy as np
from PIL import Image
import os


def init_wandb(model_name, dataset_name, params):
    run_name = f"{model_name}_{dataset_name}_{'_'.join([f'{k}{v}' for k, v in params.items()])}"
    wandb.init(
        project="PII_Detection",  # 📌 Проект
        name=run_name,  # Название запуска
        group=model_name,  # 📌 Группа (по моделям)
        config={
            "model": model_name,
            "dataset": dataset_name,
            **params
        },
    )


def log_detection_metrics(
        metrics,
        test_name,
        image_views_path,
    ):

    image_paths = os.listdir(image_views_path)
    image_paths = [os.path.join(image_views_path, file) for file in image_paths if '.DS_Store' not in file]

    # 🔹 1. Логируем таблицу с метриками по классам
    table = wandb.Table(columns=["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "F1"])

    for class_name, metrics in metrics.items():
        table.add_data(class_name, metrics["Images"], metrics["Instances"], metrics["Precision"], metrics["Recall"], metrics["mAP50"], metrics["F1"])

    wandb.log({f"Detection Metrics {test_name}": table})

    table = wandb.Table(columns=["Comparison"])

    for img_path in image_paths:
        table.add_data(wandb.Image(img_path))

    wandb.log({f"Bounding Box Comparisons {test_name}": table})

