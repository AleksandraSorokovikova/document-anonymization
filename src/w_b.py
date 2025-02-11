import wandb
import random
import numpy as np
from PIL import Image


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


def log_detection_metrics(model_name, dataset_name, class_metrics, predictions, ground_truths, image_paths,
                          inference_time, num_images=10):
    """
    Логирует метрики по классам + визуализацию предсказаний в W&B.

    :param model_name: Название модели (Yolov10, LayoutLMv3 и т.д.)
    :param dataset_name: Датасет (random_split / layout_split / benchmark)
    :param class_metrics: Словарь {class_name: {"Precision": val, "Recall": val, "mAP": val, "F1": val}}
    :param predictions: Предсказанные боксы модели
    :param ground_truths: Истинные аннотации
    :param image_paths: Пути к изображениям
    """

    wandb.init(
        project="PII_Detection",
        name=f"{model_name}_{dataset_name}_metrics",
        group=model_name,
        notes=f"Detection metrics on {dataset_name}"
    )

    # 🔹 1. Логируем таблицу с метриками по классам
    table = wandb.Table(columns=["Class", "Images", "Instances", "Precision", "Recall", "mAP", "F1"])

    for class_name, metrics in class_metrics.items():
        table.add_data(class_name, metrics["Images"], metrics["Instances"], metrics["Precision"], metrics["Recall"],
                       metrics["mAP"], metrics["F1"])

    wandb.log({f"Detection Metrics ({dataset_name})": table})

    wandb.log({f"Inference Time ({dataset_name})": inference_time})

    # 🔹 2. Логируем несколько изображений с предсказаниями
    img_table = wandb.Table(columns=["Image", "Ground Truth", "Prediction"])
    indices = random.sample(range(len(image_paths)),
                            min(num_images, len(image_paths)))  # Берём 10 случайных изображений

    for idx in indices:
        img = np.array(Image.open(image_paths[idx]))
        gt = ground_truths[idx]
        pred = predictions[idx]

        img_table.add_data(wandb.Image(img), gt, pred)

    wandb.log({f"Predictions ({dataset_name})": img_table})
    wandb.finish()
