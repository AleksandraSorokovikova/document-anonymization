from datasets import load_from_disk
from transformers import AutoProcessor
from datasets.features import ClassLabel, Value, Array2D, Array3D, Features, Sequence
from functools import partial
from transformers import LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from evaluate import load
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from src.config import pii_entities_colors_names, id_to_pii


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def normalize_bboxes(bboxes, img):
    orig_width, orig_height = img.size
    return [
        [
            int((x_min / orig_width) * 1000),
            int((y_min / orig_height) * 1000),
            int((x_max / orig_width) * 1000),
            int((y_max / orig_height) * 1000),
        ]
        for x_min, y_min, x_max, y_max in bboxes
    ]


def prepare_examples(
        examples, processor, image_column_name, text_column_name, boxes_column_name, label_column_name
):
    images = examples[image_column_name]
    images = [img.convert("RGB") for img in images]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    normalized_boxes = [normalize_bboxes(bbox, img) for bbox, img in zip(boxes, images)]
    for b in normalized_boxes:
        for bbox in b:
            for coord in bbox:
                assert 0 <= coord <= 1000, f"❌ Координата {coord} > 1000"

    encoding = processor(
        images, words, boxes=normalized_boxes, word_labels=word_labels, truncation=True, padding="max_length"
    )

    return encoding


def check_bbox_inputs(train_dataset):
    for i in range(len(train_dataset)):
        for bbox in train_dataset[i]['bbox']:
            if ((bbox[0] <= bbox[2]) or (bbox[1] <= bbox[3]) or (bbox[0] < 0) or
                    (bbox[1] < 0) or (bbox[2] > 1000) or (bbox[3] > 1000)):
                return False
    return True


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def create_image_view(img_with_gt_bboxes, img_with_pred_bboxes, output_path):
    width, height = img_with_gt_bboxes.size
    new_image = Image.new("RGB", (2 * width, height))

    new_image.paste(img_with_gt_bboxes, (0, 0))
    new_image.paste(img_with_pred_bboxes, (width, 0))

    new_image.save(output_path)


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
        class_name = id_to_pii.get(class_label, -1)
        color = pii_entities_colors_names.get(class_name, "black")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    return img


def draw_prediction(model, processor, image, draw_text=False):
    encoding = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()

    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size

    true_predictions = [model.config.id2label[pred] for pred in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    def iob_to_label(label):
        label = label[2:]
        if not label:
            return 'other'
        return label

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        if predicted_label not in pii_entities_colors_names:
            continue
        draw.rectangle(box, outline=pii_entities_colors_names[predicted_label])
        if draw_text:
            draw.text(
                (box[0] + 10, box[1] - 10),
                text=predicted_label,
                fill=pii_entities_colors_names[predicted_label],
                font=font
            )

    return image


def compute_metrics(p, label_list, metric, return_entity_level_metrics):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# def compute_token_metrics(p, label_list):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_labels_flat = []
#     pred_labels_flat = []

#     for pred_seq, label_seq in zip(predictions, labels):
#         for p, l in zip(pred_seq, label_seq):
#             if l != -100:
#                 pred_labels_flat.append(p)
#                 true_labels_flat.append(l)

#     precision = precision_score(true_labels_flat, pred_labels_flat, average="micro", zero_division=0)
#     recall = recall_score(true_labels_flat, pred_labels_flat, average="micro", zero_division=0)
#     f1 = f1_score(true_labels_flat, pred_labels_flat, average="micro", zero_division=0)
#     accuracy = accuracy_score(true_labels_flat, pred_labels_flat)

#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "accuracy": accuracy,
#     }


# def compute_metrics(p, label_list, metric, return_entity_level_metrics=False):
#     import numpy as np

#     def normalize(label):
#         if label == "O":
#             return "O"
#         entity = label.split("-")[-1]  # full_name → full_name, B-full_name → full_name
#         return f"B-{entity}"           # всегда делаем B-entity

#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_predictions = []
#     true_labels = []

#     for pred, label in zip(predictions, labels):
#         pred_labels = []
#         true_label_seq = []
#         for p_val, l_val in zip(pred, label):
#             if l_val != -100:
#                 if p_val < len(label_list) and l_val < len(label_list):
#                     pred_label = normalize(label_list[p_val])
#                     true_label = normalize(label_list[l_val])
#                     pred_labels.append(pred_label)
#                     true_label_seq.append(true_label)
#         true_predictions.append(pred_labels)
#         true_labels.append(true_label_seq)

#     results = metric.compute(predictions=true_predictions, references=true_labels)

#     if return_entity_level_metrics:
#         # Расплющиваем метрики по сущностям
#         flat_results = {}
#         for key, value in results.items():
#             if isinstance(value, dict):
#                 for sub_key, score in value.items():
#                     flat_results[f"{key}_{sub_key}"] = score
#             else:
#                 flat_results[key] = value
#         return flat_results
#     else:
#         return {
#             "precision": results.get("overall_precision", 0.0),
#             "recall": results.get("overall_recall", 0.0),
#             "f1": results.get("overall_f1", 0.0),
#             "accuracy": results.get("overall_accuracy", 0.0),
#         }


def prepare_dataset(
        path_to_dataset,
        processor_name="microsoft/layoutlmv3-base",
        label_list=None,
        id2label=None,
        label2id=None,
):
    dataset = load_from_disk(path_to_dataset)
    processor = AutoProcessor.from_pretrained(processor_name, apply_ocr=False)

    features = dataset["train"].features
    column_names = dataset["train"].column_names
    image_column_name = "image"
    text_column_name = "tokens"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"

    if label_list is None:
        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            id2label = {k: v for k, v in enumerate(label_list)}
            label2id = {v: k for k, v in enumerate(label_list)}
        else:
            label_list = get_label_list(dataset["train"][label_column_name])
            id2label = {k: v for k, v in enumerate(label_list)}
            label2id = {v: k for k, v in enumerate(label_list)}

    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    train_dataset = dataset["train"].map(
        partial(
            prepare_examples,
            processor=processor,
            image_column_name=image_column_name,
            text_column_name=text_column_name,
            boxes_column_name=boxes_column_name,
            label_column_name=label_column_name,
        ),
        batched=True,
        features=features,
        remove_columns=column_names,
    )

    eval_dataset = dataset["val"].map(
        partial(
            prepare_examples,
            processor=processor,
            image_column_name=image_column_name,
            text_column_name=text_column_name,
            boxes_column_name=boxes_column_name,
            label_column_name=label_column_name,
        ),
        batched=True,
        features=features,
        remove_columns=column_names,
    )

    test_dataset = dataset["test"].map(
        partial(
            prepare_examples,
            processor=processor,
            image_column_name=image_column_name,
            text_column_name=text_column_name,
            boxes_column_name=boxes_column_name,
            label_column_name=label_column_name,
        ),
        batched=True,
        features=features,
        remove_columns=column_names,
    )

    return train_dataset, eval_dataset, test_dataset, processor, id2label, label2id, label_list


def prepare_trainer(
        path_to_dataset,
        path_to_model_weights,
        model_name="microsoft/layoutlmv3-base",
        processor_name="microsoft/layoutlmv3-base",
        num_train_epochs=10,
        learning_rate=1e-5,
        return_entity_level_metrics=False,
        already_trained=False,
        metric_for_best_model="f1",
        per_device_train_batch_size=4,
        new_labels=None,
        layers_to_train=[10, 11],
):
    metric = load("seqeval")

    label_list = None
    id2label = None
    label2id = None

    if already_trained:
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        label_list = list(model.config.id2label.values())

        if new_labels is not None:
            label_list += new_labels
            id2label = {k: v for k, v in enumerate(label_list)}
            label2id = {v: k for k, v in enumerate(label_list)}
            print(label_list)
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
            )
        else:
            id2label = model.config.id2label
            label2id = model.config.label2id

        # for param in model.base_model.parameters():
        #     param.requires_grad = False

        for name, param in model.base_model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in layers_to_train):
                param.requires_grad = True
            else:
                param.requires_grad = False

    train_dataset, eval_dataset, test_dataset, processor, id2label, label2id, label_list = prepare_dataset(
        path_to_dataset, processor_name=processor_name, label_list=label_list, id2label=id2label, label2id=label2id
    )
    print(label_list)

    if not already_trained:
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name, id2label=id2label, label2id=label2id
        )

    training_args = TrainingArguments(
        output_dir=path_to_model_weights,
        # max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=2,
        learning_rate=learning_rate,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=partial(
            compute_metrics,
            label_list=label_list,
            metric=metric,
            return_entity_level_metrics=return_entity_level_metrics
        )
    )

    return trainer, processor, model, test_dataset
