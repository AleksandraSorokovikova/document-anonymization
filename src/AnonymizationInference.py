from transformers import AutoModelForTokenClassification, AutoProcessor
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont, Image
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from src.process import create_image_view
from doctr.models import ocr_predictor
from src.config import pii_entities_colors_rgba


class AnonymizationInference:
    def __init__(
            self,
            path_to_layoutlm_weights,
            path_to_signature_weights,
            apply_lm_ocr=False,
            apply_ocr=True,
            detection_model="db_resnet50",  # "fast_base", "db_resnet50"
            recognition_model="crnn_vgg16_bn",  # "sar_resnet31", "vitstr_base", "crnn_vgg16_bn", "parseq", "master"
            label_list=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForTokenClassification.from_pretrained(path_to_layoutlm_weights).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=apply_lm_ocr, is_split_into_words=True
        )
        self.signature_detector = YOLO(path_to_signature_weights)
        self.label_list = label_list
        self.apply_lm_ocr = apply_lm_ocr
        if not apply_lm_ocr and apply_ocr:
            self.ocr_model = ocr_predictor(
                detection_model, recognition_model, pretrained=True,
                export_as_straight_boxes=True, assume_straight_pages=True
            ).to(self.device)

    @staticmethod
    def unnormalize_box(bboxes, width, height):
        return [
            [
                int(width * (x_min / 1000)),
                int(height * (y_min / 1000)),
                int(width * (x_max / 1000)),
                int(height * (y_max / 1000)),
            ]
            for x_min, y_min, x_max, y_max in bboxes
        ]

    @staticmethod
    def normalize_bboxes(bboxes, width, height):
        return [
            [
                int((x_min / width) * 1000),
                int((y_min / height) * 1000),
                int((x_max / width) * 1000),
                int((y_max / height) * 1000),
            ]
            for x_min, y_min, x_max, y_max in bboxes
        ]

    def predict_signatures(self, image_path, words, result_boxes, ner_tags):
        results = self.signature_detector.predict(image_path, verbose=False)
        bboxes = []
        for result in results:
            boxes = result.boxes.xyxy
            bboxes.extend(boxes.cpu().numpy().astype(int).tolist())

        for box in bboxes:
            words.append("[SIGNATURE]")
            result_boxes.append(box)
            ner_tags.append("B-signature")

        return words, result_boxes, ner_tags

    def convert_ocr_result_to_bboxes(self, ocr_result, width, height):
        words = []
        boxes = []
        for page in ocr_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        (rel_x0, rel_y0), (rel_x1, rel_y1) = word.geometry
                        abs_x0 = int(rel_x0 * width)
                        abs_y0 = int(rel_y0 * height)
                        abs_x1 = int(rel_x1 * width)
                        abs_y1 = int(rel_y1 * height)
                        words.append(word.value)
                        boxes.append([abs_x0, abs_y0, abs_x1, abs_y1])
        cleaned_words = []
        cleaned_boxes = []

        for word, box in zip(words, boxes):
            if word.strip() and word.isascii() and len(word) < 50:
                cleaned_words.append(word)
                cleaned_boxes.append(box)

        normalized_boxes = self.normalize_bboxes(cleaned_boxes, width, height)

        return cleaned_words, normalized_boxes

    def get_token_length(self, word):
        tokenized = self.processor.tokenizer.tokenize(word)
        return len(tokenized)

    def split_chunks(self, image, words, boxes):
        max_tokens = 500
        chunks = []
        current_words = []
        current_boxes = []
        current_token_count = 0

        for word, box in zip(words, boxes):
            word_tokens = self.get_token_length(word)
            if current_token_count + word_tokens > max_tokens:
                chunks.append((current_words, current_boxes))
                current_words = [word]
                current_boxes = [box]
                current_token_count = word_tokens
            else:
                current_words.append(word)
                current_boxes.append(box)
                current_token_count += word_tokens

        if current_words:
            chunks.append((current_words, current_boxes))

        encoding_chunks = []
        for chunk_words, chunk_boxes in chunks:
            encoding = self.processor(
                image, chunk_words, boxes=chunk_boxes, return_tensors="pt"
            ).to(self.device)
            encoding_chunks.append(encoding)

        return encoding_chunks

    def convert_tokens_to_words(self, word_ids, token_predictions):
        word_predictions = []
        current_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != current_word_id:
                current_word_id = word_id
                word_predictions.append(token_predictions[idx])
        labels = [self.model.config.id2label[pred] for pred in word_predictions]

        return labels

    def remove_labels(self, ner_tags):
        for i, tag in enumerate(ner_tags):
            if tag == "O":
                continue
            raw_tag = tag.split("-")[-1]
            if raw_tag not in self.label_list:
                ner_tags[i] = "O"
        return ner_tags

    def process_image_with_lm_ocr(self, image_path):
        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(image, return_tensors="pt")
        words = encoding["tokens"]
        boxes = encoding["bboxes"]

        return [encoding], words, boxes

    def process_image_with_doctr_ocr(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_page = np.array(image)
        result = self.ocr_model([input_page])
        image_width, image_height = image.size
        words, boxes = self.convert_ocr_result_to_bboxes(result, image_width, image_height)
        encoding_chunks = self.split_chunks(image, words, boxes)
        boxes = self.unnormalize_box(boxes, image_width, image_height)

        return encoding_chunks, words, boxes

    def process_image_with_ocr(self, image_path):
        if not self.apply_lm_ocr:
            return self.process_image_with_doctr_ocr(image_path)
        return self.process_image_with_lm_ocr(image_path)

    def process_image(self, image_path, words, boxes):
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        normalized_boxes = self.normalize_bboxes(boxes, image_width, image_height)
        encoding_chunks = self.split_chunks(image, words, normalized_boxes)

        return encoding_chunks

    def predict(self, image_path, words=None, boxes=None):
        if words is None and boxes is None:
            encoding_chunks, words, boxes = self.process_image_with_ocr(image_path)
        else:
            encoding_chunks = self.process_image(image_path, words, boxes)

        ner_tags_predictions = []

        for encoding in encoding_chunks:
            with torch.no_grad():
                outputs = self.model(**encoding)
            token_predictions = outputs.logits.argmax(-1).squeeze().tolist()
            word_ids = encoding.word_ids(0)
            predictions = self.convert_tokens_to_words(word_ids, token_predictions)
            ner_tags_predictions.extend(predictions)

        assert len(ner_tags_predictions) == len(words)

        words, boxes, ner_tags_predictions = self.predict_signatures(
            image_path, words, boxes, ner_tags_predictions
        )

        if self.label_list:
            ner_tags_predictions = self.remove_labels(ner_tags_predictions)

        return {
            "tokens": words,
            "bboxes": boxes,
            "ner_tags": ner_tags_predictions
        }

    @staticmethod
    def draw_bboxes(image_path, predictions, add_text=False):
        image = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        for word, box, pred in zip(predictions["tokens"], predictions["bboxes"], predictions["ner_tags"]):
            if pred == "O":
                continue
            pred = pred.split("-")[-1]
            color = pii_entities_colors_rgba.get(pred, "black")
            draw.rectangle(box, fill=color)
            if add_text:
                draw.text((box[0], box[1] - 10), f"{pred}", font=font, fill=color)
        combined = Image.alpha_composite(image, overlay)
        return combined

    def get_layoutlm_predictions(
            self, images, path_to_image, path_to_gt_labeled_images, labels_saving_path, image_views_path
    ):
        for image in tqdm(images):
            if not image.endswith(".png"):
                continue

            image_path = os.path.join(path_to_image, image)
            predictions = self.predict(image_path)
            img_with_pred_bboxes = self.draw_bboxes(image_path, predictions)
            img_with_gt_bboxes = Image.open(os.path.join(path_to_gt_labeled_images, image))
            create_image_view(img_with_gt_bboxes, img_with_pred_bboxes, f"{image_views_path}/{image}")
            label_name = image.replace(".png", ".json")
            with open(f"{labels_saving_path}/{label_name}", "w") as f:
                json.dump(predictions, f, indent=4)
