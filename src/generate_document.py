import random
import json
import openai
from uuid import uuid4
import fitz
import re
import pickle
from fuzzywuzzy import fuzz
import pdfkit
from src.config import (
    pii_classes,
    document_types,
    pii_entities_colors,
    font_family,
    sections,
    layouts,
    headers,
)
from src.prompts import (
    GENERATE_HTML_CONTENT_SYSTEM_PROMPT,
    GENERATE_HTML_CONTENT_USER_PROMPT,
    USER_PROMPT,
)
import os
from collections import defaultdict
from tqdm.notebook import tqdm
import random
from datasets import load_dataset


def generate_random_address(entities, n):
    address = []
    orders = [
        ['street', 'city', 'zip_code'],
        ['building_number', 'street', 'city', 'zip_code'],
        ['zip_code', 'city', 'street'],
        ['city', 'street', 'building_number', 'zip_code'],
        ['street', 'zip_code', 'city'],
        ['zip_code', 'street', 'city', 'building_number'],
        ['city', 'zip_code', 'street'],
        ['street', 'building_number', 'zip_code', 'city'],
        ['street', 'building_number'],
        ['building_number', 'street', 'city'],
        ['zip_code', 'city'],
        ['building_number', 'street', 'zip_code'],
    ]

    for _ in range(n):
        order = random.choice(orders)
        join_symbol = random.choice([", ", " "])
        address_entities = [random.choice(entities[key]) for key in order]
        addr = join_symbol.join(address_entities)
        address.append(addr)
    return address


def generate_random_full_name(entities, n):
    full_names = []
    for _ in range(n):
        first_name = random.choice(entities['first_name'])
        last_name = random.choice(entities['last_name'])
        if random.random() <= 0.2:
            middle_name = random.choice(entities['middle_name'])
            full_name = f"{first_name} {middle_name} {last_name}"
        else:
            full_name = f"{first_name} {last_name}"
        full_names.append(full_name)
    return full_names


def load_new_entities(dataset_name="ai4privacy/pii-masking-200k", max_number_of_entities=1000):
    entities = defaultdict(list)
    dataset = load_dataset(dataset_name)

    keys = {
        'FIRSTNAME': 'first_name',
        'LASTNAME': 'last_name',
        'VEHICLEVIN': 'vin',
        'VEHICLEVRM': 'car_plate',
        'EMAIL': 'email_address',
        'DATE': 'dates',
        'MIDDLENAME': 'middle_name',
        'CREDITCARDNUMBER': 'credit_card_number',
        'PHONENUMBER': 'phone_number',
        'IBAN': 'iban',
        'COMPANYNAME': 'company_name',
        'BUILDINGNUMBER': 'building_number',
        'STREET': 'street',
        'CITY': 'city',
        'ZIPCODE': 'zip_code',
    }
    privacy_mask = dataset['train']['privacy_mask']
    random.shuffle(privacy_mask)

    for mask in tqdm(privacy_mask):
        for entity in mask:
            if entity['label'] in keys and len(entities[keys[entity['label']]]) < max_number_of_entities:
                entities[keys[entity['label']]].append(entity['value'])

    entities = dict(entities)
    entities['email_address'] = [email.lower() for email in entities['email_address']]
    entities['building_number'] = [str(b)[:3] for b in entities['building_number']]
    address = generate_random_address(entities, max_number_of_entities)
    names = generate_random_full_name(entities, max_number_of_entities)

    entities['full_name'] = names
    entities['address'] = address

    del entities['city']
    del entities['street']
    del entities['zip_code']
    del entities['building_number']

    return entities


class PIIGenerator:

    def __init__(
            self,
            output_folder="output",
            path_to_pii_values="pii_values.json",
            number_of_entities=None,
            generate_new=False,
    ):
        self.client = openai.Client()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.path_to_pii_values = path_to_pii_values
        self.pii_classes = pii_classes
        self.pii_entities = [pii["class"] for pii in self.pii_classes]
        self.signatures_files_paths = [
            path
            for path in os.listdir(os.path.join(output_folder, "signatures"))
            if path.endswith(".png")
        ]
        existing_pii_values = self.read_generated_pii_from_file(path_to_pii_values)
        if not existing_pii_values:
            self.pii_values = self.generate_pii_entities(number_of_entities)
            self.write_generated_pii_to_file(self.pii_values, path_to_pii_values)
        elif generate_new and number_of_entities:
            self.pii_values = self.generate_pii_entities(number_of_entities)
            for key, value in self.pii_values.items():
                value.extend(existing_pii_values[key])
                self.pii_values[key] = list(set(value))
            self.write_generated_pii_to_file(self.pii_values, path_to_pii_values)
        else:
            self.pii_values = existing_pii_values
        self.output_folder = output_folder
        self.documents = []

    def generate(self, system_prompt, user_prompt, use_json=True, temp=0.5):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=3000,
            n=1,
            temperature=temp,
        )
        message = response.choices[0].message.content.strip()
        if use_json:
            return json.loads(message.replace("```json", "").replace("```", ""))
        return message.replace("```html", "").replace("```", "")

    def generate_pii_entities(self, number_of_entities):
        pii_values = {}
        for pii_entity in self.pii_classes:
            if pii_entity["pii_id"] in ("signature", "full_name"):
                continue
            pii_values[pii_entity["pii_id"]] = list(
                set(
                    self.generate(pii_entity["prompt"], USER_PROMPT(number_of_entities))
                )
            )
        full_names = [
            f"{random.choice(pii_values['first_name'])} {random.choice(pii_values['middle_name']) + ' ' if random.random() <= 0.2 else ''}{random.choice(pii_values['last_name'])}"
            for _ in range(number_of_entities)
        ]
        pii_values["full_name"] = list(set(full_names))
        return pii_values

    def generate_full_names(self, number_of_entities):
        full_names = [
            f"{random.choice(self.pii_values['first_name'])} {random.choice(self.pii_values['middle_name']) + ' ' if random.random() <= 0.2 else ''}{random.choice(self.pii_values['last_name'])}"
            for _ in range(number_of_entities)
        ]
        self.pii_values["full_name"].extend(full_names)
        self.pii_values["full_name"] = list(set(self.pii_values["full_name"]))
        self.write_generated_pii_to_file(self.pii_values, self.path_to_pii_values)
        return full_names

    def generate_new_pii_entity(self, entity, write=True):
        number_of_entities = len(self.pii_values[list(self.pii_values.keys())[0]])
        print(f"Generating {number_of_entities} {entity} entities")
        prompt = [pii for pii in self.pii_classes if pii["pii_id"] == entity][0]["prompt"]
        generated_entities = list(set(self.generate(prompt, USER_PROMPT(number_of_entities))))
        self.pii_values[entity] = generated_entities
        if write:
            self.write_generated_pii_to_file(self.pii_values, self.path_to_pii_values)
        return generated_entities

    def update_pii_values(self, entity, new_values):
        if entity in self.pii_values:
            self.pii_values[entity].extend(new_values)
            self.pii_values[entity] = list(set(self.pii_values[entity]))
            self.write_generated_pii_to_file(self.pii_values, self.path_to_pii_values)
        else:
            self.pii_values[entity] = new_values
            self.write_generated_pii_to_file(self.pii_values, self.path_to_pii_values)

    def create_random_pii_values(self):

        generate_vin = int(random.random() <= 0.35)
        generate_car_plate = int(random.random() <= 0.35)

        random_pii_entities = (
            ["full_name"] * random.randint(2, 3) +
            ["address"] * random.randint(1, 2) +
            ["phone_number"] * random.randint(1, 2) +
            ["email_address"] * random.randint(1, 2) +
            ["dates"] * random.randint(1, 2) +
            ["credit_card_number"] * random.randint(0, 1) +
            ["iban"] * random.randint(0, 1) +
            ["company_name"] * random.randint(0, 1) +
            ["vin"] * generate_vin +
            ["car_plate"] * generate_car_plate
        )

        random_pii_values = [
            (entity, random.choice(self.pii_values[entity]))
            for entity in random_pii_entities if entity != "signature"
        ]
        first_names = [("first_name", entity[1].split()[0]) for entity in random_pii_values if entity[0] == "full_name"]
        first_names = first_names[:random.randint(0, max(len(first_names) - 1, 1))]
        if first_names:
            random_pii_values.extend(first_names)

        for i, (entity, value) in enumerate(random_pii_values):
            if random.random() <= 0.15:
                value = value.upper()
                random_pii_values[i] = (entity, value)

        return random_pii_values

    def generate_html_document(self):
        random_pii_values = self.create_random_pii_values()
        document_type = random.choice(document_types)

        random_font_family = random.choice(font_family)
        random_signature = random.choice(self.signatures_files_paths)

        chosen_layout = random.choice(layouts)
        chosen_sections = random.sample(sections, 2)
        chosen_headers = random.sample(headers, 3)

        html_content = self.generate(
            GENERATE_HTML_CONTENT_SYSTEM_PROMPT("\n      - ".join(self.pii_entities)),
            GENERATE_HTML_CONTENT_USER_PROMPT(
                document_type,
                [entity[1] for entity in random_pii_values],
                random_font_family,
                chosen_layout,
                chosen_sections,
                chosen_headers,
            ),
            use_json=False,
            temp=0.2,
        )

        if "path/to/signature.png" in html_content:
            html_content = html_content.replace(
                "path/to/signature.png",
                os.path.abspath(f"{self.output_folder}/signatures/{random_signature}"),
            )
            has_signature = True
        else:
            has_signature = False

        document_meta_info = {
            "document_type": document_type,
            "font_family": random_font_family,
            "layout": chosen_layout,
            "sections": chosen_sections,
            "headers": chosen_headers,
            "signature": random_signature if has_signature else None,
        }

        return html_content, document_meta_info, random_pii_values

    def extract_bounding_boxes(self, document, entities):
        entity_bounding_boxes = []
        found_entities = []
        page = document[0]
        words = page.get_text("words")

        for entity, value in entities:
            entity_boxes = []
            entity_words = value.split()
            entity_index = 0

            for i, word in enumerate(words):
                if (
                        fuzz.WRatio(word[4].lower(), entity_words[entity_index].lower())
                        >= 95
                        or word[4] == entity_words[entity_index]
                ):
                    entity_boxes.append(word)
                    entity_index += 1

                    if entity_index == len(entity_words):

                        lines = [word[6] for word in entity_boxes]
                        blocks = [word[5] for word in entity_boxes]
                        # if all lines are the same, then it is a single line entity
                        # otherwise, it is a multi-line entity and it should be separated
                        if len(set(lines)) == 1 and len(set(blocks)) == 1:
                            x0 = min(box[0] for box in entity_boxes)
                            y0 = min(box[1] for box in entity_boxes)
                            x1 = max(box[2] for box in entity_boxes)
                            y1 = max(box[3] for box in entity_boxes)

                            entity_bounding_boxes.append(
                                (entity, value, (x0, y0, x1, y1))
                            )
                            found_entities.append((entity, value))
                            entity_boxes = []
                            entity_index = 0
                        else:
                            # group words by line and block
                            parts = {}
                            for box in entity_boxes:
                                line = box[6]
                                block = box[5]
                                if (line, block) not in parts:
                                    parts[(line, block)] = []
                                parts[(line, block)].append(box)

                            for part, boxes in parts.items():
                                x0 = min(box[0] for box in boxes)
                                y0 = min(box[1] for box in boxes)
                                x1 = max(box[2] for box in boxes)
                                y1 = max(box[3] for box in boxes)

                                entity_bounding_boxes.append(
                                    (entity, value, (x0, y0, x1, y1))
                                )
                                found_entities.append((entity, value))
                            entity_boxes = []
                            entity_index = 0

                else:
                    entity_boxes = []

        return list(set(entity_bounding_boxes)), list(set(found_entities))

    @staticmethod
    def extract_signature_bounding_boxes(document):
        try:
            page = document[0]
            image = page.get_image_info()[-1]
            bbox = image["bbox"]

            return bbox
        except:
            return None

    @staticmethod
    def check_entities(found_entities, entities, document_page):
        missing_entities = []
        not_embedded_entities = []
        document_page = document_page[0]
        document_text = document_page.get_text("text").replace("\n", " ")
        for entity, value in entities:
            if entity == "signature":
                continue
            if (entity, value) not in found_entities and value in document_text:
                missing_entities.append((entity, value))
        for entity, value in entities:
            if entity == "signature":
                continue
            if value not in document_text:
                not_embedded_entities.append((entity, value))
        return missing_entities, not_embedded_entities

    @staticmethod
    def draw_bounding_boxes(document, bounding_boxes):
        document_page = document[0]
        for entity, value, inst in bounding_boxes:
            rect = fitz.Rect(inst[0], inst[1], inst[2], inst[3])
            document_page.draw_rect(
                rect, color=pii_entities_colors[entity], overlay=True, width=1
            )

    @staticmethod
    def write_generated_pii_to_file(pii_entities, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pii_entities, f, indent=4, ensure_ascii=False)

    @staticmethod
    def read_generated_pii_from_file(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def adjust_html_content(html_content):
        existing_font_size = re.findall(r"font-size: \d+px;", html_content)
        new_font_size = int(existing_font_size[0].split(": ")[1].split("px")[0]) - 1
        html_content = re.sub(
            r"font-size: \d+px;", f"font-size: {new_font_size}px;", html_content
        )
        return html_content

    def create_pdf_from_html(self, html_content, file_name):

        while True:
            pdfkit.from_string(
                html_content,
                os.path.join(self.output_folder, "original", file_name),
                options={"enable-local-file-access": ""},
            )
            pdf = fitz.open(os.path.join(self.output_folder, "original", file_name))
            number_of_pages = pdf.page_count
            if number_of_pages > 1:
                html_content = self.adjust_html_content(html_content)
            else:
                break

        with open(
                os.path.join(
                    self.output_folder, "html", file_name.replace(".pdf", ".html")
                ),
                "w",
        ) as f:
            f.write(html_content)

        pdf.save(
            os.path.join(self.output_folder, "original", file_name),
            incremental=True,
            encryption=fitz.PDF_ENCRYPT_KEEP,
        )
        return pdf

    def create_document(self):
        html_content, document_meta_info, random_pii_values = (
            self.generate_html_document()
        )
        document_type = document_meta_info["document_type"]
        file_name = f"{document_type.replace(' ', '')}_{str(uuid4())[:6]}.pdf"
        pdf = self.create_pdf_from_html(html_content, file_name)
        bounding_boxes, found_entities = self.extract_bounding_boxes(
            pdf, random_pii_values
        )
        if document_meta_info["signature"] is not None:
            signature_bbox = self.extract_signature_bounding_boxes(pdf)
            random_pii_values.append(("signature", document_meta_info["signature"]))
            if signature_bbox:
                bounding_boxes.append(
                    ("signature", document_meta_info["signature"], signature_bbox)
                )
        # missing_entities, not_embedded_entities = self.check_entities(
        #     found_entities, random_pii_values, pdf
        # )
        # if missing_entities or not_embedded_entities:
        #     print(f"Entities not found in file {file_name}")
        #     print(f"Missing entities: {missing_entities}")
        #     print(f"Not embedded entities: {not_embedded_entities}")
        #     print("\n")

        self.draw_bounding_boxes(pdf, bounding_boxes)
        pdf.save(
            os.path.join(
                self.output_folder,
                "annotated",
                f"{file_name.replace('.pdf', '_annotated.pdf')}",
            ),
        )
        # save bounding_boxes pickle
        with open(
                os.path.join(
                    self.output_folder,
                    "entities",
                    f"{file_name.replace('.pdf', '_bounding_boxes.json')}",
                ),
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(bounding_boxes, f, indent=4, ensure_ascii=False)

        self.documents.append(
            {
                "document_meta_info": document_meta_info,
                "bounding_boxes": bounding_boxes,
                "file_name": file_name,
                "random_pii_values": random_pii_values,
            }
        )

    def save_documents(self, path, json_format=True):
        try:
            with open(os.path.join(self.output_folder, path), "r") as f:
                existing_documents = json.load(f)
                self.documents.extend(existing_documents)
        except FileNotFoundError:
            print("No existing documents found")
        if json_format:
            with open(
                    os.path.join(self.output_folder, path), "w", encoding="utf-8"
            ) as f:
                json.dump(self.documents, f, indent=4, ensure_ascii=False)
        else:
            with open(os.path.join(self.output_folder, path), "wb") as f:
                pickle.dump(self.documents, f)
