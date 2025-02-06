import json
import openai
from uuid import uuid4
import fitz
import re
import subprocess
import pickle
from datetime import datetime
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
    latex_templates,
    subjects,
    font_family_latex
)
from src.prompts import (
    GENERATE_HTML_CONTENT_SYSTEM_PROMPT,
    GENERATE_HTML_CONTENT_USER_PROMPT,
    USER_PROMPT,
    GENERATE_LATEX_CONTENT_SYSTEM_PROMPT,
    GENERATE_LATEX_CONTENT_USER_PROMPT,
    GENERATE_LATEX_FROM_PICTURE_AND_PII_SYSTEM_PROMPT,
    GENERATE_LATEX_FROM_PICTURE_AND_PII_USER_PROMPT
)
import os
from collections import defaultdict
from tqdm.notebook import tqdm
import random
from datasets import load_dataset
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
        self.create_directories()

    def create_directories(self):
        for folder in ["original", "annotated", "entities", "html", "latex"]:
            if not os.path.exists(os.path.join(self.output_folder, folder)):
                os.makedirs(os.path.join(self.output_folder, folder))

    def generate(self, system_prompt, user_prompt, mes_type='json', temp=0.5, image_path=None):

        if image_path:
            user_message = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
            ]
        else:
            user_message = user_prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=3000,
            n=1,
            temperature=temp,
        )
        message = response.choices[0].message.content.strip()
        if mes_type=="json":
            return json.loads(message.replace("```json", "").replace("```", ""))
        elif mes_type=="html":
            return message.replace("```html", "").replace("```", "")
        elif mes_type=="latex":
            return message.replace("```latex", "").replace("```", "")
        else:
            return message

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
        first_name = random.choice(pii_values['first_name'])
        last_name = random.choice(pii_values['last_name'])
        middle_name = random.choice(pii_values['middle_name']) + ' ' if random.random() <= 0.2 else ''
        full_names = [
            f"{first_name} {middle_name}{last_name}"
            for _ in range(number_of_entities)
        ]
        pii_values["full_name"] = list(set(full_names))
        return pii_values

    @staticmethod
    def adjust_pii(random_pii_entities):
        for pii in random_pii_entities:
            if pii[0] == "full_name":
                if len(pii[1].split()) == 2:
                    if random.random() <= 0.2:
                        # Surname, Name
                        pii[1] = f"{pii[1].split()[1]}, {pii[1].split()[0]}"
                    elif random.random() <= 0.2:
                        # N. Surname
                        pii[1] = f"{pii[1].split()[0][0]}. {pii[1].split()[1]}"

                    add_mr_or_ms = random.random() <= 0.2 and ',' not in pii[1]
                    if add_mr_or_ms:
                        prefix = random.choice(["Mr.", "Ms."])
                        pii[1] = f"{prefix} {pii[1]}"
                if len(pii[1].split()) == 3 and random.random() <= 0.2:
                    # Name M. Surname
                    pii[1] = f"{pii[1].split()[0]} {pii[1].split()[1][0]}. {pii[1].split()[2]}"

            if pii[0] == "email_address":
                if random.random() <= 0.2:
                    entity = pii[1].split('@')
                    pii[1] = entity[0].upper() + '@' + entity[1]

        # make tuples
        for i, pii in enumerate(random_pii_entities):
            random_pii_entities[i] = tuple(pii)

        return random_pii_entities

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

        random_pii_entities = (
            ["full_name"] * random.randint(5, 6) +
            ["address"] * random.randint(3, 4) +
            ["phone_number"] * random.randint(2, 3) +
            ["email_address"] * random.randint(2, 3) +
            ["dates"] * random.randint(2, 3) +
            ["credit_card_number"] * random.randint(1, 2) +
            ["iban"] * random.randint(1, 2) +
            ["company_name"] * random.randint(1, 2) +
            ["vin"] * random.randint(1, 2) +
            ["car_plate"] * random.randint(1, 2)
        )

        random_pii_values = [
            [entity, random.choice(self.pii_values[entity])]
            for entity in random_pii_entities if entity != "signature"
        ]
        first_names = [["first_name", entity[1].split()[0]]
                       for entity in random_pii_values if entity[0] == "full_name"]
        last_names = [["last_name", entity[1].split()[-1]]
                      for entity in random_pii_values if entity[0] == "full_name"]

        random_pii_values.extend(first_names)
        random_pii_values.extend(last_names)

        for i, (entity, value) in enumerate(random_pii_values):
            if random.random() <= 0.2:
                value = value.upper()
                random_pii_values[i] = [entity, value]

        random_pii_values = self.adjust_pii(random_pii_values)

        return random_pii_values

    def generate_latex_document(self, doc_img_folder_path=None):
        random_pii_values = self.create_random_pii_values()
        document_subject = random.choice(subjects)
        random_signature = random.choice(self.signatures_files_paths)
        random_font_family = random.choice(font_family_latex)
        document_layout = random.choice(list(latex_templates.keys()))

        if not doc_img_folder_path:
            latex_content = self.generate(
                GENERATE_LATEX_CONTENT_SYSTEM_PROMPT("\n      - ".join(self.pii_entities)),
                GENERATE_LATEX_CONTENT_USER_PROMPT(
                    random_pii_values,
                    document_subject,
                    latex_templates[document_layout],
                    random_font_family
                ),
                mes_type="latex",
                temp=0.4,
            )
        else:
            images = [img for img in os.listdir(doc_img_folder_path)]
            images = [img for img in images if img.endswith(".jpg") and "train" in img]
            chosen_image = random.choice(images)
            path_to_picture = os.path.join(doc_img_folder_path, chosen_image)
            document_layout = chosen_image.split('.jpg')[0]
            latex_content = self.generate(
                GENERATE_LATEX_FROM_PICTURE_AND_PII_SYSTEM_PROMPT,
                GENERATE_LATEX_FROM_PICTURE_AND_PII_USER_PROMPT(
                    random_pii_values,
                    document_subject,
                    random_font_family
                ),
                image_path=path_to_picture,
                mes_type="latex",
                temp=0.4
            )

        if "path/to/signature.png" in latex_content:
            latex_content = latex_content.replace(
                "path/to/signature.png",
                f"{self.output_folder}/signatures/{random_signature}",
                1
            )
            has_signature = True
            latex_content.replace("path/to/signature.png", "")
        else:
            has_signature = False

        document_meta_info = {
            "document_type": f"from_image_{document_layout}" if doc_img_folder_path else document_layout,
            "font_family": random_font_family,
            "doc_format": "latex",
            "document_subject": document_subject,
            "signature": random_signature if has_signature else None
        }

        return latex_content, document_meta_info, random_pii_values


    def generate_html_document(self):
        random_pii_values = self.create_random_pii_values()

        random_font_family = random.choice(font_family)
        random_signature = random.choice(self.signatures_files_paths)

        document_type = random.choice(document_types)
        chosen_layout = random.choice(layouts)
        chosen_sections = random.sample(sections, 2)
        chosen_headers = random.sample(headers, 3)

        html_content = self.generate(
            GENERATE_HTML_CONTENT_SYSTEM_PROMPT("\n      - ".join(self.pii_entities)),
            GENERATE_HTML_CONTENT_USER_PROMPT(
                document_type,
                random_pii_values,
                random_font_family,
                chosen_layout,
                chosen_sections,
                chosen_headers,
            ),
            mes_type="html",
            temp=0.4,
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
            "doc_format": "html",
            "document_type": document_type,
            "font_family": random_font_family,
            "layout": chosen_layout,
            "sections": chosen_sections,
            "headers": chosen_headers,
            "signature": random_signature if has_signature else None,
        }

        return html_content, document_meta_info, random_pii_values

    @staticmethod
    def extract_bounding_boxes(document, entities):
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
                            lines_y0 = {}
                            for box in entity_boxes:
                                line = box[6]
                                y0 = int(box[1])
                                if y0 not in lines_y0:
                                    lines_y0[y0] = []
                                lines_y0[y0].append(line)
                            lines = {line: i for i, line in enumerate(lines_y0.keys())}
                            parts = {}
                            for box in entity_boxes:
                                # line = box[6]
                                block = box[5]
                                y0 = int(box[1])
                                line = lines[y0]
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
    def insert_graphics_package(latex_content, output_pdf_name):
        if "usepackage{graphicx}" not in latex_content:
            latex_content = latex_content.replace(
                "pt]{article}",
                "pt]{article}\n\\usepackage{graphicx}"
            )
        return latex_content

    def compile_latex(self, latex_content, output_folder, output_pdf_name):
        tex_file_path = os.path.join(output_folder, output_pdf_name.replace('.pdf', '.tex'))
        output_pdf_path = os.path.join(output_folder, output_pdf_name)

        latex_content = self.insert_graphics_package(latex_content, output_pdf_name)

        with open(tex_file_path, 'w') as f:
            f.write(latex_content)

        try:
            subprocess.run(
                ['xelatex', '-interaction=batchmode', '-output-directory', output_folder, tex_file_path],
                stdout=subprocess.DEVNULL,
                check=True
            )
            if not os.path.exists(output_pdf_path):
                print("PDF generation failed.")
        except subprocess.CalledProcessError as e:
            print(f"Error in LaTeX compilation: {e}")

        extensions_to_remove = ['.aux', '.out', '.log', '.tex']
        for ext in extensions_to_remove:
            file_to_remove = os.path.join(output_folder, f'{output_pdf_name.replace(".pdf", "")}{ext}')
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)

    @staticmethod
    def adjust_html_content(html_content):
        existing_font_size = re.findall(r"font-size: \d+px;", html_content)
        new_font_size = int(existing_font_size[0].split(": ")[1].split("px")[0]) - 1
        html_content = re.sub(
            r"font-size: \d+px;", f"font-size: {new_font_size}px;", html_content
        )
        return html_content

    @staticmethod
    def adjust_latex_content(latex_content):
        existing_font_size = int(
            re.findall(r"\\documentclass\[\d+pt\]", latex_content)[0].split("[")[1].split("pt")[0])
        new_font_size = existing_font_size - 1
        latex_content = latex_content.replace(
            f"documentclass[{existing_font_size}pt]",
            f"documentclass[{new_font_size}pt]"
        )
        return latex_content

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

    def create_pdf_from_latex(self, latex_content, file_name):

        self.compile_latex(latex_content, os.path.join(self.output_folder, "original"), file_name)
        pdf = fitz.open(os.path.join(self.output_folder, "original", file_name))
        if pdf.page_count == 2:
            pdf.delete_page(1)
            pdf.save(os.path.join(self.output_folder, "original", file_name))
        elif pdf.page_count > 2:
            raise ValueError("Latex document has more than 1 page")

        with open(
                os.path.join(
                    self.output_folder, "latex", file_name.replace(".pdf", ".tex")
                ),
                "w",
        ) as f:
            f.write(latex_content)

        pdf.save(
            os.path.join(self.output_folder, "original", file_name),
            incremental=True,
            encryption=fitz.PDF_ENCRYPT_KEEP,
        )
        return pdf

    def create_document(self, doc_format, path=None, doc_img_folder_path=None):
        if doc_format == "html":
            html_content, document_meta_info, random_pii_values = (
                self.generate_html_document()
            )
            document_type = document_meta_info["document_type"]
            file_name = f"{document_type.replace(' ', '')}_{str(uuid4())[:7]}.pdf"
            pdf = self.create_pdf_from_html(html_content, file_name)
        elif doc_format == "latex":
            latex_content, document_meta_info, random_pii_values = (
                self.generate_latex_document(doc_img_folder_path)
            )
            document_type = document_meta_info["document_type"]
            file_name = f"{document_type.replace(' ', '')}_{str(uuid4())[:7]}.pdf"
            pdf = self.create_pdf_from_latex(latex_content, file_name)
        else:
            raise ValueError("Format should be either 'html' or 'latex'")

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

        self.draw_bounding_boxes(pdf, bounding_boxes)
        pdf.save(
            os.path.join(
                self.output_folder,
                "annotated",
                f"{file_name.replace('.pdf', '_annotated.pdf')}",
            ),
        )

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

        final_documents = {
                "file_name": file_name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "document_meta_info": document_meta_info,
                "bounding_boxes": bounding_boxes,
                "random_pii_values": random_pii_values,
            }
        self.documents.append(final_documents)
        if path:
            self.append_document_json(final_documents, path)

    def append_document_json(self, document, path):
        try:
            with open(os.path.join(self.output_folder, path), "r") as f:
                existing_documents = json.load(f)
                existing_documents.append(document)
            with open(
                    os.path.join(self.output_folder, path), "w", encoding="utf-8"
            ) as f:
                json.dump(existing_documents, f, indent=4, ensure_ascii=False)
        except FileNotFoundError:
            pass

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
