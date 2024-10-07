from src.prompts import *


pii_classes = [
    {
        "class": "First Name",
        "pii_id": "first_name",
        "prompt": GENERATE_FIRST_NAMES_SYSTEM_PROMPT,
    },
    {
        "class": "Last Name",
        "pii_id": "last_name",
        "prompt": GENERATE_LAST_NAMES_SYSTEM_PROMPT,
    },
    {
        "class": "Middle Name",
        "pii_id": "middle_name",
        "prompt": GENERATE_MIDDLE_NAMES_SYSTEM_PROMPT,
    },
    {
        "class": "Address",
        "pii_id": "address",
        "prompt": GENERATE_ADDRESS_SYSTEM_PROMPT
    },
    {
        "class": "Phone Number",
        "pii_id": "phone_number",
        "prompt": GENERATE_PHONE_NUMBER_SYSTEM_PROMPT,
    },
    {
        "class": "Email Address",
        "pii_id": "email_address",
        "prompt": GENERATE_EMAIL_ADDRESS_SYSTEM_PROMPT,
    },
    {
        "class": "Dates",
        "pii_id": "dates",
        "prompt": GENERATE_DATES_SYSTEM_PROMPT,
    },
    {
        "class": "Credit Card Number",
        "pii_id": "credit_card_number",
        "prompt": GENERATE_CREDIT_CARD_NUMBER_SYSTEM_PROMPT,
    },
    {
        "class": "IBAN",
        "pii_id": "iban",
        "prompt": GENERATE_IBAN_SYSTEM_PROMPT
    },
    {
        "class": "Company Name",
        "pii_id": "company_name",
        "prompt": GENERATE_COMPANY_NAMES_SYSTEM_PROMPT,
    },
    {
        "class": "Signature",
        "pii_id": "signature",
        "prompt": None,
    },
    {
        "class": "Full Name",
        "pii_id": "full_name",
        "prompt": None,
    },
    {
        "class": "VIN",
        "pii_id": "vin",
        "prompt": GENERATE_VIN_SYSTEM_PROMPT,
    },
    {
        "class": "Car Plate",
        "pii_id": "car_plate",
        "prompt": GENERATE_CAR_PLATE_NUMBER_SYSTEM_PROMPT,
    },
]

pii_to_id = {pii["pii_id"]: i for i, pii in enumerate(pii_classes)}
id_to_pii = {i: pii["pii_id"] for i, pii in enumerate(pii_classes)}

document_types = [
    "Invoice",
    "Receipt",
    "Letter",
    "Email",
    "Bank Statement",
    "Insurance Claim Form",
    "Medical Report",
    "Employment Contract",
    "Tax Form",
    "Legal Document",
    "Shipping Label",
    "Purchase Order",
    "Rental Agreement",
    "Payslip",
    "Loan Agreement",
]


sections = [
    "Introduction",
    "Body",
    "Conclusion",
    "Summary",
    "Contact Information",
    "Disclaimers",
    "Terms and Conditions",
    "Important Notes",
    "Instructions",
    "Invoice Items",
    "Payment Details",
    "Signature Section",
    "Legal Disclaimer",
    "Legal Notices",
    "Risk Warnings",
    "Compliance Statements",
    "Background Information",
    "Action Items",
    "Legal Agreements",
    "Regulatory Disclosures",
]
layouts = [
    "Single Column Layout",
    "Tables",
    "Numbered Lists",
    "Indented Paragraphs",
    "Forms",
    "Tables with Headers",
    "Paragraphs with Subheadings",
    "Tabbed Sections",
    "Footer and Header Design",
    "Form-Based Layout",
    "Grid Layout",
    "Side-by-Side Tables",
    "Left-Aligned Lists",
    "Right-Aligned Lists",
    "Multi-Layered Forms",
    "Highlighted Call-Out Sections",
    "Bordered Sections",
]
headers = [
    "Main Title",
    "Subtitle",
    "Document Title",
    "Date",
    "Section Headers (e.g., Introduction, Summary)",
    "Subsection Headers",
    "Header with Contact Information",
    "Footer with Legal Information",
    "Header with Document Type",
    "Recipient Information",
    "Signature Line",
    "Document Title with Subtitle",
    "Recipient Information Header",
    "Subject Line",
    "Confidentiality Header",
    "Page Numbering in Header",
    "Header with Date",
    "Header with Versioning",
    "Section Headers with Numbers",
    "Company Information Header",
    "Recipient Company Information",
    "Signature Section Header",
    "Report Title Header",
    "Table of Contents Header",
    "Disclaimer Header",
    "Action Required Header",
    "Approval Header",
    "Header with Contact Information",
]


pii_entities_colors = {
    "first_name": (1, 0, 0),  # Красный
    "last_name": (1, 0.5, 0.5),  # Розовый
    "middle_name": (1, 0.5, 1),  # Фуксия
    "address": (0, 1, 0),  # Зеленый
    "phone_number": (0, 0, 1),  # Синий
    "email_address": (1, 1, 0),  # Желтый
    "dates": (1, 0, 1),  # Фиолетовый
    "credit_card_number": (0, 1, 1),  # Бирюзовый
    "iban": (1, 0.5, 0),  # Оранжевый
    "company_name": (0, 0.5, 0.5),  # Темно-зеленый
    "signature": (0.5, 0.5, 0.5),  # Серый
    "full_name": (0.5, 0, 0.5),  # Темно-фиолетовый
    "vin": (0.5, 1, 0),  # Ярко-зеленый
    "car_plate": (0, 0.5, 1),  # Светло-синий
}

pii_entities_colors_names = {
    "first_name": "Red",
    "last_name": "Pink",
    "middle_name": "Fuchsia",
    "address": "Green",
    "phone_number": "Blue",
    "email_address": "Yellow",
    "dates": "Purple",
    "credit_card_number": "Turquoise",
    "iban": "Orange",
    "company_name": "Green",
    "signature": "Grey",
    "full_name": "Dark Purple",
    "vin": "Bright Green",
}


font_family = [
    "Arial",
    "Times New Roman",
    "Calibri",
    "Verdana",
    "Courier New",
    "Georgia",
    "Helvetica",
    "Comic Sans MS",
    "Impact",
    "Lucida Console",
    "Palatino Linotype",
    "Trebuchet MS",
    "Book Antiqua",
    "Century Gothic",
    "Galvji",
    "Malayalam Sangam MN",
    "Mali",
    "Symbol",
    "Andale Mono",
    "Monaco",
    "Katari",
    "Gotu",
    "Nadeem",
    "Mukta Vaani",
    "Tahoma",
]

font_family_latex = [
    "Arial",
    "Times New Roman",
    "Verdana",
    "Courier New",
    "Georgia",
    "Helvetica",
    "Comic Sans MS",
    "Impact",
    "Trebuchet MS",
    "Galvji",
    "Malayalam Sangam MN",
    "Andale Mono",
    "Monaco",
    "Tahoma",
]

latex_templates = {
    "column_prices": columns_with_prices_template,
    "email": email_template,
    "form_many_text": form_with_many_text_template,
    "form_text": form_with_text_template,
    "letter_plain_text": letter_with_plain_text_template,
    "multi_tables": multi_tables_columns_template,
    "multi_format": multi_format_template,
    "letter_table": letter_with_table_template,
    "formal_email": formal_email_template,
    "invoice": invoice_template,
    "multi_tables_simple": multi_tables_columns_template_simple,
    "columns_prices_simple": columns_with_prices_template_simple,
    "email_simple": email_template_simple,
    "form_many_text_simple": form_with_many_text_template_simple,
    "form_text_simple": form_with_text_template_simple,
    "letter_plain_text_simple": letter_with_plain_text_template_simple,
    "multi_format_simple": multi_format_template_simple,
    "letter_table_simple": letter_with_table_template_simple,
    "formal_email_simple": formal_email_template_simple,
    "invoice_simple": invoice_template_simple,
}

subjects = [
    "Apartment",
    "Vehicle",
    "Medical issues",
    "Insurance",
    "Travel",
    "Any property ownership",
    "Service rendered",
    "Legal issues",
]
