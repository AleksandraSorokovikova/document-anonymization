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
    {
        "class": "Middle Name",
        "pii_id": "middle_name",
        "prompt": GENERATE_MIDDLE_NAMES_SYSTEM_PROMPT,
    },
]

pii_to_id = {pii["pii_id"]: i for i, pii in enumerate(pii_classes) if pii["pii_id"] != "middle_name"}
id_to_pii = {i: pii["pii_id"] for i, pii in enumerate(pii_classes) if pii["pii_id"] != "middle_name"}
layoutlm_ner_classes = ['B-payment_information', 'I-payment_information', 'B-full_name', 'I-full_name', 'B-company_name', 'I-company_name', 'O', 'B-address', 'I-address', 'B-phone_number', 'I-phone_number', 'B-car_plate', 'I-car_plate', 'B-email_address', 'B-vin']


pii_names = [
    "First Name",
    "Last Name",
    "Address",
    "Phone Number",
    "Email Address",
    "Dates",
    "Credit Card Number",
    "IBAN",
    "Company Name",
    "Signature",
    "Full Name",
    "VIN",
    "Car Plate"
]


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
    "first_name": (1, 0, 0),  # Red
    "last_name": (1, 0.5, 0.5),  # Pink
    "address": (0, 1, 0),  # Green
    "phone_number": (0, 0, 1),  # Blue
    "email_address": (1, 1, 0),  # Yellow
    "dates": (0.5, 0, 0.5),  # Purple
    "credit_card_number": (0.25, 0.88, 0.82),  # Turquoise
    "iban": (1, 0.5, 0),  # Orange
    "company_name": (0.5, 0, 0),  # Maroon
    "signature": (0.5, 0.5, 0.5),  # Grey
    "full_name": (1, 0, 1),  # Fuchsia
    "vin": (0, 1, 0),  # Lime
    "car_plate": (0, 1, 1),  # Cyan
}

pii_entities_colors_names = {
    "first_name": "Red",
    "last_name": "Pink",
    "address": "Green",
    "phone_number": "Blue",
    "email_address": "Yellow",
    "dates": "Purple",
    "credit_card_number": "Turquoise",
    "iban": "Orange",
    "company_name": "Maroon",
    "signature": "Teal",
    "full_name": "Fuchsia",
    "vin": "Lime",
    "car_plate": "Cyan",
    "invoice_id": "Aqua",
    "payment_information": "Orange"
}

pii_entities_colors_rgba = {
    'first_name': (255, 0, 0, 60),           # Red
    'last_name': (255, 192, 203, 60),        # Pink
    'address': (0, 128, 0, 60),              # Green
    'phone_number': (0, 0, 255, 60),         # Blue
    'email_address': (255, 255, 0, 60),      # Yellow
    'dates': (128, 0, 128, 60),              # Purple
    'credit_card_number': (64, 224, 208, 60),# Turquoise
    'iban': (255, 165, 0, 60),               # Orange
    'company_name': (128, 0, 0, 60),         # Maroon
    'signature': (0, 128, 128, 60),          # Teal
    'full_name': (255, 0, 255, 60),          # Fuchsia
    'vin': (0, 255, 0, 60),                  # Lime
    'car_plate': (0, 255, 255, 60),          # Cyan
    'invoice_id': (0, 255, 255, 60),         # Aqua (same as Cyan)
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
    "columns_with_prices_template_plain": columns_with_prices_template_plain,
    "email_template_plain": email_template_plain,
    "form_with_many_text_template_plain": form_with_many_text_template_plain,
    "form_with_text_template_plain": form_with_text_template_plain,
    "letter_with_plain_text_template_plain": letter_with_plain_text_template_plain,
    "multi_tables_columns_template_plain": multi_tables_columns_template_plain,
    "multi_format_template_plain": multi_format_template_plain,
    "letter_with_table_template_plain": letter_with_table_template_plain,
    "formal_email_template_plain": formal_email_template_plain,
    "invoice_template_plain": invoice_template_plain,
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


funsd_label_list = ["full_name", "phone_number", "address", "company_name", "email_address", "signature"]
receipt_label_list = ["full_name", "phone_number", "address", "company_name", "signature"]
invoices_label_list = ["full_name", "phone_number", "address", "company_name", "email_address", "invoice_id"]
