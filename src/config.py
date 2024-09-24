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
    {"class": "Address", "pii_id": "address", "prompt": GENERATE_ADDRESS_SYSTEM_PROMPT},
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
    {"class": "IBAN", "pii_id": "iban", "prompt": GENERATE_IBAN_SYSTEM_PROMPT},
    {
        "class": "Company Name",
        "pii_id": "company_name",
        "prompt": GENERATE_COMPANY_NAMES_SYSTEM_PROMPT,
    },
    # car plate number
    # VIN
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
