GENERATE_PHONE_NUMBER_SYSTEM_PROMPT = """
Your task is to generate a diverse list of realistic phone numbers. Ensure that the phone numbers come from various countries and regions, reflecting different formats and styles.

Guidelines for phone number generation:
1. Include phone numbers from different regions, with appropriate country codes and formats.
2. Ensure diversity in the structure of phone numbers (e.g., variations in length and formatting conventions).
3. Avoid generating repetitive patterns or numbers that look overly fictional.

The phone numbers should appear authentic, representing a wide range of countries and regions.
Possible formats (but don't limit yourself to these):
    - +1 (111) 111-1111
    - (111) 1111 1111
    - 111-1111-1111
    - +11 11 111111
    - 11-11111/11
    - 111-1111111
    - +1111111111

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF PHONE NUMBERS.
"""

GENERATE_ADDRESS_SYSTEM_PROMPT = """
Your task is to generate a diverse list of realistic addresses. Ensure the addresses come from a variety of countries and regions, reflecting different formats and structures.

Guidelines for address generation:
1. Include addresses from various countries, ensuring diversity in street names, postal codes, and regions.
2. Ensure the addresses follow realistic formats for the countries they belong to (e.g., house number, street, city, postal code).
3. Avoid repetitive patterns, and ensure the addresses appear authentic.

The addresses should look realistic and reflect the conventions of different countries and regions.

Possible formats (but don't limit yourself to these):
    - <Postal Code> <City>
    - <Street> <House Number>
    - <Street> <Postal Code> <City>
    - <House Number> <Street> St, <City>, <Postal Code>, <Country>
    - <House Number> <Street>, <City>, <Country>

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF ADDRESSES.
"""

GENERATE_FIRST_NAMES_SYSTEM_PROMPT = """
Your task is to generate a diverse list of realistic first names. Ensure that the names come from a wide range of cultures and regions, reflecting various naming conventions and styles.

Guidelines for first name generation:
1. Include first names from different regions, cultures, and languages.
2. Ensure that the names are diverse and not overly repetitive.
3. Avoid generating names that seem overly fictional or unrealistic.

The generated names should resemble common real-world first names from various parts of the world.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF FIRST NAMES.
"""

GENERATE_LAST_NAMES_SYSTEM_PROMPT = """

Your task is to generate a diverse list of realistic second (last) names. Ensure that the names come from a wide range of regions, reflecting different cultures and languages.

Guidelines for second name generation:
1. Include last names from various countries, regions, and cultures.
2. Ensure diversity in naming conventions, avoiding overly repetitive or fictional names.
3. The last names should appear realistic and reflect different linguistic patterns from across the world.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF SECOND NAMES.
"""


GENERATE_MIDDLE_NAMES_SYSTEM_PROMPT = """
Your task is to generate a diverse list of realistic middle names. Middle names should reflect cultural diversity and vary in length and style.

Guidelines for middle name generation:
1. Include middle names from a variety of regions and languages.
2. Ensure diversity, avoiding overly repetitive or fictional middle names.
3. Middle names should reflect realistic naming conventions.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF MIDDLE NAMES.
"""


GENERATE_IBAN_SYSTEM_PROMPT = """
Your task is to generate a diverse and realistic list of IBAN numbers. Ensure that the IBANs follow the correct country-specific formats. Each IBAN should begin with a valid country code and adhere to the standard length and structure for that country.

Guidelines for IBAN generation:
1. Use valid two-letter country codes at the start of each IBAN.
2. Vary the IBANs by including multiple countries, ensuring a wide range of regional formats.
3. Ensure the length and format for each IBAN correspond to the country's specific standard.
4. Avoid generating repetitive patterns, and ensure that each IBAN appears unique and realistic.

The output should look like authentic IBANs from different countries.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF IBAN NUMBERS.
"""

GENERATE_EMAIL_ADDRESS_SYSTEM_PROMPT = """
Your task is to generate a diverse and realistic list of email addresses. Ensure the email addresses follow valid formats and vary in style, domain, and top-level domain (TLD). The email addresses should appear to come from various regions and providers.

Guidelines for email generation:
1. Use a variety of domain names, including personal, corporate, and regional domains.
2. Include different TLDs (e.g., .com, .net, .org) as well as regional ones (e.g., .de, .uk, .fr).
3. Ensure the email addresses are unique and realistic, avoiding repetitive patterns.
4. Incorporate a mix of professional, personal, and informal email formats.

Ensure that the email addresses look authentic and do not follow a predictable pattern.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF EMAIL ADDRESSES.
"""

GENERATE_COMPANY_NAMES_SYSTEM_PROMPT = """
Your task is to generate a diverse list of realistic company names. Ensure that the company names vary in industry, region, and naming style to reflect a wide range of businesses.

Guidelines for company name generation:
1. Generate company names that sound realistic and can belong to various industries (e.g., technology, finance, retail).
2. Include a variety of naming conventions, such as professional, creative, and corporate styles.
3. Ensure that the company names are diverse and not overly repetitive.

The company names should appear authentic and realistic, representing businesses from different sectors and regions.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF COMPANY NAMES.
"""

GENERATE_DATES_SYSTEM_PROMPT = """
Your task is to generate a diverse list of realistic dates. Ensure that the dates are formatted correctly and represent a wide range of time periods.

Guidelines for date generation:
1. Use a variety of date formats (e.g., day-month-year, month-day-year).
2. Include dates from different decades and time periods, both recent and historical.
3. Ensure the dates are realistic, avoiding overly repetitive or future/past unrealistic dates.

The dates should be well-formatted and realistic, reflecting different timeframes and styles.

Possible formats (but don't limit yourself to these):
    - MM/DD/YYYY
    - DD-MM-YYYY
    - YYYY/MM/DD
    - DD.MM.YYYY
    - YYYY.MM.DD
    - Month DD, YYYY
    - DD Month YYYY
    - Month DD, YYYY
    - DD Month, YYYY
    - Month DD YYYY
    - DD Month YYYY
YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF DATES OF BIRTH.
"""

GENERATE_CREDIT_CARD_NUMBER_SYSTEM_PROMPT = """
Your task is to generate a diverse and realistic list of credit card numbers. Ensure that the numbers follow valid credit card formats, including correct lengths and number groupings.

Guidelines for credit card number generation:
1. Ensure that the credit card numbers adhere to realistic formats and lengths (e.g., 16-digit numbers).
2. Include variety by using different card number formats and issuer ranges.
3. Avoid repetitive patterns, and ensure the numbers appear authentic.

The credit card numbers should look realistic, with variations in issuer type and number groupings.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF CREDIT CARD NUMBERS.
"""

GENERATE_VIN_SYSTEM_PROMPT = """
Your task is to generate a diverse and realistic list of Vehicle Identification Numbers (VINs). Ensure that the VINs follow valid formats and represent a variety of vehicle types and manufacturers.

Guidelines for VIN generation:
1. Use valid VIN formats, including the correct number of characters and structure.
2. Include a variety of VINs representing different vehicle types (e.g., cars, trucks, motorcycles).
3. Ensure that the VINs are unique and do not follow a predictable pattern.

The VINs should look authentic and reflect the diversity of vehicle identification numbers.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF VEHICLE IDENTIFICATION NUMBERS.
"""

GENERATE_CAR_PLATE_NUMBER_SYSTEM_PROMPT = """
Your task is to generate a diverse and realistic list of car plate numbers. Ensure that the plate numbers follow valid formats and represent a variety of regions and vehicle types.

Guidelines for car plate number generation:
1. Use valid car plate formats, including the correct number of characters and structure.
2. Include a variety of plate numbers representing different regions and countries.
3. Ensure that the plate numbers are unique and do not follow a predictable pattern.

The car plate numbers should look authentic and reflect the diversity of vehicle registration numbers.

YOUR OUTPUT SHOULD BE A JSON LIST (NOT DICT) OF CAR PLATE NUMBERS.
"""

USER_PROMPT = """
Number of entities to generate: {0}
YOU MUST RETURN LIST, NOT DICT.
""".format

GENERATE_HTML_CONTENT_SYSTEM_PROMPT = """
**Task Overview:**
You will be provided with a list of Personal Identifiable Information (PII) entities, a document type, and other relevant details. Your task is to generate an **HTML document** that, when rendered and converted to PDF, represents the document type with the provided PII entities embedded in it.

**Instructions:**

1. **Provided Information:**
    - Document Type
    - List of PII Entities
    - Font Family
    - Document Structure (including sections, headers, and layout)

2. **HTML Document Requirements:**
    - The HTML document must represent the provided document type with the PII entities embedded in their corresponding fields.
    - Ensure that the HTML content fits within a single page. Use the following CSS to manage page size:
    ```css
    @page 
        size: A4;
        margin: 0;
    
    ```
    - Set a font size of 15px by adding this to the CSS in the body:
    ```css
    body 
        font-size: 15px;
    
    ```
    - **Do not** highlight PII entities with different font colors or background effects; they should appear as regular plain text.

3. **Content Embedding:**
    - You must choose which PII entities to embed in the document based on the provided document structure. You don't need to embed all PII entities in the document if they are not required within the layout.
    - YOU MUST NOT CHANGE EXISTING PII ENTITIES, namely, add extra spaces and other symbols to the provided PII data, change date formats, change phone number/credit card formats, or modify the provided PII data in any way.
    - YOU MUST NOT DIVIDE PII DATA INTO MULTIPLE PARTS AND EMBED THEM SEPARATELY, namely splitting a phone number into area code and number, or separating a date into day, month, and year, or split addresses and put town os street separately.
    - Ensure that all chosen PII entities are embedded within the HTML content and placed in appropriate sections (e.g., full name in the full name field, address in the address field).
    - Vary the sentence structure for embedding PII data.
    - Follow the structure provided in the **Document Structure** section to organize the content.
    - YOU MUST NOT USE ANY ADDITIONAL PII DATA BEYOND THE PROVIDED LIST.

4. **Document Structure and Layout:**
    The structure of the document will be specified by the following randomly selected components:
    - **Layout:** The document layout may include tables, bullet points, single or multi-column formats, or forms. Ensure the content follows the layout structure specified.
    - **Sections:** The provided sections in Sections are not exhaustive; you may include additional sections if needed and may exclude some if they are not relevant.
    - **Headers:** Use the provided header elements such as document titles, dates, or section headers. You are also not limited to these headers; you may include additional headers if needed or exclude some if they are not relevant.
    You MUST NOT put the names of the sections or headers in the document; only the content.
    
5. Signatures insertion:
    - If the generated document requires a signature, you must place in the HTML file the path to signature image in the corresponding field (in tag `img` with width="200" and height="100").
    - Do not put more than one signature in the document.
    - The path should look exactly like this: `src="path/to/signature.png"`.

6. **Content Focus:**
    - The main text of the document should align with the document type provided.
    - Add filler text related to the document type to make it realistic (at least 300 words), but **do not** introduce any new PII data not provided.
    - The generated text should be meaningful and consist of real sentences (avoid gibberish).

7. **Strictly Prohibited Entities:**
    - You **MUST NOT** generate or include any additional PII data beyond the provided list.
    - This includes but is not limited to:
      - {0}
    - Only PII entities from the provided list should appear in the document. No placeholders or random extra PII should be generated.

8. **Diversity in Presentation:**
    - **Fonts & Styles:** You must use the provided font family but may vary sizes, boldness, and italics where appropriate.
    - **Layouts:** The layout should vary according to the specified structure, such as alternating between tables, paragraphs, bullet points, or multi-column layouts.
    - **Language Variations:** Keep the content in English but use international variations where appropriate (e.g., “Postcode” vs. “Zip Code” or currency symbols like $, €, £).

**Output format:**
    - A valid HTML file.
""".format

GENERATE_HTML_CONTENT_USER_PROMPT = """
**Document Type:**
{0}
**PII Entities:**
{1}
**Font Family:**
{2}

**Document Structure:**
- Layout: {3}
- Sections: {4}
- Headers: {5}
""".format


GENERATE_LATEX_CONTENT_SYSTEM_PROMPT = """
**Task Overview:**
You will be provided with a list of Personal Identifiable Information (PII) entities, a document layout description, and document subject. Your task is to generate a **LaTeX document** that, when compiled, represents the document type with the provided PII entities embedded in it.

**Instructions:**

1. **Provided Information:**
    - List of PII Entities
    - Document Subject
    - Document Layout Description
    - Font-Family
    
2. **LaTeX Document Requirements:**
    - The LaTeX document must represent the provided document type with the PII entities embedded in their corresponding fields.
    - Ensure that the LaTeX content fits within a single page. Use the following LaTeX template to manage page size:
    geometry: a4paper, margin=0.6in
    - Set a font size of 10pt by adding this to the preamble:
    ```latex
    \\documentclass[10pt]...
    ```
    - You should very the font size and style in different sections of the document.
    - You also can add bold/italic/underline styles where appropriate (even in PII entities).
    - You can write some text With Uppercase Letters.
    - Remember to add usepackage: graphicx.
    - **Do not** highlight PII entities with different font colors or background effects; but sometimes you can make them bold or italic.
    - Use the provided font family for the document content. Include the font package in the preamble:
        - usepackage: fontspec
        - setmainfont: Font-Family
    - **Prevent Intersection Issues:**
      - To prevent horizontal lines from intersecting, include the following in the preamble:
        - use package microtype for automatic text adjustment.
        - Add extra horizontal spacing where lines may intersect by increasing the space in tabular environments or by using `\hspace` when needed. If the line is very long, add hspace = 5cm.

3. **Content Embedding:**
    - You must choose which PII entities to embed in the document based on the provided layout description. You don't need to embed all PII entities in the document, if they are not required within the layout.
    - YOU MUST NOT CHANGE EXISTING PII ENTITIES, namely, add extra spaces and other symbols to the provided PII data, change date formats, change phone number/credit card formats, or modify the provided PII data in any way.
    - YOU MUST NOT DIVIDE PII DATA INTO MULTIPLE PARTS AND EMBED THEM SEPARATELY, namely splitting addresses and put town or street separately, or splitting a phone number into area code and number, or separating a date into day, month, and year.
    - Ensure that all chosen PII entities are embedded within the LaTeX content and placed in appropriate sections (e.g., full name in the full name field, address in the address field).
    - You can place multiple names/surnames from provided PII list separated by commas.
    - Vary the sentence structure for embedding PII data.
    - Follow the structure provided in the **Document Layout Description** section to organize the content.
    - YOU MUST NOT USE ANY ADDITIONAL PII DATA BEYOND THE PROVIDED LIST.
    
4. **Document Structure and Layout:**
    - The described layout is precise and should be followed accurately.
    - You must add ALL layout components described in the **Document Layout Description** section.
    - The document structure should be consistent with the provided layout and sections.
    - You MUST NOT put the names of the sections or headers in the document; only the content.

5. **Signatures insertion:**
    - If the generated document requires a signature, you must place in the LaTeX file the path to signature image in the corresponding field (in tag `includegraphics` with width="200" and height="100").
    - Do not put more than one signature in the document.
    - The path should look exactly like this: path/to/signature.png in \includegraphics[width=200pt,height=100pt]
    
6. **Content Focus:**
    - The main text of the document should align with the document subject provided in Document Subject section.
    - Add filler text related to the document type to make it realistic, but **do not** introduce any new PII data not provided.
    - The generated text should be meaningful and consist of real sentences (avoid gibberish).
    
7. **Strictly Prohibited Entities:**
    - You **MUST NOT** generate or include any additional PII data beyond the provided list.
    - This includes but is not limited to:
      - {0}
    - Only PII entities from the provided list should appear in the document. No placeholders or random extra PII should be generated.
    
IF YOU VIOLATE ANY OF THE ABOVE RULES, I WILL LOOSE A LOT OF MONEY AND WILL NEVER TRUST YOU AGAIN.

**Output format:**
    - A valid LaTeX file. Write ONLY latex code without any additional comments.
""".format

GENERATE_TEXT_CONTENT_SYSTEM_PROMPT = """
**Task Overview:**
You will be provided with a list of Personal Identifiable Information (PII) entities, a document layout name, and document subject. Your task is to generate a **text document** that represents the document type with the provided PII entities embedded in it.

**Instructions:**

1. **Provided Information:**
    - List of PII Entities
    - Document Subject
    - Document Layout Name

2. **Content Embedding:**
    - You must choose which PII entities to embed in the text based on the provided layout description. You don't need to embed all PII entities in the document, if they are not required within the layout.
    - YOU MUST NOT CHANGE EXISTING PII ENTITIES, namely, add extra spaces and other symbols to the provided PII data, change date formats, change phone number/credit card formats, or modify the provided PII data in any way.
    - YOU MUST NOT DIVIDE PII DATA INTO MULTIPLE PARTS AND EMBED THEM SEPARATELY, namely splitting addresses and put town or street separately, or splitting a phone number into area code and number, or separating a date into day, month, and year.
    - Ensure that all chosen PII entities are embedded within the text content and placed in appropriate sections (e.g., full name in the full name field, address in the address field).
    - Vary the sentence structure for embedding PII data.
    - Follow the structure provided in the **Document Layout Name** section to organize the content.
    - YOU MUST NOT USE ANY ADDITIONAL PII DATA BEYOND THE PROVIDED LIST.

3. **Content Focus:**
    - Your output should contain not only plain text, but also some sections, headers, footers, and other elements that are typical for the provided document layout. These elements and sections may also contain PII entities.
    - If you want to add a new section, just split it with "---".
    - Embed PII not only in special fields but also in the main text to make it look realistic.
    - The main text of the document should align with the document subject provided in Document Subject section.
    - Add filler text related to the document type to make it realistic, but **do not** introduce any new PII data not provided.
    - The generated text should be meaningful and consist of real sentences (avoid gibberish).

4. **Strictly Prohibited Entities:**
    - You **MUST NOT** generate or include any additional PII data beyond the provided list.
    - This includes but is not limited to:
        - First Name
        - Last Name
        - Address
        - Phone Number
        - Email Address
        - Dates
        - Credit Card Number
        - IBAN
        - Company Name
        - Signature
        - Full Name
        - VIN
        - Car Plate
    - Only PII entities from the provided list should appear in the document. No placeholders or random extra PII should be generated.

IF YOU VIOLATE ANY OF THE ABOVE RULES, I WILL LOOSE A LOT OF MONEY AND WILL NEVER TRUST YOU AGAIN.

**Output format:**
    - A plain text file with specific sections and headers.
""".format


GENERATE_LATEX_FROM_PICTURE_AND_PII_SYSTEM_PROMPT = """
You are provided with:
  - PII entities
  - A photo of a document
  - Document subject
  - Font Family

Your task is to write valid LaTeX code with a layout based by the document in the photo, removing the text content from the document and embedding the given PII entities into it.
Try to closely and precisely reproduce the layout of the document in the photo, following all the rules below.
To embed PII entities you must generate suitable text, which contains these entities (provided document subject is the topic you should rely on).
The provided document may contain its own dates, addresses, phone numbers, and other PII entities. YOU MUST REMOVE THEM IF THEY ARE PRESENT IN THE PHOTO AND REPLACE THEM WITH THE PROVIDED PII ENTITIES.
The provided document on the photo can be rotated, but you should not rotate it in your output.
**IGNORE ALL TEXT IN THE DOCUMENT PHOTO ENTIRELY. Do not attempt to extract, read, or interpret any text from it. The only textual content allowed in the output is the one you generate using the provided PII entities. If you include any text from the photo, you will fail the task.**  
Do not add rotated (vertical) elements.
Be creative in how to combine document layout and provided PII entities.

**Important rules regarding PII entities:**
    - **ALL text-based PII entities in the document MUST be removed and replaced with the provided ones.**  
    - You must choose which PII entities to embed in the text based on the provided layout. 
    - **ANY original text that looks like a name, date, address, or phone number must be removed completely and replaced.**  
    - **Failure to do this will be considered a mistake.**  
    - **DO NOT CREATE OR PRESERVE ANY PII BEYOND THE PROVIDED LIST.**  
    - **If there is any text in the document that resembles PII but is not explicitly in the provided list, it must be REMOVED and NOT replaced.**  
    - **YOU MUST NOT CHANGE EXISTING PII ENTITIES, namely, add extra spaces and other symbols to the provided PII data, change date formats, change phone number/credit card formats, or modify the provided PII data in any way even if the document format requires it.** IF YOU VIOLATE THIS RULE, I WILL LOOSE A LOT OF MONEY AND WILL NEVER TRUST YOU AGAIN. If the date in the PII entities is written like "1910-03-07", it should remain absolutely the same in your output. It applies to all PII entities.
    - YOU MUST NOT DIVIDE PII DATA INTO MULTIPLE PARTS AND EMBED THEM SEPARATELY, namely splitting addresses and put town or street separately, or splitting a phone number into area code and number, or separating a date into day, month, and year.
    - Ensure that all chosen PII entities are embedded within the text content and placed in appropriate sections (e.g., full name in the full name field, address in the address field).
    - YOU MUST NOT USE ANY ADDITIONAL PII DATA BEYOND THE PROVIDED LIST.
    - You **MUST NOT** generate or include any additional PII data beyond the provided list.
    - This includes but is not limited to:
        - First Name
        - Last Name
        - Address
        - Phone Number
        - Email Address
        - Dates
        - Credit Card Number
        - IBAN
        - Company Name
        - Signature
        - Full Name
        - VIN
        - Car Plate

**LaTeX Document Requirements:**
    - Ensure that the LaTeX content fits within a single page. Use the following LaTeX template to manage page size:
    geometry: a4paper, margin=0.6in
    - Set a font size of 10pt by adding this to the preamble:
    ```latex
    \\documentclass[10pt]...
    ```
    - You should very the font size and style in different sections of the document.
    - Remember to add usepackage: graphicx.
    - Use the provided font family for the document content. Include the font package in the preamble:
        - usepackage: fontspec
        - setmainfont: Font-Family
    - **Prevent Intersection Issues:**
      - To prevent horizontal lines from intersecting, include the following in the preamble:
        - use package microtype for automatic text adjustment.
        - Add extra horizontal spacing where lines may intersect by increasing the space in tabular environments or by using `\hspace` when needed. If the line is very long, add hspace = 5cm.

**Signatures insertion:**
    - If the generated document requires a signature, you must place in the LaTeX file the path to signature image in the corresponding field (in tag `includegraphics` with width="200" and height="100").
    - Do not put more than one signature in the document.
    - The path should look exactly like this: path/to/signature.png in \includegraphics[width=200pt,height=100pt]

**Output:**
A valid latex code.
"""

GENERATE_LATEX_FROM_PICTURE_AND_PII_USER_PROMPT = """
**PII Entities:**
{0}
--------------------------------------------------
**Document Subject:**
{1}
--------------------------------------------------
**Font Family:**
{2}
""".format


GENERATE_PURE_LATEX_FROM_PICTURE_SYSTEM_PROMPT = """
Write latex code to precisely reproduce the document layout as in the picture.
You must delete all the text content from the document and leave ONLY elements which represent the layout itself (even titles and column names).
Do not add rotate elements.
Make sure that the layout fits in one page (add \documentclass[10pt], geometry: a4paper, margin=0.6in).
"""


GENERATE_LATEX_CONTENT_USER_PROMPT = """
**PII Entities:**
{0}
--------------------------------------------------
**Document Subject:**
{1}
--------------------------------------------------
**Document Layout Description:**
{2}
--------------------------------------------------
**Font Family:**
{3}
""".format


GENERATE_TEXT_CONTENT_USER_PROMPT = """
**PII Entities:**
{0}
--------------------------------------------------
**Document Subject:**
{1}
--------------------------------------------------
**Document Layout Name:**
{2}
""".format


columns_with_prices_template = """
The layout of the document consists of a structured format with various sections that use columns to present information. Here is the detailed breakdown of the layout:

### Document Header:

1. **Top-left corner:**
    - Order number.
    - Client’s information and personal details
    - This line is located at the top of the page, aligned with the left margin.
2. **Top-right corner:**
    - Date
    - Company’s details and contact information
    - This line is aligned to the right margin, directly opposite the order number.
3. **Centered Title:**
    - The title is located at the top center, aligned horizontally between the order number and date.
    - This section uses uppercase letters with a monospaced font to create even spacing between letters.

### Sub-header (Below the Title):

1. **First Row:**
    - Names of further subsequent
    - These are presented in a single line with clear gaps between each section, extending across the width of the page.
2. **Second Row:**
    - Some additional information about a client.

### Body of the Document:

- The body follows a table structure with **six main columns**:
    1. **Column 1:**
        - Contains numbered entries, all aligned to the left.
    2. **Column 2:**
        - Contains short descriptions that follow the numbered entries. This column has variable-length text, justified to the left.
    3. **Column 3:**
        - Contains single letters (likely classifying the entries). The content is centered in this column.
    4. **Column 4:**
        - A fixed numeric value repeated down the column, aligned centrally.
    5. **Column 5:**
        - Contains hours worked. This text is aligned centrally.
    6. **Column 6:**
        - Contains prices in EUR. The text is right-aligned in this column, with values spaced carefully to the right margin.
- **Row Layout:**
    - Each row represents an individual entry. Rows are evenly spaced and have consistent gaps between them, aligning all columns clearly.
    - Rows of text are consistently justified across the table, with clear separation between columns.

### Footer (Bottom of Document):

- On the bottom left of the page, there is text, aligned to the left margin.
- On the bottom right, there is a page number, aligned to the right margin.

### General Observations:

- The document uses a monospaced font for most of the table to ensure even spacing and alignment between columns.
- The document contains approximately **200-300 words** and **about 30 sentences** (counting individual rows as part of sentences).

### Document Type Suggestion:

- **Repair Estimate Form**
"""


columns_with_prices_template_simple = """
The document layout follows a structured format with multiple sections presented in columns.

### Header Section:
- Information such as the order number, client's details, and date is displayed at the top, split between the left and right corners.
- A centered title is placed at the top of the page, using uppercase letters for uniform spacing.

### Sub-header:
- Additional details are organized in two rows below the title, with spaced-out sections covering the page width.

### Main Body:
- The body consists of a table with six columns:
    1. Numbered entries.
    2. Short descriptions.
    3. Single-letter classifications.
    4. Repeated numeric values.
    5. Hours worked.
    6. Prices in EUR.
- The table has evenly spaced rows, and text alignment varies across the columns for clarity.

### Footer:
- Text is aligned to the bottom left, with a page number on the bottom right.

### General Observations:
- The document maintains a clean, column-based structure with consistent spacing for easy reading.
"""

columns_with_prices_template_plain = """
The document layout consists of a structured format with multiple sections presented in columns.
The document contains approximately 200-300 words and about 30 sentences.
"""

email_template = """
The layout of the provided document follows the structure of a formal email. Here’s a detailed description of the layout:

### Document Header:

1. **Top Section:**
    - The document begins with a line at the top, showing an email account name enclosed in quotes.
    - This line is aligned to the left margin, and it contains a noticeable amount of hidden or redacted information, as indicated by blacked-out or blue-outlined boxes.
2. **Sender/Receiver Information Block:**
    - Below the account name, there is a structured block that lists key email metadata.
    - This section contains three fields:
        - **From:** and **Sent:** with respective sender details and date/time information.
        - **To:** with recipient information.
        - **Subject:** with a subject line, containing a number and date.
    - Each of these fields is arranged in two columns:
        - The left column contains the labels ("From:", "Sent:", etc.) aligned to the left.
        - The right column contains the corresponding information, with each entry spaced and aligned with its label.

### Horizontal Divider:

- A black horizontal line separates the header section from the main body of the email.

### Main Body of the Email:

- **Salutation:**
    - "Dear <Surname>," appears at the top of the body text, left-aligned.
- **Main Text:**
    - The body of the email contains a short paragraph formatted as follows:
        - **First Paragraph:** Provides a description of a subject.
        - **Second Paragraph:** Additional details about the subject.
    - The text is simple and left-aligned without indentation, covering around 7-8 short lines.
- **Closing:**
    - The closing is a standard email closing:
        - A double dash (--) followed by a polite phrase "Kind regards,".
        - This line is left-aligned and followed by a blank line before the signature block.

### Signature Block:

- **Contact Information:**
    - The signature block contains information and includes fields for:
        - A name, possibly a title, and company information.
        - **"Tel.:" (Telephone)** and **"Mail:" (Email)** fields, each aligned in a two-column format similar to the earlier header fields.

### General Observations:

- The document uses a simple, non-stylized font for both the body and header.
- Text alignment is consistently left-aligned across the entire document.
- The document has approximately 120-140 **words** and 7-8 **sentences** in total.

### Document Type Suggestion:

- **Email**
"""

email_template_simple = """
The document layout resembles a formal email structure, with key sections presented in a straightforward manner.

### Header Section:
- The top of the document includes the sender's email, with parts redacted or highlighted.
- Below, there is a block containing fields for the sender, recipient, date/time, and subject, organized in two aligned columns.

### Horizontal Divider:
- A line separates the header from the body of the email.

### Main Body:
- Starts with a salutation followed by a short message in two paragraphs, left-aligned with no indentation.
- Closes with "Kind regards" and a signature block containing contact details in a two-column format.

### General Observations:
- The document maintains consistent left alignment throughout and uses a simple font. It has around 120-140 words and 7-8 sentences.
"""

email_template_plain = """
The document layout resembles a formal email structure, with key sections presented in a straightforward manner.
The document contains approximately 120-140 words and 7-8 sentences.
"""


form_with_many_text_template = """
The layout of the provided document follows the structure of a **form**. Below is the detailed description of the layout:

### Document Header:

1. **Top-left corner:**
    - There is a field labeled "Name of client".
    - The rest of this section contains redacted information beneath the name field, presumably contact details.

### Main Title:

- A large, bolded title spans across the page near the top center: "Cost assumption confirmation". This title stands out as the central heading and is followed by an explanation of the form's purpose in smaller text directly beneath it.

### **Form Structure for Information Input**

- This section is designed to gather details from the client regarding.
- It has multiple labeled fields, each with the respective input areas. Each input field is a rectangle aligned either to the left or right, and they are arranged in a two-column format.
1. **Name and Address:**
    - Fields for the clients's name, address, and contact details are placed in the left column, labeled with "Name and address of the client".
    - The right side has similar fields for client’s bank information, labeled.
2. **Additional Information Fields:**
    - This section contains several input boxes for phone numbers, license plate numbers, insurance policy details, and the bank account details. Each input box is aligned with the corresponding label.

### **Payment Instructions**

- The following section, labeled "Payment instructions", is a large paragraph of instructions related to payment.
- **Subsection B.2:** Additional details related to liability are included here. It contains more checkboxes to confirm payment instructions.

### **Company Confirmation**

- This section includes a title: "Confirmation from the company".
- There are input fields and checkboxes here that allow the company to specify how much they will cover for the repair. There are also sections for signatures and form completion dates.

### **Signatures and Final Details:**

- At the bottom of the document, there are input areas for multiple signatures (Signature of the injured party", "Signature of the repair shop").
- There are also multiple date fields in the bottom section of the form.

### General Observations:

- The document uses a combination of text blocks, labeled input fields, and checkboxes.
- The structure of the form is organized into sections and subsections, with clearly defined areas for filling in information.
- The form has approximately **300-400 words** across the text blocks and fields, and about **15-20 sentences** for explanation and instructions.

### Document Type Suggestion:

- **Repair Cost Confirmation Form**
"""

form_with_many_text_template_simple = """
The document follows a structured layout typical of a form, organized into clear sections for information input and instructions.

### Header Section:
- The top-left contains a field for "Name of client," followed by redacted contact details.

### Main Title:
- A bold, central title reads "Cost assumption confirmation," followed by a brief explanation of the form's purpose.

### Form Structure:
- Multiple labeled fields are arranged in two columns:
    1. The left column includes fields for client name, address, and contact details.
    2. The right column has fields for bank and other financial details.
- Additional fields for phone numbers, license plate, and insurance information are also provided.

### Payment Instructions:
- A section with instructions and checkboxes related to payment and liability confirmation.

### Company Confirmation:
- A section for the company's confirmation of coverage, including input fields, checkboxes, and signature spaces.

### Signatures and Dates:
- Signature fields for the client and repair shop, along with date fields at the bottom of the form.

### General Observations:
- The form is structured with labeled input fields, text blocks, and checkboxes, organized into distinct sections for clarity. It contains around 300-400 words and 15-20 sentences of explanation and instructions.
"""

form_with_many_text_template_plain = """
The document layout follows a structured form format with multiple sections for information input and instructions.
The document contains approximately 300-400 words and 15-20 sentences.
"""

form_with_text_template = """
The layout of this document is structured as a **legal declaration form** related to the transfer of claims after an accident. Here is the detailed description of its layout:

### Document Header:

1. **Title:**
    - The document starts with a bold, large title at the top.
    - A smaller subtitle follows immediately below. Both the title and subtitle are centered at the top of the page.
2. **Between Parties:**
    - There is a section introducing the two parties involved in the declaration:
        - The first entity has personal data of the first party.
        - The second entity has company’s name, address, and contact details.

### **Bank Information Section:**

- Directly below the workshop details, there is a section labeled "Bank connection" containing fields for:
    - **IBAN**, **BIC**, "Account Number", and "Bank Code".

### **Information Section:**

1. "Date/location" field.
2. **Vehicle/Company/Apartment Information:**
    - First line for information details.
    - Second line for information details.
    - Third line this information details.

### **Opponent Information Section:**

- Fields for the opposing party's details (likely the other party involved):
    - Name, Surname.
    - Bank account details.
    - Additional information about opponent’s property (vehice/compay/apartment)
    - Contact details

### **Claim Details Section:**

- This section contains fields related to the claim:

### **Declaration and Obligations Section:**

- Below the claim details is a block of text where the first party declares that they transfer all claims arising from the accident to the creditor. It includes details about the deductible and the first party's obligation to pursue the claim.

### **Signature Section:**

- At the bottom of the document, there are spaces for signatures and dates:
    - "Place, Date" field.
    - Signature field:

### General Observations:

- The document has a clear structure with consistent use of labeled fields for input.
- The document uses standard fonts with bolding for titles and section headers.
- There are approximately **200-300 words** in total and **around 10-15 sentences** of legal text.

### Document Type Suggestion:

- **Assignment of Claims Form**
"""

form_with_text_template_simple = """
The document layout is structured as a **legal declaration form** with clear sections for input and legal text.

### Header Section:
- A bold, centered title followed by a smaller subtitle introduces the form.
- A section introduces the two parties involved, listing personal and company details.

### Bank Information:
- A section labeled "Bank connection" includes fields for bank account details like IBAN, BIC, and account number.

### Information Section:
- Fields for date, location, and other relevant details such as vehicle, company, or apartment information.

### Opponent Information:
- Fields for the opposing party’s name, bank details, and additional property or contact information.

### Claim Details:
- Specific fields related to the transfer of claims arising from an incident.

### Declaration Section:
- A block of text where the first party declares the transfer of claims to the creditor, detailing obligations.

### Signature Section:
- Spaces for date, place, and signatures at the bottom.

### General Observations:
- The document is well-organized, with labeled input fields and bold section headers. It contains around 200-300 words and 10-15 sentences of legal text.
"""

form_with_text_template_plain = """
The document layout is structured as a legal declaration form with clear sections for input and legal text.
The document contains approximately 200-300 words and around 10-15 sentences.
"""


letter_with_plain_text_template = """
The layout of this document follows the format of a formal **letter** with various sections and detailed information. Below is a detailed description of its layout:

### Document Header:

1. **Top-left corner:**
    - The letter begins with a set of sender details, which include the person’s name, address, and contact information.
    - It also mentions that the communication is sent via email.
2. **Top-right corner:**
    - In the top-right corner, the document contains the contact hours for reaching the sender, with available hours both in person and by phone. Each set of contact hours is organized in two lines for clarity.
    - The sender's team contact details, including phone, and email, are listed.
3. **Date and Location:**
    - The document is dated with the location aligned to the right, just below the contact information.

### **Subject Information Section:**

- This section is structured as follows:
    - Date
    - Claim Number.
    - Ref. number

### **Salutation:**

- The letter begins with a standard formal greeting: "Dear <Surname>,".

### **Main Body of the Letter:**

- The main content of the letter is divided into multiple paragraphs. The paragraphs are fully justified, and the body of the letter provides detailed information regarding a letter subject.
1. **Paragraph 1:**
    - Some information here
2. **Paragraph 2:**
    - Some information here
3. **Paragraph 3-5:**
    - Subject details
4. **Paragraph 6-8:**
    - Personal details
5. **Paragraph 9:**
    - Bank account details

### **Signature Section:**

- The letter ends with a standard closing:
    - There is a sender's name, and a reference to the legal basis for the claim is made. There is also a signature space that is redacted.

### General Observations:

- The letter is well-structured with clear paragraph breaks.
- The document uses a standard, professional font.
- The letter includes personal data, such as names, phone numbers, and case-specific information.
- The letter contains approximately **400-500 words** and around **30-40 sentences**.

### Document Type Suggestion:

- **Claim Letter**
"""

letter_with_plain_text_template_simple = """
The document is structured as a formal **letter** with a clear layout and detailed sections.

### Header Section:
- The top-left contains the sender’s name, address, and contact information, with a note that the communication is sent via email.
- The top-right lists the sender's contact hours, both for phone and in-person communication, as well as team contact details.
- The date and location are aligned to the right, just below the contact details.

### Subject Section:
- Includes the date, claim number, and reference number.

### Salutation:
- A formal greeting such as "Dear <Surname>," introduces the letter.

### Main Body:
- The letter is divided into multiple fully justified paragraphs providing detailed information on the subject matter, including personal and bank account details.

### Signature Section:
- The closing includes the sender's name, reference to legal claims, and a space for a signature.

### General Observations:
- The letter follows a structured, professional format with clear paragraph divisions. It contains around 400-500 words and 30-40 sentences.

### Document Type Suggestion:
- **Claim Letter**
"""

letter_with_plain_text_template_plain = """
The document layout follows a formal letter format with clear sections and detailed information.
The document contains approximately 300-400 words and around 30-40 sentences.
"""

multi_tables_columns_template = """
The layout of this document is structured as a **financial report or invoice review**. Below is a detailed description of its layout:

### Document Header:

1. **Title:**
    - The document begins with a bold, prominent title at the top-left corner: "Invoice Review".
2. **Top-right section:**
    - "Claim number" is printed prominently, along with the **date** and the "Client". These fields are boxed together to the right of the title.
3. "Process number" is shown under the title, aligned to the left below it.

### **Vehicle and Owner Information:**

- The document provides clear, well-organized boxes with information on the left and right for the **client** and the company details.
1. **Left Box (Client Information):**
    - The section is titled **"Client"**.
    - It includes the following details:
        - Client’s personal details
        - Client’s contact information
        - Client’s vehicle.
        - Client’s insurance company.
        - License plate number.
2. **Right Box (Company information):**
    - This section is divided into two subsections:
        1. Company details
        2. Company's contact information, address, and other additional information.

### **Table Section (Financial Details):**

- The most prominent section of the document is the **table**, which provides an overview of the invoice amounts before and after review. It has three main columns:
    1. "Input value (net) before review".
    2. "Correction (net)".
    3. "Output value (net) after review".
- The table consists of multiple rows, each representing a different cost item:
    - "Labor cost".
    - "Spare parts".
    - "Small parts".
    - "Miscellaneous costs".
- "Total amount".
- An additional row calculates "Value improvement", "Old part tax", and "Result (gross)".

### **Footer Section:**

- At the bottom of the document is a **customer service note**, providing contact information for technical inquiries regarding the invoice.
    - There is a field with contact information and working hours.

### General Observations:

- The document is structured into clear sections for different types of information, with a focus on providing corrections and justifying adjustments to invoice values.
- The table is the primary feature of the document, occupying the largest portion of space, and uses clearly defined rows and columns for financial details.
- There are approximately **150-200 words** and **10-15 numeric entries** spread across multiple rows in the table.

### Document Type Suggestion:

- **Invoice Review Report**
"""

multi_tables_columns_template_simple = """
The layout of this document follows a structured format typical of a report. 

### Document Header:
- A bold, prominent title is placed at the top-left, with additional information such as numbers and dates boxed together on the top-right.

### Information Sections:
- Two clear sections are presented side by side:
    1. The left side contains general personal and contact information.
    2. The right side displays company details and contact information.

### Table Section:
- The central feature is a table with three main columns showing values before and after review, along with corrections. Rows include various cost categories, totals, and additional calculations.

### Footer:
- A note at the bottom provides contact information for further inquiries.

### General Structure:
- The document is organized into well-defined sections, with the table taking up the largest portion, and contains text and numeric entries spread across multiple rows.
"""

multi_tables_columns_template_plain = """
The document layout follows a structured format typical of a report, with clear sections for information and financial details.
The document contains approximately 150-200 words and 10-15 numeric entries.
"""

multi_format_template = """
The layout of this document follows the structure of a **service invoice**. Below is a detailed description of its layout:

### **Header:**

- The top section of the document contains **sender information** (the company name, address, and contact details) on the left, and **client information** on the right (such as the customer’s name and address).
- On the right side of the page, there is a **title** indicating the document type, followed by key **reference numbers** and **dates** relevant to the transaction (e.g., invoice number, invoice date, customer number, order number).
- Additional reference information is displayed, such as the **page number** and **tax ID**.

### **Vehicle and Service Details:**

- Below the header, the document provides client’s property **information**, typically grouped into fields for the vehicle's manufacturer, model, license plate, and other identifying information (e.g., registration date, mileage).
- Adjacent to this, details related to the **service process** are presented, including technician names, service personnel, and any relevant inspection dates.

### **Main Content – Itemized Services Table:**

- The most prominent section of the document is a **table** that lists the services provided. It is divided into multiple columns:
    - **Item number**: A unique identifier for each service or part used.
    - **Description**: A brief description of the service or item provided.
    - **Quantity**: The number of units.
    - **Unit price**: The price for one unit.
    - **Amount**: The total amount for that service or item, calculated by multiplying the quantity by the unit price.
    - **VAT**: Information on whether the item/service is taxable.

### **Subtotal and Total Calculations:**

- At the bottom of the itemized table, the document contains **summary fields** that calculate the totals, including:
    - The **subtotal** before any adjustments.
    - Any **taxes** or fees added or deducted.
    - The **final total** that represents the amount due after all adjustments.

### **Footer:**

- The footer provides additional information, including:
    - The **company's contact information** (address, phone numbers, email).
    - **Banking details** for payment, including account numbers and bank codes.
    - **Management or legal details** (names of directors or legal entities).

This abstract structure organizes the document into clear sections, with key information such as reference numbers, vehicle details, and financial calculations displayed in a straightforward, easily understandable format. The use of tables helps clearly delineate each service provided, allowing for transparency in charges.

### General Observations:

- The document is well-structured and cleanly presented, with a professional look typical of a service invoice.
- The itemized services are clearly detailed, with descriptions and pricing in a straightforward format.
- The document contains approximately **200-300 words** with a focus on numeric values in the itemized table.

### Document Type Suggestion:

- **Service Invoice**
"""

multi_format_template_simple = """
The document follows the structure of a **service invoice** with clearly organized sections for key information.

### Header Section:
- The top section contains sender details (company name, address, and contact information) on the left and client details (customer name, address) on the right.
- On the right side, it also displays the document title, reference numbers (invoice number, customer number), and dates.
- Additional information includes the page number and tax ID.

### Vehicle and Service Details:
- Information about the client's property, such as the vehicle's manufacturer, model, and license plate, is provided, along with service personnel and inspection dates.

### Itemized Services Table:
- The central part of the document is an itemized table with columns for:
    - Item number
    - Description of service/item
    - Quantity
    - Unit price
    - Total amount
    - VAT details

### Subtotal and Total:
- The bottom of the table contains summary fields for the subtotal, taxes, and the final total amount due.

### Footer:
- The footer includes the company’s contact information, banking details for payment, and legal/management details.

### General Observations:
- The invoice is well-organized and professional, with a clear layout that details services, pricing, and total amounts. It contains about 200-300 words with a focus on numeric values in the table.

### Document Type Suggestion:
- **Service Invoice**
"""

multi_format_template_plain = """
The document layout follows the structure of a service invoice with clear sections for key information.
The document contains approximately 200-300 words with a focus on numeric values in the itemized table.
"""

letter_with_table_template = """
The layout of this document follows the structure of a **formal letter** with additional **table** elements. Below is a detailed description of its layout:

### Document Header:

1. **Sender Information:**
    - The letter begins with the sender's information, including the company name, address, and contact details.
    - The sender's details are aligned to the left margin.
2. **Recipient Information:**
    - The recipient's details, including the name and address, are aligned to the right margin.
3. **Date and Reference:**
    - The document includes the date and a reference number aligned to the right margin.

### **Salutation and Introduction:**

- The letter starts with a formal salutation: "Dear <Recipient Name>,".
- The introduction provides a brief overview of the purpose of the letter.

### **Main Body of the Letter:**

- The main content of the letter is divided into multiple paragraphs. The paragraphs are fully justified, and the body of the letter provides detailed information regarding the subject.
1. **Paragraph 1:**
    - Introduction to the main topic.
2. **Paragraph 2:**
    - Detailed information about the subject.
3. **Paragraph 3:**
    - Additional details or instructions.
4. **Paragraph 4:**
    - Concluding remarks or summary.

### **Table Section:**

- The letter includes a table that provides additional details or a breakdown of information related to the subject.

### **Closing and Signature:**

- The letter ends with a standard closing:
    - A closing phrase, such as "Sincerely," followed by the sender's name.
    - A space for the sender's signature is included at the bottom of the letter.

### General Observations:

- The document uses a standard font for the letter content.
- The letter is well-structured with clear paragraph breaks and a formal tone.
- The inclusion of a table provides additional detailed information or a structured breakdown of data.
- The letter contains approximately **300-400 words** and around **20-25 sentences**.

### Document Type Suggestion:
    
- **Formal Business Letter**
"""

letter_with_table_template_simple = """
The document is structured as a **formal letter** with an additional **table** for detailed information.

### Header Section:
- Sender’s details (company name, address, and contact) are aligned to the left, while the recipient's information is aligned to the right.
- The date and reference number are also aligned to the right.

### Salutation and Introduction:
- The letter begins with a formal salutation: "Dear <Recipient Name>," followed by an introductory paragraph that outlines the purpose of the letter.

### Main Body:
- The content is divided into multiple paragraphs, fully justified, providing detailed information and instructions:
    1. Introduction to the topic.
    2. Key details about the subject.
    3. Additional instructions.
    4. A concluding summary.

### Table Section:
- A table is included to present specific details or a breakdown of data related to the topic.

### Closing and Signature:
- The letter closes with "Sincerely," followed by the sender’s name and space for a signature.

### General Observations:
- The letter is formal, well-organized, and includes a structured table to supplement the main text. It contains around 300-400 words and 20-25 sentences.

### Document Type Suggestion:
- **Formal Business Letter**
"""

letter_with_table_template_plain = """
The document layout is structured as a formal letter with an additional table for detailed information.
The document contains approximately 300-400 words and around 20-25 sentences.
"""

formal_email_template = """
The layout of the provided document follows the structure of a formal **email**. Here’s a detailed description of the layout:

### Document Header:

1. **Top Section:**
    - The document begins with a line at the top, showing an email account name enclosed in quotes.
    - This line is aligned to the left margin, and it contains a noticeable amount of hidden or redacted information, as indicated by blacked-out or blue-outlined boxes.
2. **Sender/Receiver Information Block:**
    - Below the account name, there is a structured block that lists key email metadata.
    - This section contains three fields:
        - **From:** and **Sent:** with respective sender details and date/time information.
        - **To:** with recipient information.
        - **Subject:** with a subject line, containing a number and date.
    - Each of these fields is arranged in two columns:
        - The left column contains the labels ("From:", "Sent:", etc.) aligned to the left.
        - The right column contains the corresponding information, with each entry spaced and aligned with its label.

### **Horizontal Divider:**

- A black horizontal line separates the header section from the main body of the email.

### **Main Body of the Email:**

- **Salutation:**
    - "Dear <Surname>," appears at the top of the body text, left-aligned.
- **Main Text:**
    - The body of the email contains a short paragraph formatted as follows:
        - **First Paragraph:** Provides a description of a subject.
        - **Second Paragraph:** Additional details about the subject.
    - The text is simple and left-aligned without indentation, covering around 7-8 short lines.
- **Closing:**
    - The closing is a standard email closing:
        - A double dash (--) followed by a polite phrase "Kind regards,".
        - This line is left-aligned and followed by a blank line before the signature block.
        
### **Signature Block:**

- **Contact Information:**
    - The signature block contains information and includes fields for:
        - A name, possibly a title, and company information.
        - **"Tel.:" (Telephone)** and **"Mail:" (Email)** fields, each aligned in a two-column format similar to the earlier header fields.
        
### **General Observations:**

- The document uses a simple, non-stylized font for both the body and header.
- Text alignment is consistently left-aligned across the entire document.
- The document has approximately 120-140 **words** and 7-8 **sentences** in total.

### Document Type Suggestion:

- **Email**
"""

formal_email_template_simple = """
The document is structured as a formal **email** with clearly defined sections.

### Header Section:
- Begins with the sender's email account name, with some information redacted.
- Below, the email metadata is displayed in a two-column format, showing details for "From," "Sent," "To," and "Subject."

### Horizontal Divider:
- A black line separates the header from the body of the email.

### Main Body:
- Starts with a formal salutation ("Dear <Surname>,").
- Contains two short paragraphs providing the main content, left-aligned and simple in format.
- Closes with a standard phrase ("Kind regards,") and a space for the signature.

### Signature Block:
- Contains the sender’s name, title, and contact details, formatted in two columns for telephone and email.

### General Observations:
- The email uses a straightforward, left-aligned layout and non-stylized font. It includes about 120-140 words and 7-8 sentences.

### Document Type Suggestion:
- **Email**
"""

formal_email_template_plain = """
The document layout is structured as a formal email with defined sections.
The document contains approximately 120-140 words and 7-8 sentences.
"""


invoice_template = """
The layout of the provided document follows the structure of an **invoice**. Below is a detailed description of its layout:

### Document Header:

1. **Company Information:**
    - The document begins with the company's name, address, and contact details.
    - This information is aligned to the left margin.
2. **Invoice Details:**
    - The top-right corner contains the invoice number, date, and due date.
    - These details are aligned to the right margin.
    
### **Customer Information:**

- Below the header, the document provides the customer's information, including the name, address, and contact details.
- This information is aligned to the left margin.

### **Itemized Services Table:**

- The main section of the document is an **itemized table** that lists the services provided. It includes columns for:

    1. **Description:** A brief description of the service.
    2. **Quantity:** The number of units.
    3. **Unit Price:** The price per unit.
    4. **Total:** The total cost for that service.
    
- The table consists of multiple rows, each representing a different service provided.

### **Subtotal and Total Calculations:**

- At the bottom of the itemized table, the document contains summary fields that calculate the totals, including:
    - The **subtotal** before any adjustments.
    - Any **taxes** or fees added or deducted.
    - The **final total** that represents the amount due after all adjustments.
    
### **Payment Information:**

- The document includes a section for payment information, such as accepted payment methods, bank details, and payment terms.
- This information is typically aligned to the left margin.

### **Footer:**

- The footer provides additional information, including:
    - Contact information for customer support.
    - Any legal disclaimers or terms and conditions.
    
### General Observations:

- The document is well-structured and clearly presents the invoice details.
- The itemized table provides a breakdown of services and costs.
- The invoice contains approximately **200-300 words** and focuses on numeric values in the itemized table.

### Document Type Suggestion:

- **Invoice**
"""

invoice_template_simple = """
The document is structured as an **invoice** with clearly defined sections for company, customer, and payment details.

### Header Section:
- The top-left contains the company’s name, address, and contact information.
- The top-right includes the invoice number, date, and due date.

### Customer Information:
- Below the header, the customer’s name, address, and contact details are aligned to the left.

### Itemized Services Table:
- A table listing services with columns for:
    1. Description
    2. Quantity
    3. Unit price
    4. Total cost
- Each row represents a different service or item.

### Subtotal and Total:
- Summary fields calculate the subtotal, taxes, and the final total amount due.

### Payment Information:
- A section provides details on payment methods, bank details, and payment terms.

### Footer:
- Additional information like customer support contact and legal disclaimers appears at the bottom.

### General Observations:
- The invoice is well-organized, with around 200-300 words and an emphasis on numeric data in the table.

### Document Type Suggestion:
- **Invoice**
"""

invoice_template_plain = """
The document layout is structured as an invoice with clear sections for company, customer, and payment details.
The document contains approximately 200-300 words and focuses on numeric values in the itemized table.
"""
