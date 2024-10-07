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
    - Ensure that all provided PII entities are embedded within the HTML content and placed in appropriate sections (e.g., full name in the full name field, address in the address field).
    - Vary the sentence structure for embedding PII data. For example:
      - “Contact [Full Name] at [Phone Number].”
      - “[Full Name] can be reached via [Phone Number].”
    - Follow the structure provided in the **Document Structure** section to organize the content.
    - You MUST put ALL provided PII entities in the document. Check if any PII entity is missing.

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
