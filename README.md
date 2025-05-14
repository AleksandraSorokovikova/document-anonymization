# SAND: Synthetic PII Document Generation and Evaluation

## Overview

**SAND** - **S**ynthetic **AN**onymized **D**ocuments - is a synthetic document **generation and evaluation pipeline** for creating realistic, layout-rich one-page documents containing personally identifiable information (PII) annotations. The repository enables researchers to **generate a large corpus of documents with fine-grained PII labels**, apply visual augmentations to simulate scanned document noise, and evaluate the performance of layout-aware models on the synthetic benchmark. Each generated document comes with token-level bounding boxes and BIO-format entity tags across **13 PII categories** (e.g. names, addresses, dates, identifiers), making the data directly compatible with LayoutLM-style models for document understanding. The codebase also includes tools to **fine-tune a LayoutLM model on the synthetic data** and evaluate it against real-world datasets, as well as baseline evaluations using zero-shot multimodal models (Pixtral) and rule-based anonymization (Microsoft Presidio).

---

## Quickstart

This section provides step-by-step instructions to reproduce the core stages of the SAND pipeline. Each notebook corresponds to a key experiment in the paper and can be executed directly via Jupyter. All required files in `data/` and `weights/` are already included.

### Prerequisites

Make sure you installed all dependencies listed in `requirements.txt` before running the notebooks. The repository is designed to be run in a Jupyter Notebook environment.

```bash
pip install -r requirements.txt
```

And make sure the following **system-level tools** are installed to enable LaTeX and HTML-to-PDF rendering:

### System Requirements (Non-Python)

To generate and render documents from LaTeX and HTML templates, the following tools must be available on your system:

* **`xelatex`** — Required for rendering LaTeX documents (comes with TeX Live or MiKTeX)
* **WeasyPrint** — Used for converting HTML to PDF (requires Cairo, Pango, and other libraries)
* Make sure system fonts used in HTML templates are available locally.

<details>
<summary>Installation instructions</summary>

#### On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install texlive-xetex texlive-latex-extra \
                 libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 \
                 libffi-dev libxml2 libxslt1.1 libjpeg-dev zlib1g-dev
```

#### On macOS (using Homebrew):

```bash
brew install cairo pango gdk-pixbuf libffi
```

Then install WeasyPrint via pip:

```bash
pip install weasyprint
```

</details>

These tools are necessary to fully reproduce the PDF rendering and annotation pipeline described in the paper.

---

### 📦 Downloading Supplementary Data

⚠️ **Required for reproducing results and running code**

📁 This step downloads the `weights/` and `data/` folders used in evaluation, and diversity analysis.

You have two options:

---

#### ✅ Option 1: Download and unpack manually

1. Download the archive containing both folders from the following link (provided in the `README.md`):

   > 🔗 **[Download resources.zip](https://drive.google.com/uc?export=download&id=1Ht6PHCGK8U1U3hINlDo2AmQGtPVYC69U)**

2. Unzip the archive. It will contain:

   ```
   supplementary_material.zip
   └── weights/
   └── data/
   ```

3. Move both folders (`weights/`, `data/`) to the **root directory of the repository** — the same location where the Jupyter notebooks are located.

---

#### ⚙️ Option 2: Use the automatic script

You can also use the provided script to download and set everything up automatically:

```bash
pip install gdown
python download_resources.py
```

This script will:

* Download the ZIP archive from Google Drive
* Unpack it
* Move the folders `weights/` and `data/` to the root directory
* Clean up any temporary files

💡 You can customize the target location by editing `download_resources.py`.

❗️**Don't skip this step — without these folders, the notebooks will not run properly.**

---

### 1. Synthetic Document Generation

📄 `dataset_creation.ipynb`

**Generates synthetic documents with token-level PII annotations**

📌 Paper section: `Automatic Annotation and Output Artifacts`

**To run:**

1. Set your OpenAI API key as `OPENAI_API_KEY` in your environment or `.env` file.
2. Open the notebook in Jupyter.
3. Run all cells.

**Outputs saved to:** `dataset/original/`, `layoutlm_labels/`, `entities/`, `latex/` / `html/`, `documents.json`

---

### 2. Visual Augmentation

🖼 `augmentation.ipynb`

**Adds visual distortions to mimic scanned document noise**

📌 Paper section: `Visual Augmentation`

**To run:**

1. Open the notebook in Jupyter.
2. Run all cells.

**Outputs saved to:** `dataset/augmented/` (augmented PNGs, updated annotations, `mapping.json`)

---

### 3. Diversity Metrics and Heatmaps

📊 `measure_datasets.ipynb`

**Computes layout and visual diversity metrics across datasets**

📌 Paper reference: `Table 1: Diversity metrics across three datasets`

**To run:**

1. Open the notebook in Jupyter.
2. Run all cells.

**Outputs:** diversity scores, category entropy, layout uniqueness metrics, heatmaps

---

### 4. LayoutLM Evaluation on SAND and FATURA

📈 `predictions_and_metrics.ipynb`

**Evaluates LayoutLM models fine-tuned on SAND and FATURA**

📌 Paper sections: `Experiment 1`, `Experiment 2` (Tables 2 & 3)

**To run:**

1. Open the notebook in Jupyter.
2. Select which model to evaluate (trained on SAND or FATURA).
3. Run the relevant sections.

**Outputs:** per-category metrics (Precision/Recall/F1), macro-averages, visual comparisons

---

### 5. Baseline Evaluation with Pixtral and Presidio

🧠 `presidio.ipynb`
🧠 `pixtral.ipynb`
**Evaluates non-fine-tuned anonymization baselines**
📌 Paper reference: `Table 4: Performance of Pixtral and Presidio on the FUNSD-PII benchmark`

**To run Presidio:**

1. Open the notebook in Jupyter.
2. Run all cells (CPU execution is sufficient).

⚠️ **Note**: The function `ImageRedactorEngine.redact()` from the `presidio-image-redactor` library was modified to return both the redacted image and the list of bounding boxes.
This is required to evaluate detection performance. The modified version is:

<details>
<summary>Modified redact() function</summary>

<pre><code>
    def redact(
        self,
        image: Image,
        fill: Union[int, Tuple[int, int, int]] = (0, 0, 0),
        ocr_kwargs: Optional[dict] = None,
        ad_hoc_recognizers: Optional[List[PatternRecognizer]] = None,
        **text_analyzer_kwargs,
    ) -> Image:
        """Redact method to redact the given image.

        Please notice, this method duplicates the image, creates a new instance and
        manipulate it.
        :param image: PIL Image to be processed.
        :param fill: colour to fill the shape - int (0-255) for
        grayscale or Tuple(R, G, B) for RGB.
        :param ocr_kwargs: Additional params for OCR methods.
        :param ad_hoc_recognizers: List of PatternRecognizer objects to use
        for ad-hoc recognizer.
        :param text_analyzer_kwargs: Additional values for the analyze method
        in AnalyzerEngine.

        :return: the redacted image
        """

        image = ImageChops.duplicate(image)

        # Check the ad-hoc recognizers list
        self._check_ad_hoc_recognizer_list(ad_hoc_recognizers)

        # Detect PII
        if ad_hoc_recognizers is None:
            bboxes = self.image_analyzer_engine.analyze(
                image,
                ocr_kwargs=ocr_kwargs,
                **text_analyzer_kwargs,
            )
        else:
            bboxes = self.image_analyzer_engine.analyze(
                image,
                ocr_kwargs=ocr_kwargs,
                ad_hoc_recognizers=ad_hoc_recognizers,
                **text_analyzer_kwargs,
            )

        draw = ImageDraw.Draw(image)

        for box in bboxes:
            x0 = box.left
            y0 = box.top
            x1 = x0 + box.width
            y1 = y0 + box.height
            draw.rectangle([x0, y0, x1, y1], fill=fill)

        return image, bboxes
</code></pre>

</details>

If you are using the original version of `presidio`, make sure to apply the patch shown in the notebook or replace the `redact` method manually.

**To run Pixtral (requires GPU):**

1. Use a GPU-enabled environment with support for `flash-attn`.
2. Install required libraries manually (not listed in `requirements.txt` as they cannot be installed on CPU-only systems).
3. Open the notebook in Jupyter and run all cells.

**Outputs:** detection metrics across PII categories on FUNSD-PII

---

## Repository Structure

The repository is organized as follows:

* **`src/`** – Contains all Python modules and scripts for dataset generation, processing, augmentation, training, and evaluation (detailed in [Source Code Modules](#source-code-modules)).
* **`dataset/`** – Output directory for the synthetic dataset. After running the generation pipeline, this folder will contain subfolders with generated documents, annotations, and related metadata (see [Synthetic Dataset Structure](#synthetic-dataset-structure) below for details).
* **`weights/`** – Placeholder for trained model weights. For example, this is expected to hold the fine-tuned LayoutLMv3 model on SAND (not provided in the repository due to size).
* **`data/`** – Placeholder for supplementary data used in analysis. This may include precomputed document embeddings and external datasets (e.g., FUNSD, FATURA with PII annotations) used to replicate experiments from the paper.
* **Jupyter notebooks (in root)** – Several notebooks are provided at the repository root to demonstrate and orchestrate key processes: dataset generation, augmentation, measuring dataset diversity, model training, and evaluation (described in [Jupyter Notebooks](#jupyter-notebooks) below).

## Synthetic Dataset Structure

---

After running the dataset generation pipeline, the `dataset/` folder will be populated with the following content and subfolders (one file or subfolder per generated document unless noted):

* **`original/`** – Contains the **original generated documents in PDF format**. These PDFs are one-page synthetic documents created via LaTeX or HTML templates filled with synthetic PII content.

* **`annotated/`** – Contains **annotated versions of the documents** (e.g. PDFs or images with bounding boxes drawn around PII fields). These serve as a visualization of ground-truth annotations, showing where each PII entity appears on the page.

* **`layoutlm_labels/`** – JSON files for each document with **token-level annotations**, formatted for easy use with LayoutLM models. Each JSON includes a list of `tokens`, their corresponding `bboxes` (bounding box coordinates), and `ner_tags` (BIO-formatted PII entity labels).

* **`entities/`** – Merged entity annotations, where sequences of tokens that form a single PII entity are grouped together. Each entry typically provides the entity text, its category, and a merged bounding box covering the entire entity span (useful for entity-level evaluation).

* **`latex/`** and **`html/`** – The source code used to generate each document. Depending on the template, a document is generated either from a LaTeX source (`.tex` file) or an HTML source (`.html` file). These files contain the text (with inserted PII values) and formatting that was rendered to create the PDF.

* **`signatures/`** – *(Not included in public release)*. This folder originally contained **418 unique PNG files of real-looking signatures** used to simulate handwritten signatures in synthetic documents. Although this signature set was used in the generation of the published dataset, it is **not publicly released** due to licensing and privacy constraints.
  If you wish to use custom signatures when generating new documents, place your PNG files in a directory and specify the path to that folder when initializing the `PIIGenerator` class. If no such folder is provided, a default placeholder signature will be inserted automatically.

* **`augmented/`** – This folder contains the **visually augmented version of each document** to mimic scanned documents:

  * **`augmented/images/`** – Rasterized page images (e.g. PNGs) after applying augmentation (noise, blur, rotation, etc.). These images resemble scanned copies of the original PDFs.

  * **`augmented/layoutlm_labels/`** – Token-level JSON annotations corresponding to the augmented images. The content is analogous to `layoutlm_labels/` above (tokens, bounding boxes, tags), possibly adjusted if augmentations (like slight rotation) alter coordinate positions.

  * **`augmented/labels/`** – Additional label files for augmented images in an object-detection format (if applicable). For example, if using an object detection model for PII (or signatures), this may include annotation files (e.g., YOLO format) with class IDs and bounding boxes for each image.

  * **`augmented/mapping.json`** – A mapping file linking each augmented image back to the original document. This JSON maps document identifiers to their augmented image filenames and ensures consistency between original and augmented data.

* **`documents.json`** – A JSON metadata file listing all generated documents and their attributes. Each entry contains metadata such as document ID, document type (template name or category, e.g. Invoice, Letter, etc.), font family used, whether a signature was included, generation format (LaTeX or HTML), and the assigned dataset split (e.g., train/validation/test). This provides a convenient summary of the dataset composition and allows filtering documents by type or split.

---

## Jupyter Notebooks

To facilitate reproducibility and easy experimentation, the repository provides several Jupyter notebooks at the root. These notebooks demonstrate how to use the code modules for different stages of the project:

* **`dataset_creation.ipynb`** – Runs the end-to-end **generation of the SAND synthetic dataset**. This notebook uses the scripts in `src/` to prompt the language model, render documents (via LaTeX or HTML to PDF), extract annotations, and save all outputs into the `dataset/` folder structure. In order to generate files, you should place your OPENAI API KEY to .env file or set it as an environment variable.
* **`augmentation.ipynb`** – Applies the **visual augmentation pipeline** to the generated documents. It takes the rendered PDFs, rasterizes them to images, and adds realistic noise/transformations (such as blur, speckles, rotations, brightness/contrast shifts) to simulate scanner artifacts and aging. The result is a set of augmented images and updated annotations in `dataset/augmented/`.
* **`measure_datasets.ipynb`** – Computes **layout and content diversity metrics** on SAND and compares them with other datasets (e.g., FUNSD and FATURA). This notebook uses the `measure_dataset.py` module to calculate metrics like PII coverage, entropy of PII class distribution, layout uniqueness, and visual intra-diversity, providing quantitative insight into how diverse the synthetic dataset is relative to real document collections.
* **`pixtral.ipynb`** – Evaluates a **zero-shot multimodal model (Pixtral)** on the SAND benchmark. Pixtral (a large vision-language model by Mistral AI) is applied to the documents without fine-tuning, and this notebook assesses its ability to detect PII out-of-the-box. Results from Pixtral serve as a baseline for comparison against fine-tuned models.
* **`train_layoutlm.ipynb`** – While training can also be done via script, a notebook may be provided to interactively fine-tune **LayoutLMv3 on the synthetic data**. This would walk through preparing the dataset, configuring the model and training hyperparameters, and launching the fine-tuning process, producing a model saved under `weights/`.
* **`predictions_and_metrics.ipynb`** – Evaluates a **trained LayoutLM model** on a test set. Using the fine-tuned LayoutLM (either provided in `weights/` or obtained via training), this notebook runs inference on a set of documents (e.g., the validation or test split, or real annotated documents like FUNSD-PII) and computes evaluation metrics. It reports token-level and entity-level Precision, Recall, and F1 scores for PII detection, and can visualize predictions versus ground truth.
* **`presidio.ipynb`** – Evaluates **rule-based anonymization using Microsoft Presidio** on the synthetic benchmark or real documents. This notebook runs Presidio (a rule/pattern-based PII detection tool) to identify PII in the documents and compares its performance to the ground truth annotations. The results illustrate how a rule-based system performs relative to learning-based approaches on this dataset.

## Source Code Modules

All core functionality is implemented in Python scripts under the `src/` directory. Below is a summary of the key modules and their roles:

* **`generate_document.py`** – Handles **synthetic document content generation**. This script uses a Large Language Model (LLM) (via the OpenAI API) to generate document content given a set of PII fields. It prompts the LLM to produce either LaTeX or HTML code for a document that embeds the provided PII values in a realistic layout. The output is a self-contained LaTeX or HTML source that can be rendered to a one-page PDF. (Internally, this module orchestrates layout template selection, calls the LLM with appropriate prompts, and may call rendering tools like XeLaTeX or HTML-to-PDF converters.)
Вот исправленный вариант описания в нужном формате:
* **`dataset_processing.py`** – Prepares the **final training dataset for LayoutLM** based on generated documents. It applies visual augmentations (by interacting with `augmentation.py`), cleans and merges BIO-style entity tags, and converts token-level annotations into JSON files compatible with LayoutLM models. It also handles grouping tokens into lines, sorting by reading order, and generating `tokens`, `bboxes`, and `ner_tags` for each document. This module finalizes the structured annotations used during model training.
* **`augmentation.py`** – Implements the **visual augmentation pipeline** that simulates scanning artifacts. It contains functions to apply a variety of image transformations while preserving layout structure. Augmentations include adding Gaussian noise, salt-and-pepper noise, motion blur, Gaussian blur, small rotations and translations (to mimic misalignment), brightness/contrast adjustments (simulating faded ink or scanner exposure), and JPEG compression artifacts. This module takes a clean document image and outputs a degraded version; it is used by `augmentation.ipynb` to produce the `augmented/images/` and updated annotations.
* **`measure_dataset.py`** – Contains routines for **quantifying dataset diversity**. This includes computing metrics such as:

  * *PII coverage* (what percentage of the page is covered by PII bounding boxes),
  * *Entropy-based diversity score* (how balanced the frequency of different PII entity types is across the dataset),
  * *Layout uniqueness* (the proportion of documents with unique spatial layouts of PII elements),
  * *Visual intra-diversity* (the average feature distance between documents’ appearances, e.g. using CLIP embeddings for pages).
    It may utilize pre-trained models or embeddings (e.g., CLIP from OpenAI) to compute visual similarity. This module is used to produce the statistics and plots comparing SAND with other datasets.
* **`train_layoutlm.py`** – Script for **fine-tuning a LayoutLM model** (specifically LayoutLMv3 in our experiments) on the synthetic dataset. It uses Hugging Face Transformers and Datasets libraries to load the token-level annotations into a dataset, configures the LayoutLM processor and model for token classification (NER), and runs training. After training, the fine-tuned model is saved (expected to be placed in `weights/` directory).
* **`evaluate.py`** – Supports **evaluation of model predictions** against ground truth labels. Given a set of predicted tokens with tags (e.g., from a model on a test document) and the true annotations, this script calculates precision, recall, and F1 scores at the token level and entity level. It can produce overall metrics as well as breakdowns per category. It also has utilities to create visual comparisons of predictions vs. ground truth (for example, drawing bounding boxes for true vs predicted PII on the document image for inspection).
* **`AnonymizationInference.py`** – A utility script for **running inference with a trained model on new documents**. This script streamlines the process of loading a fine-tuned LayoutLM model (or other detection model) and applying it to document images. It can take in a folder of test images (e.g., real scanned documents), use the model to predict PII bounding boxes or masks, and then output annotated images or JSON predictions. In our use case, this was helpful for applying the fine-tuned LayoutLMv3 to the real FUNSD forms. *(Note: This module also includes integration with YOLO from Ultralytics, which might be used to detect non-textual elements like signatures or to experiment with an object-detection approach to PII detection.)*
* **`process.py`** – Contains various **post-processing and helper functions** for filtering and visualizing results. For example, it can create side-by-side image comparisons of ground truth vs. model predictions, apply filtering rules to model outputs (such as removing low-confidence detections or merging overlapping boxes), and prepare results for final evaluation. It assists the evaluation pipeline by cleaning up model outputs and generating figures used in analysis.
* **`config.py`** – Defines **global configuration settings** and constants used across the codebase. This includes definitions of the PII entity categories and their mappings, color codes for visualization, prompt templates for the LLM (imported from `prompts.py`), and other adjustable parameters (like the list of document types, layout structures, fonts, and any random seed or counters for document generation). By adjusting values in `config.py`, users can control aspects of the dataset generation (such as which document templates to sample or how many documents to create).
* **`prompts.py`** - Contains **prompt templates** used to generate document content via the LLM. This includes templates for generating LaTeX or HTML code, as well as specific instructions for the LLM to fill in PII fields. The prompts are designed to elicit realistic and coherent document structures while embedding the specified PII values. This module is used by `generate_document.py` to create the content of each document.

## Dependencies and Installation

To use this repository, the following major dependencies are required:

* **Python 3.x** (development and testing done with Python 3.9+).
* **PyTorch** (for model training and inference, e.g. torch 1.12+ compatible with Transformers).
* **Hugging Face Transformers** (for the LayoutLMv3 model and AutoProcessor).
* **Hugging Face Datasets** (to handle the dataset of tokens and labels for training).
* **OpenAI API** (for document content generation using GPT; the `openai` Python package is used. An API key and internet access are needed to actually call the model).
* **PyMuPDF (`fitz`)** (for PDF parsing and text extraction to get token bounding boxes).
* **OpenCV (`cv2`)** (used in augmentation for image processing operations).
* **Pillow (PIL)** (image manipulation for drawing annotations, etc.).
* **numpy, pandas** (general data handling).
* **scikit-learn** (for metrics and dimensionality reduction like t-SNE in diversity analysis).
* **matplotlib, seaborn** (for plotting diversity metrics and visualizations).
* **tqdm** (for progress bars in processing scripts).
* **FuzzyWuzzy** (for string matching or fuzzy comparisons if needed during generation).
* **Ultralytics YOLOv8** (the `ultralytics` package, if using the YOLO-based inference for signatures or alternate PII detection approach in `AnonymizationInference.py`).
* **Microsoft Presidio** (if replicating the Presidio baseline, the `presidio-analyzer` package should be installed).

In addition to Python packages, **several system-level tools** are needed for rendering documents:

* **XeLaTeX** – A TeX engine capable of compiling LaTeX files with custom fonts. A TeX distribution (e.g. TeX Live or MiKTeX) including the `xelatex` command must be installed and accessible in the system PATH. This is used to render the LaTeX templates into PDF documents.
* **WeasyPrint** – An HTML/CSS to PDF rendering tool. Install the `weasyprint` Python package (and ensure its dependencies, such as Cairo and Pango, are satisfied on your system) to enable conversion of HTML templates to PDF. WeasyPrint is used to accurately render HTML-based document templates with complex layouts into PDF format.

*Note:* If using `pdfkit` as an alternative for HTML to PDF, ensure `wkhtmltopdf` is installed. However, WeasyPrint is recommended for its compatibility with advanced CSS and for maintaining consistent layout. All LaTeX/HTML generation code will automatically call the appropriate renderer (XeLaTeX for LaTeX sources, WeasyPrint for HTML sources) given these dependencies.

Before running the notebooks or scripts, make sure to install the required Python libraries (e.g., via `pip install -r requirements.txt`). Also, ensure that the system commands `xelatex` and `weasyprint` are working. If the OpenAI API is used for generation, set your API key (e.g., in an environment variable `OPENAI_API_KEY`) prior to running the generation script.

## Pretrained Models and External Data

The repository is primarily focused on synthetic data generation and model training using that data. To fully reproduce the experiments from the accompanying research paper, additional resources are used which are not distributed in the code repository:

* **Fine-Tuned LayoutLMv3 Weights (`weights/` folder)** – After training on SAND, a LayoutLMv3 model checkpoint was saved.
* **External Datasets and Precomputed Embeddings (`data/` folder)** – The `data/` directory to contains additional data used for evaluation:

  * *FUNSD-PII*: A version of the FUNSD form-understanding dataset manually annotated with PII entity labels (used as a real-world test set in the paper). Approximately 250 real scanned forms with ground-truth PII boxes were used.
  * *FATURA-PII*: A set of 100 invoice documents from the FATURA synthetic dataset, annotated with PII labels for evaluation.
  * *Embeddings/Features*: Precomputed document embeddings (CLIP fine-tuned image encoder for each page) and other intermediate data used in the diversity analysis.

  These files are only required if one intends to reproduce the *exact experimental comparisons* (such as those in the paper’s Section 4 or diversity analysis). Using the generation pipeline and training on SAND **does not require** these external files. In other words, you can generate the synthetic dataset and train a model with just the code and dependencies, but to compare with FUNSD or FATURA in evaluations, you would need the annotated versions of those datasets as prepared by the authors.

**Important:** The absence of `weights/` and `data/` in the repository will not affect the core functionality of generating the SAND dataset or training a new model on it. They are provided for completeness in experiments replication. If you only wish to use SAND for training or evaluation of your own methods, you can do so with the synthetic data alone. 
