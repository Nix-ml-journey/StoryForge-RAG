import fitz 
import logging 
import re
import yaml
import os 
from tqdm import tqdm
from pathlib import Path
from epub_to_text import EpubProcessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

config_dir = Path(__file__).parent
config_file = Path(__file__).parent.parent / "setup.yaml"
with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
    downloaded_rawbook_dir = config["Downloaded_rawbook_dir"]
    extracted_text_dir = config["Extracted_text_dir"]

BASE_PATH = config.get("BASE_PATH")

def extract_text_from_pdf(file_path):
    doc = None
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text += (b[4] or "") + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {type(e).__name__}: {str(e) or repr(e)}")
        return None
    finally:
        if doc is not None:
            doc.close()

def extract_text_from_epub(file_path):
    try:
        base_out = Path(BASE_PATH) / extracted_text_dir
        base_out.mkdir(parents=True, exist_ok=True)
        book_name = Path(file_path).stem
        output_dir = base_out / book_name
        output_dir.mkdir(parents=True, exist_ok=True)
        epub_processor = EpubProcessor(str(file_path), str(output_dir))
        epub_processor.export_chapters_markdown()

        parts = []
        for root, _, files in os.walk(output_dir):
            for f in sorted(files):
                if f.endswith(('.md', '.txt')):
                    path = Path(root) / f
                    parts.append(path.read_text(encoding='utf-8'))
        return '\n\n'.join(parts) if parts else None
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        return None
        
def format_ocr_text(text):
    if text is None:
        return ""
    lines = text.split("\n")

    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            processed_lines.append(stripped_line)

    if not processed_lines:
        return text

    reformatted_text = []
    current_paragraph = []

    for i, line in enumerate(processed_lines):
        is_new_paragraph = False
        
        if i == 0:
            is_new_paragraph = True
        elif current_paragraph:
            last_line = current_paragraph[-1]

            if(last_line.endswith((".", "?", "!")) and 
                line and line[0].isupper()):
                is_new_paragraph = True

            elif re.match(r'^\d+$', line) or re.match(r'^[A-Z\s]+$', line):
                is_new_paragraph = True

        if is_new_paragraph and current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            reformatted_text.append(paragraph_text)
            current_paragraph = []

        current_paragraph.append(line)

    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        reformatted_text.append(paragraph_text)

    result = '\n\n'.join(reformatted_text)
    result = re.sub(r' +', ' ', result)
    result = re.sub(fr'\n{3,}', '\n\n', result)
    return result

def process_ocr_file(input_path, output_path=None):

    input_file = Path(input_path)
    if not input_file.exists():
        logging.error(f"File not found: {input_path}")
        return None
    original_text = extract_text_from_pdf(input_file)
    if original_text is None:
        return None
    formatted_text = format_ocr_text(original_text)
    if output_path is None:
        output_file = input_file.parent / f"{input_file.stem}_formatted{input_file.suffix}"
    else:
        output_file = Path(output_path)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        logging.info(f"Reformatted text saved to: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error writing file {output_file}: {e}")
        return None

def process_epub_file(input_path, output_path=None):

    input_file = Path(input_path)
    if not input_file.exists():
        logging.error(f"File not found: {input_path}")
        return None
    original_text = extract_text_from_epub(input_file)
    if original_text is None:
        return None
    formatted_text = format_ocr_text(original_text)
    if output_path is None:
        output_file = input_file.parent / f"{input_file.stem}_formatted{input_file.suffix}"
    else:
        output_file = Path(output_path)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        logging.info(f"Formatted text saved to: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error writing file {output_file}: {e}")
        return None

def extract_text_from_books():
    books_path = Path(BASE_PATH) / downloaded_rawbook_dir
    out_dir = Path(BASE_PATH) / extracted_text_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(books_path.glob('*.pdf'))
    epub_files = list(books_path.glob('*.epub'))

    for pdf_file in tqdm(pdf_files, desc="Extracting text", unit="file"):
        try:
            extracted_text = extract_text_from_pdf(pdf_file)

            if extracted_text is None:
                logging.warning(f"Skipped {pdf_file.name} due to extraction error")
                continue

            better_text = format_ocr_text(extracted_text)

            txt_file = out_dir / f"{pdf_file.stem}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(better_text)
                logging.info(f"Successfully extracted text: {txt_file.name}")

        except Exception as e:
            logging.error(f"Error processing {pdf_file.name}: {e}")
            continue

    for epub_file in tqdm(epub_files, desc="Extracting text", unit="file"):
        try:
            extracted_text = extract_text_from_epub(epub_file)
            if extracted_text is None:
                logging.warning(f"Skipped {epub_file.name} due to extraction error")
                continue
            better_text = format_ocr_text(extracted_text)
            txt_file = out_dir / f"{epub_file.stem}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(better_text)
                logging.info(f"Successfully extracted text: {txt_file.name}")
        except Exception as e:
            logging.error(f"Error processing {epub_file.name}: {e}")
            continue

def save_extracted_text():
    out_dir = Path(BASE_PATH) / extracted_text_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in out_dir.glob('*.txt'):
        logging.info(f"Text file present: {file.name}")
    return out_dir