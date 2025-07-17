import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Update this path to where Tesseract is installed on your system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Input/output directories
PDF_FOLDER = "Train"  # put your 5 PDFs in this folder
TEXT_FOLDER = "extracted_texts"
os.makedirs(TEXT_FOLDER, exist_ok=True)

def extract_text_with_pypdf2(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"[PDF2 ERROR] {pdf_path}: {e}")
        return ""

def extract_text_with_ocr(pdf_path):
    try:
        print(f"OCR processing: {pdf_path}")
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"[OCR ERROR] {pdf_path}: {e}")
        return ""

def extract_and_save_text(pdf_path):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    text_output_path = os.path.join(TEXT_FOLDER, f"{filename}.txt")

    text = extract_text_with_pypdf2(pdf_path)
    if not text or len(text) < 100:  # fallback if text is too short
        text = extract_text_with_ocr(pdf_path)

    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved text to: {text_output_path}")

def main():
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            extract_and_save_text(pdf_path)

if __name__ == "__main__":
    main()
