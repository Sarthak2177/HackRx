import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import docx
import extract_msg

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Input/output
DOC_FOLDER = "Train"  # put all .pdf, .docx, .msg here
TEXT_FOLDER = "extracted_texts"
os.makedirs(TEXT_FOLDER, exist_ok=True)

# ---------------- PDF ----------------
def extract_text_with_pypdf2(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[PDF2 ERROR] {pdf_path}: {e}")
        return ""

def extract_text_with_ocr(pdf_path):
    try:
        print(f"OCR processing: {pdf_path}")
        images = convert_from_path(pdf_path)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        print(f"[OCR ERROR] {pdf_path}: {e}")
        return ""

# ---------------- Word ----------------
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[DOCX ERROR] {docx_path}: {e}")
        return ""

# ---------------- Email ----------------
def extract_text_from_msg(msg_path):
    try:
        msg = extract_msg.Message(msg_path)
        return f"Subject: {msg.subject}\nBody:\n{msg.body}"
    except Exception as e:
        print(f"[MSG ERROR] {msg_path}: {e}")
        return ""

# ---------------- Universal ----------------
def extract_and_save_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(TEXT_FOLDER, f"{filename}.txt")

    if ext == ".pdf":
        text = extract_text_with_pypdf2(file_path)
        if not text or len(text) < 100:
            text = extract_text_with_ocr(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext == ".msg":
        text = extract_text_from_msg(file_path)
    else:
        print(f"[SKIPPED] Unsupported file: {file_path}")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())
    print(f"✅ Saved text to: {output_path}")

# ---------------- Main ----------------
def main():
    for filename in os.listdir(DOC_FOLDER):
        if filename.lower().endswith((".pdf", ".docx", ".msg")):
            extract_and_save_text(os.path.join(DOC_FOLDER, filename))

if __name__ == "__main__":
    main()