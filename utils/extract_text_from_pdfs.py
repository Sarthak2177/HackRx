import requests
from PyPDF2 import PdfReader
import tempfile

def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp.flush()
        reader = PdfReader(tmp.name)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text
