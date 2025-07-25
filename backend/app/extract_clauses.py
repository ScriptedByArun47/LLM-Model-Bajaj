# extract_clauses.py
import tempfile
import requests
import os
import fitz  # PyMuPDF
from uuid import uuid4

def download_pdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise exception if download fails

    # Create a temp file with .pdf extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

def extract_clauses_from_url(url):
    file_path = download_pdf_from_url(url)
    doc = fitz.open(file_path)
    text_blocks = []

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block[4].strip()) > 80:  # only non-trivial text
                text_blocks.append(block[4].strip())

    doc.close()
    os.remove(file_path)

    # Improved clause segmentation
    clauses = [{"clause": tb} for tb in text_blocks if len(tb) > 80]
    return clauses
