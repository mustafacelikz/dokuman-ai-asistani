from pathlib import Path


def read_txt(file_path):
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    return text

from pypdf import PdfReader


def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def read_document(file_path):
    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".txt":
        return read_txt(file_path)

    elif extension == ".pdf":
        return read_pdf(file_path)

    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {extension}")