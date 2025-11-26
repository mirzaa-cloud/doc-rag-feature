import pdfplumber
import docx
import io

def extract_txt(file_bytes: bytes) -> str:
    """
    Extract plain text from a .txt file.
    """
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")


def extract_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF using pdfplumber.
    """
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)


def extract_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX using python-docx.
    """
    file_stream = io.BytesIO(file_bytes)
    doc = docx.Document(file_stream)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text_by_extension(filename: str, file_bytes: bytes) -> str:
    """
    Detect extension and call the correct extractor.
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "txt":
        return extract_txt(file_bytes)
    elif ext == "pdf":
        return extract_pdf(file_bytes)
    elif ext == "docx":
        return extract_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
