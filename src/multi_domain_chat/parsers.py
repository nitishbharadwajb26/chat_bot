from pathlib import Path
from docx import Document
from pypdf import PdfReader

def parse_docx(file_path: Path) -> str:
    """
    Parses a .docx file and returns its text content.
    """
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join([cell.text.strip() for cell in row.cells])
            text += f"\n{row_text}"
    return text

def parse_pdf(file_path: Path) -> str:
    """
    Parses a .pdf file and returns its text content.
    """
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def parse_file(file_path: Path) -> str:
    """
    Determines the file type based on its extension and parses it accordingly.
    Supports .docx and .pdf files.
    """
    if file_path.suffix.lower() == ".docx":
        return parse_docx(file_path)
    elif file_path.suffix.lower() == ".pdf":
        return parse_pdf(file_path)
    elif file_path.suffix.lower() in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")