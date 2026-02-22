# ingest.py
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from docx import Document
from pypdf import PdfReader
from pypdf.errors import DependencyError

TEXT_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PDF_EXTS  = {".pdf"}
XLSX_EXTS = {".xlsx"}  # (si querés .xls luego vemos)

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts)

def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))

        # Si está encriptado y no se puede abrir sin password
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # intenta sin password
            except Exception:
                return ""  # no se puede extraer texto

        parts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            txt = txt.strip()
            if txt:
                parts.append(f"\n--- PAGE {i+1} ---\n{txt}")
        return "\n".join(parts)

    except DependencyError:
        # Falta cryptography para AES u otros
        return ""
    except Exception:
        # PDF corrupto, raro, etc.
        return ""

def read_xlsx(path: Path, max_rows_per_sheet: int = 400) -> str:
    # Convierte cada hoja en texto tipo tabla
    xl = pd.ExcelFile(path)
    parts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).head(max_rows_per_sheet)
        df = df.dropna(how="all")
        if df.empty:
            continue
        parts.append(f"\n--- SHEET: {sheet} ---\n")
        parts.append(df.to_csv(index=False))
    return "\n".join(parts)

def extract_text(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return read_text_file(path)
    if ext in DOCX_EXTS:
        return read_docx(path)
    if ext in PDF_EXTS:
        return read_pdf(path)
    if ext in XLSX_EXTS:
        return read_xlsx(path)
    return None