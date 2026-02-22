# sync_docs.py
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from openai import OpenAI

from db import init_db, get_doc, upsert_doc
from ingest import sha256_of_file, extract_text

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".xlsx"}


# ----------------------------
# METADATA DESDE RUTA
# ----------------------------
def extract_metadata_from_path(path: Path, docs_root: Path, normalized: bool):
    rel = path.relative_to(docs_root)
    parts = list(rel.parts)

    metadata = {
        "source_path": str(path),
        "source_name": path.name,
        "source_ext": path.suffix.lower(),
        "normalized": normalized
    }

    def is_year(s: str) -> bool:
        return s.isdigit() and len(s) == 4

    if not parts:
        return metadata

    # Caso 1: empieza con año
    if is_year(parts[0]):
        metadata["anio"] = parts[0]
        if len(parts) >= 2:
            metadata["materia"] = parts[1]
        if len(parts) >= 3:
            metadata["unidad"] = parts[2]
        return metadata

    # Caso 2: empieza con materia
    metadata["materia"] = parts[0]

    # Si el segundo es año
    if len(parts) >= 2 and is_year(parts[1]):
        metadata["anio"] = parts[1]
        if len(parts) >= 3:
            metadata["unidad"] = parts[2]
    else:
        # si no hay año, y hay 2do nivel, lo usamos como "unidad" si existe
        if len(parts) >= 2:
            metadata["unidad"] = parts[1]

    return metadata

# ----------------------------
# VECTOR STORE
# ----------------------------
def get_or_create_vector_store(client: OpenAI, name: str) -> str:
    stores = client.vector_stores.list(limit=100)
    for s in stores.data:
        if getattr(s, "name", None) == name:
            return s.id
    created = client.vector_stores.create(name=name)
    return created.id


# ----------------------------
# UPLOADS
# ----------------------------
def upload_text_as_file(client: OpenAI, text: str):
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="assistants")
        return uploaded.id
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def upload_original_file(client: OpenAI, path: Path):
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    return uploaded.id


def attach_file_to_vector_store(client: OpenAI, vector_store_id: str, file_id: str, attributes: dict):
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id,
        attributes=attributes
    )


# ----------------------------
# MAIN
# ----------------------------
def main():
    load_dotenv()

    api_key = os.environ["OPENAI_API_KEY"]
    docs_root = Path(os.environ["DOCS_ROOT"]).expanduser()
    vs_name = os.environ.get("VECTOR_STORE_NAME", "ISDM_DOCENCIA")

    client = OpenAI(api_key=api_key)

    init_db()
    vector_store_id = get_or_create_vector_store(client, vs_name)
    print("Vector store:", vector_store_id)

    for path in docs_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        file_sha = sha256_of_file(path)
        rec = get_doc(str(path))

        # Si no cambió, no reindexamos
        if rec and rec[1] == file_sha:
            continue

        print("Processing:", path)

        text = extract_text(path)

        # Si se pudo normalizar texto (excel/docx/pdf)
        if text and len(text.strip()) > 0:
            openai_file_id = upload_text_as_file(client, text)
            attrs = extract_metadata_from_path(path, docs_root, normalized=True)
        else:
            openai_file_id = upload_original_file(client, path)
            attrs = extract_metadata_from_path(path, docs_root, normalized=False)

        attach_file_to_vector_store(client, vector_store_id, openai_file_id, attrs)

        upsert_doc(str(path), file_sha, openai_file_id)

        print("Indexed:", path)

    print("Done.")


if __name__ == "__main__":
    main()