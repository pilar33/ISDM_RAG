import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from openai import OpenAI

from db import init_db, get_doc, upsert_doc
from ingest import sha256_of_file, extract_text

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".xlsx"}

ROMAN_MAP = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
}


def parse_unidad(folder_name: str) -> dict:
    """
    Devuelve:
      - unidad (texto original)
      - unidad_num (1..n) si se puede detectar
      - tipo (programa/unidad/otros)
    """
    name = folder_name.strip()

    if not name:
        return {"unidad": None, "unidad_num": None, "tipo": "otros"}

    # Programa
    if name.lower() == "programa":
        return {"unidad": "Programa", "unidad_num": None, "tipo": "programa"}

    # Casos tipo: UnidadI, Unidad I, Unidad1, Unidad 1, UnidadIV
    compact = name.lower().replace(" ", "")
    m = re.match(r"^unidad([ivx]+|\d+)$", compact)
    if m:
        raw = m.group(1).upper()
        unidad_num = ROMAN_MAP.get(raw) if raw.isalpha() else raw
        return {"unidad": folder_name, "unidad_num": unidad_num, "tipo": "unidad"}

    # Intento alternativo
    m2 = re.match(r"^unidad\s*([IVX]+|\d+)$", name, flags=re.IGNORECASE)
    if m2:
        raw = m2.group(1).upper()
        unidad_num = ROMAN_MAP.get(raw) if raw.isalpha() else raw
        return {"unidad": folder_name, "unidad_num": unidad_num, "tipo": "unidad"}

    return {"unidad": folder_name, "unidad_num": None, "tipo": "otros"}


def extract_metadata_from_path(path: Path, docs_root: Path, normalized: bool):
    """
    Estructura asumida:
        DOCS_ROOT = .../2025
        rel.parts = [materia, unidad/programa/otros, subcarpetas..., archivo]

    Ejemplos:
        TallerDeProgramacion / Unidad I / clase1.docx
        Practica Profesional II / Programa / programa.pdf
        IntroduccionProgramacion / Materiales / apunte.docx
    """
    rel = path.relative_to(docs_root)
    parts = list(rel.parts)

    metadata = {
        "source_path": str(path),
        "source_name": path.name,
        "source_ext": path.suffix.lower(),
        "normalized": normalized,
    }

    # Año tomado desde DOCS_ROOT si termina en 2025, 2026, etc.
    root_name = docs_root.name
    if root_name.isdigit() and len(root_name) == 4:
        metadata["anio"] = root_name

    if not parts:
        return metadata

    # Primera carpeta debajo de DOCS_ROOT
    # La guardamos como metadata disponible, pero no se usa para filtrar
    if len(parts) >= 1:
        metadata["materia"] = parts[0]

    # Segunda carpeta: programa / unidad / otros
    if len(parts) >= 2:
        uinfo = parse_unidad(parts[1])
        metadata.update(uinfo)

    # Subcarpetas intermedias, por si después sirven
    if len(parts) > 3:
        metadata["subfolders"] = list(parts[2:-1])

    return metadata


def get_or_create_vector_store(client: OpenAI, name: str) -> str:
    stores = client.vector_stores.list(limit=100)
    for s in stores.data:
        if getattr(s, "name", None) == name:
            return s.id
    created = client.vector_stores.create(name=name)
    return created.id


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
        attributes=attributes,
    )


def main():
    load_dotenv()

    api_key = os.environ["OPENAI_API_KEY"]
    docs_root = Path(os.environ["DOCS_ROOT"]).expanduser()
    vs_name = os.environ.get("VECTOR_STORE_NAME", "ISDM_DOCENCIA")

    if not docs_root.exists():
        raise FileNotFoundError(f"DOCS_ROOT no existe: {docs_root}")

    client = OpenAI(api_key=api_key)

    init_db()
    vector_store_id = get_or_create_vector_store(client, vs_name)
    print("Vector store:", vector_store_id)

    for path in docs_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        name = path.name

        # Ignorar temporales / basura
        if name.startswith("~$") or name.startswith("."):
            continue
        if name.lower() in {"thumbs.db", "desktop.ini"}:
            continue

        # Saltar archivos vacíos
        if path.stat().st_size == 0:
            print("Skipping empty file:", path)
            continue

        file_sha = sha256_of_file(path)
        rec = get_doc(str(path))

        # Si no cambió, no reindexar
        if rec and rec[1] == file_sha:
            continue

        print("Processing:", path)

        text = extract_text(path)

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