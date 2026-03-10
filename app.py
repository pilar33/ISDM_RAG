# app.py
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

app = FastAPI(title="ISDM RAG API", version="0.4.1")

# -------------------------
# SEGURIDAD API KEY
# -------------------------

API_KEY = os.getenv("ISDM_API_KEY")
API_KEY_NAME = "X-API-KEY"

api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    auto_error=False
)


def require_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="ISDM_API_KEY not configured")

    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return api_key


# -------------------------
# CONFIG / CLIENT
# -------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing env var: OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

VECTOR_STORE_NAME = os.environ.get("VECTOR_STORE_NAME", "ISDM_2025_C1")

# Ruta base pedida por vos
OUTPUT_ROOT_DIR = Path(
    os.environ.get(
        "OUTPUT_ROOT_DIR",
        r"C:\xampp\htdocs\ProfePilar\data\ISDM\2026\ISDM"
    )
)


def get_vector_store_id() -> str:
    stores = client.vector_stores.list(limit=100)
    for s in stores.data:
        if getattr(s, "name", None) == VECTOR_STORE_NAME:
            return s.id
    created = client.vector_stores.create(name=VECTOR_STORE_NAME)
    return created.id


VECTOR_STORE_ID = get_vector_store_id()

# -------------------------
# BASIC ENDPOINTS (PÚBLICOS)
# -------------------------


@app.get("/")
def root():
    return {"ok": True, "service": "ISDM_RAG", "version": app.version}


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# MODELOS
# -------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    k: int = Field(8, ge=1, le=20)
    anio: Optional[str] = None
    materia: Optional[str] = None
    unidad: Optional[str] = None
    unidad_num: Optional[str] = None
    tipo: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    score: float
    file_id: str
    attributes: Optional[dict] = None


class SearchResponse(BaseModel):
    vector_store_id: str
    results: list[SearchResult]


class ImproveRequest(BaseModel):
    objetivo: str = Field(..., min_length=5)
    query: str = Field(..., min_length=2)
    anio: Optional[str] = None
    materia: Optional[str] = None
    unidad: Optional[str] = None
    k: int = Field(10, ge=3, le=20)
    unidad_num: Optional[str] = None
    tipo: Optional[str] = None

    modo_fuentes: str = Field(
        "optional",
        pattern="^(required|optional|none)$"
    )

    nivel: str = "Superior ISDM"
    formato_salida: str = "plan_clase"
    tono: str = "profesional_cercano"
    restricciones: Optional[str] = None


class ImproveResponse(BaseModel):
    used_sources: list[dict]
    output: str


# -------------------------
# HELPERS
# -------------------------


def sanitize_filename(value: Optional[str], fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        text = fallback

    replacements = {
        "á": "a", "à": "a", "ä": "a", "â": "a",
        "é": "e", "è": "e", "ë": "e", "ê": "e",
        "í": "i", "ì": "i", "ï": "i", "î": "i",
        "ó": "o", "ò": "o", "ö": "o", "ô": "o",
        "ú": "u", "ù": "u", "ü": "u", "û": "u",
        "ñ": "n",
        "Á": "A", "À": "A", "Ä": "A", "Â": "A",
        "É": "E", "È": "E", "Ë": "E", "Ê": "E",
        "Í": "I", "Ì": "I", "Ï": "I", "Î": "I",
        "Ó": "O", "Ò": "O", "Ö": "O", "Ô": "O",
        "Ú": "U", "Ù": "U", "Ü": "U", "Û": "U",
        "Ñ": "N",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text.strip())
    return text or fallback


def normalize_materia_folder(materia: Optional[str]) -> str:
    raw = (materia or "").strip().lower()

    aliases = {
        "taller de programación": "TallerDeProgramacion",
        "taller de programacion": "TallerDeProgramacion",
        "taller_programacion": "TallerDeProgramacion",
        "tallerdeprogramacion": "TallerDeProgramacion",

        "práctica profesional ii": "PracticaProfesionalII",
        "practica profesional ii": "PracticaProfesionalII",
        "práctica profesional 2": "PracticaProfesionalII",
        "practica profesional 2": "PracticaProfesionalII",
        "pp2": "PracticaProfesionalII",
        "practicaprofesionalii": "PracticaProfesionalII",
    }

    if raw in aliases:
        return aliases[raw]

    cleaned = sanitize_filename(materia, "General")
    parts = [p for p in cleaned.split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "General"


def detect_file_prefix(req: ImproveRequest) -> str:
    text = f"{req.query} {req.objetivo} {req.formato_salida} {req.tipo or ''}".lower()

    if "primera clase" in text or "clase 1" in text or "clase uno" in text:
        return "Clase_01"
    if "segunda clase" in text or "clase 2" in text or "clase dos" in text:
        return "Clase_02"
    if "tercera clase" in text or "clase 3" in text or "clase tres" in text:
        return "Clase_03"
    if "cuarta clase" in text or "clase 4" in text:
        return "Clase_04"
    if "quinta clase" in text or "clase 5" in text:
        return "Clase_05"

    if "cronograma" in text:
        return "Cronograma"

    if "planificación" in text or "planificacion" in text:
        return "Planificacion"

    return "Documento"


def build_generation_prompt(req: ImproveRequest, context_body: str, has_context: bool) -> str:
    fuentes_texto = (
        "Usar únicamente el contexto ISDM recuperado para afirmaciones institucionales. "
        "Si falta evidencia puntual, indicarlo explícitamente."
        if has_context
        else "No hay contexto ISDM disponible. Generar una propuesta general y aclararlo explícitamente."
    )

    return f"""
Objetivo del encargo:
{req.objetivo}

Pedido del usuario:
{req.query}

Parámetros de generación:
- Nivel: {req.nivel}
- Formato de salida: {req.formato_salida}
- Tono: {req.tono}
- Materia: {req.materia or "No especificada"}
- Año: {req.anio or "No especificado"}
- Unidad: {req.unidad or "No especificada"}
- Unidad numérica: {req.unidad_num or "No especificada"}
- Tipo: {req.tipo or "No especificado"}
- Restricciones: {req.restricciones or "Ninguna"}

Reglas de respuesta:
- Responder en español.
- Entregar una salida directamente utilizable.
- Mantener estilo profesional, claro y operativo.
- No inventar respaldo institucional.
- {fuentes_texto}
- Si se usaron fuentes ISDM, cerrar con una sección breve llamada "Fuentes ISDM recuperadas:".
- Si no se encontraron o no se usaron fuentes, aclararlo explícitamente al final.

Contexto recuperado:
{context_body}
""".strip()


def save_output_document(output_text: str, req: ImproveRequest) -> Path:
    """
    Guarda la salida en:
    C:\\xampp\\htdocs\\ProfePilar\\data\\ISDM\\2026\\ISDM\\<MateriaNormalizada>\\
    Crea la carpeta de la materia una sola vez si no existe.
    Luego guarda allí todos los documentos.
    """

    materia_dir_name = normalize_materia_folder(req.materia)
    target_dir = OUTPUT_ROOT_DIR / materia_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = detect_file_prefix(req)

    unidad_part = sanitize_filename(req.unidad, "")
    tipo_part = sanitize_filename(req.tipo, "")
    formato_part = sanitize_filename(req.formato_salida, "")

    extras = [x for x in [unidad_part, tipo_part, formato_part] if x]
    base_filename = prefix
    if extras:
        base_filename += "_" + "_".join(extras)

    candidate_docx = target_dir / f"{base_filename}.docx"
    candidate_md = target_dir / f"{base_filename}.md"

    if candidate_docx.exists() or candidate_md.exists():
        suffix = 2
        while True:
            alt_docx = target_dir / f"{base_filename}_{suffix}.docx"
            alt_md = target_dir / f"{base_filename}_{suffix}.md"
            if not alt_docx.exists() and not alt_md.exists():
                candidate_docx = alt_docx
                candidate_md = alt_md
                break
            suffix += 1

    try:
        from docx import Document  # type: ignore

        document = Document()

        for block in output_text.split("\n\n"):
            clean_block = block.strip()
            if not clean_block:
                continue

            lines = clean_block.splitlines()
            first_line = lines[0].strip()

            if len(lines) == 1 and len(first_line) <= 120:
                document.add_heading(first_line, level=2)
            elif first_line.startswith("#"):
                heading_text = first_line.lstrip("#").strip()
                document.add_heading(heading_text or "Sección", level=2)
                for extra in lines[1:]:
                    if extra.strip():
                        document.add_paragraph(extra.strip())
            else:
                for line in lines:
                    if line.strip():
                        document.add_paragraph(line.strip())

        document.save(candidate_docx)
        return candidate_docx

    except Exception:
        candidate_md.write_text(output_text, encoding="utf-8")
        return candidate_md


# -------------------------
# SEARCH (PROTEGIDO)
# -------------------------


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, _: str = Depends(require_api_key)):
    filter_clauses = []

    if req.anio:
        filter_clauses.append({
            "type": "eq",
            "key": "anio",
            "value": req.anio
        })

    if req.materia:
        filter_clauses.append({
            "type": "eq",
            "key": "materia",
            "value": req.materia
        })

    if req.unidad:
        filter_clauses.append({
            "type": "eq",
            "key": "unidad",
            "value": req.unidad
        })

    if req.unidad_num:
        filter_clauses.append({
            "type": "eq",
            "key": "unidad_num",
            "value": req.unidad_num
        })

    if req.tipo:
        filter_clauses.append({
            "type": "eq",
            "key": "tipo",
            "value": req.tipo
        })

    filters = None
    if len(filter_clauses) == 1:
        filters = filter_clauses[0]
    elif len(filter_clauses) > 1:
        filters = {
            "type": "and",
            "filters": filter_clauses
        }

    try:
        resp = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=req.query,
            max_num_results=req.k,
            filters=filters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store search failed: {e}")

    out = []
    for item in resp.data:
        txt = ""

        content = getattr(item, "content", None)

        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if "text" in c and c["text"]:
                        parts.append(str(c["text"]))
                    elif "content" in c and c["content"]:
                        parts.append(str(c["content"]))
                else:
                    c_text = getattr(c, "text", None)
                    if c_text:
                        parts.append(str(c_text))
                    else:
                        c_content = getattr(c, "content", None)
                        if c_content:
                            parts.append(str(c_content))
            txt = "\n".join(parts).strip()

        elif isinstance(content, dict):
            txt = str(content.get("text") or content.get("content") or "").strip()

        elif content is not None:
            txt = str(content).strip()

        out.append(SearchResult(
            text=txt,
            score=float(getattr(item, "score", 0.0)),
            file_id=str(getattr(item, "file_id", "")),
            attributes=getattr(item, "attributes", None)
        ))

    return SearchResponse(vector_store_id=VECTOR_STORE_ID, results=out)


# -------------------------
# IMPROVE FOR 2026 (PROTEGIDO)
# -------------------------


@app.post("/improve_2026", response_model=ImproveResponse)
def improve_2026(req: ImproveRequest, _: str = Depends(require_api_key)):
    sources: list[dict] = []
    context = ""

    if req.modo_fuentes != "none":
        sreq = SearchRequest(
            query=req.query,
            k=req.k,
            anio=req.anio,
            materia=req.materia,
            unidad=req.unidad,
            unidad_num=req.unidad_num,
            tipo=req.tipo
        )
        sresp: SearchResponse = search(sreq, API_KEY)

        context_blocks = []
        for i, r in enumerate(sresp.results, start=1):
            meta = r.attributes or {}
            sources.append({
                "rank": i,
                "score": r.score,
                "source_name": meta.get("source_name"),
                "source_path": meta.get("source_path"),
                "anio": meta.get("anio"),
                "materia": meta.get("materia"),
                "unidad": meta.get("unidad"),
                "unidad_num": meta.get("unidad_num"),
                "tipo": meta.get("tipo"),
                "file_id": r.file_id,
            })
            context_blocks.append(
                f"[FUENTE {i}] "
                f"{meta.get('source_name', '(sin nombre)')} | "
                f"{meta.get('materia', '?')} {meta.get('anio', '?')} {meta.get('unidad', '')}\n"
                f"{r.text}\n"
            )

        context = "\n\n".join(context_blocks)

        if req.modo_fuentes == "required" and len(sresp.results) == 0:
            return ImproveResponse(
                used_sources=[],
                output="No encontré fuentes ISDM con esos filtros."
            )

    has_context = bool(context.strip())

    if has_context:
        rule = (
            "REGLA: Usá el contexto recuperado como base para cualquier afirmación institucional. "
            "No inventes normativa, documentos ni atribuciones al ISDM."
        )
        context_body = context
    else:
        rule = (
            "REGLA: No hay contexto ISDM recuperado. Generá una propuesta general, "
            "aclarando explícitamente que fue elaborada sin respaldo ISDM encontrado."
        )
        context_body = "[SIN CONTEXTO ISDM]"

    prompt = build_generation_prompt(
        req=req,
        context_body=context_body,
        has_context=has_context
    )

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": rule},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI generation failed: {e}")

    output_text = getattr(resp, "output_text", None) or str(resp)

    # Guardado automático sin romper compatibilidad del response model
    try:
        save_output_document(output_text=output_text, req=req)
    except Exception:
        # Si falla el guardado, no romper la respuesta del endpoint
        pass

    return ImproveResponse(
        used_sources=sources,
        output=output_text
    )