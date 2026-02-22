# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

app = FastAPI(title="ISDM RAG API", version="0.3.0")

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
    anio: str | None = None
    materia: str | None = None
    unidad: str | None = None


class SearchResult(BaseModel):
    text: str
    score: float
    file_id: str
    attributes: dict | None = None


class SearchResponse(BaseModel):
    vector_store_id: str
    results: list[SearchResult]


class ImproveRequest(BaseModel):
    objetivo: str = Field(..., min_length=5)
    query: str = Field(..., min_length=2)
    anio: str | None = None
    materia: str | None = None
    unidad: str | None = None
    k: int = Field(10, ge=3, le=20)

    modo_fuentes: str = Field(
        "optional",
        pattern="^(required|optional|none)$"
    )

    nivel: str = "Superior ISDM"
    formato_salida: str = "plan_clase"
    tono: str = "profesional_cercano"
    restricciones: str | None = None


class ImproveResponse(BaseModel):
    used_sources: list[dict]
    output: str


# -------------------------
# SEARCH (PROTEGIDO)
# -------------------------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, _: str = Depends(require_api_key)):
    filters = {}
    if req.anio:
        filters["anio"] = req.anio
    if req.materia:
        filters["materia"] = req.materia
    if req.unidad:
        filters["unidad"] = req.unidad

    try:
        resp = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=req.query,
            max_num_results=req.k,
            filters=filters if filters else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store search failed: {e}")

    out = []
    for item in resp.data:
        content = item.content
        if isinstance(content, list) and content:
            txt = "\n".join([c.get("text", "") for c in content if isinstance(c, dict)])
        else:
            txt = str(content)

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
            unidad=req.unidad
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
                "file_id": r.file_id,
            })
            context_blocks.append(
                f"[FUENTE {i}] "
                f"{meta.get('source_name','(sin nombre)')} | "
                f"{meta.get('materia','?')} {meta.get('anio','?')} {meta.get('unidad','')}\n"
                f"{r.text}\n"
            )

        context = "\n\n".join(context_blocks)

        if req.modo_fuentes == "required" and len(sresp.results) == 0:
            return ImproveResponse(
                used_sources=[],
                output="No encontré fuentes ISDM con esos filtros."
            )

    if context.strip():
        rule = (
            "REGLA: basate únicamente en el CONTEXTO provisto "
            "y citá lo recuperado."
        )
        context_body = context
    else:
        rule = (
            "REGLA: NO hay contexto ISDM. Generá propuesta general "
            "y aclaralo explícitamente."
        )
        context_body = "[SIN CONTEXTO ISDM]"

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": rule},
                {"role": "user", "content": context_body},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI generation failed: {e}")

    output_text = getattr(resp, "output_text", None) or str(resp)

    return ImproveResponse(
        used_sources=sources,
        output=output_text
    )