# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

app = FastAPI(title="ISDM RAG API", version="0.3.0")

# -------------------------
# CONFIG / CLIENT
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Mejor fallar con mensaje claro que con KeyError
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
# BASIC ENDPOINTS
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
    # qué querés lograr
    objetivo: str = Field(..., min_length=5)

    # consulta para recuperar contexto (si modo_fuentes != "none")
    query: str = Field(..., min_length=2)

    # filtros
    anio: str | None = None
    materia: str | None = None
    unidad: str | None = None
    k: int = Field(10, ge=3, le=20)

    # comportamiento respecto a fuentes
    # - required: si no hay fuentes, NO inventa (devuelve mensaje pidiendo material)
    # - optional: usa fuentes si hay; si no, propone igual pero avisa que es general
    # - none: no busca; propone directamente (modo remoto)
    modo_fuentes: str = Field(
        "optional",
        pattern="^(required|optional|none)$"
    )

    # parámetros docentes
    nivel: str = "Superior ISDM"
    formato_salida: str = "plan_clase"  # plan_clase | cronograma | guia_practica | rubrica | mejoras_documento
    tono: str = "profesional_cercano"
    restricciones: str | None = None  # ej: "sin PP1", "solo cuatrimestre 1", etc.


class ImproveResponse(BaseModel):
    used_sources: list[dict]
    output: str


# -------------------------
# SEARCH
# -------------------------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
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
# IMPROVE FOR 2026
# -------------------------
@app.post("/improve_2026", response_model=ImproveResponse)
def improve_2026(req: ImproveRequest):
    sources: list[dict] = []
    context = ""

    # 1) Buscar material relevante (según modo)
    if req.modo_fuentes != "none":
        sreq = SearchRequest(
            query=req.query,
            k=req.k,
            anio=req.anio,
            materia=req.materia,
            unidad=req.unidad
        )
        sresp: SearchResponse = search(sreq)

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

        # si exige fuentes y no hay
        if req.modo_fuentes == "required" and len(sresp.results) == 0:
            return ImproveResponse(
                used_sources=[],
                output=(
                    "No encontré fuentes ISDM en el Vector Store con esos filtros.\n"
                    "✅ Podés:\n"
                    "- aflojar filtros (anio/materia/unidad)\n"
                    "- cambiar la query\n"
                    "- o reindexar material 2025 relacionado\n"
                    "Si querés igualmente una propuesta general, pedime 'modo remoto' o usa modo_fuentes='optional'/'none'."
                )
            )

    # 2) Instrucciones según haya contexto o no
    if context.strip():
        rule = (
            "REGLA: basate únicamente en el CONTEXTO provisto (fragmentos recuperados) "
            "y citá/respaldá con lo recuperado. Si falta info, decilo y pedí qué documento haría falta."
        )
        context_header = "CONTEXTO (material 2025 recuperado):"
        context_body = context
    else:
        # modo remoto o no encontró fuentes
        rule = (
            "REGLA: NO hay contexto ISDM provisto. Generá una propuesta general y práctica, "
            "pero ACLARÁ explícitamente que no se encontraron fuentes ISDM para respaldarla."
        )
        context_header = "CONTEXTO:"
        context_body = "[SIN CONTEXTO ISDM RECUPERADO]"

    system_instructions = f"""
Sos un asistente pedagógico-profesional para nivel {req.nivel}.
{rule}

Objetivo del usuario: {req.objetivo}
Formato de salida: {req.formato_salida}
Tono: {req.tono}
Restricciones: {req.restricciones or "ninguna"}

Entregá propuestas concretas, mejoradas, reutilizables para 2026.
Incluí: mejoras, reestructuración, y un borrador listo para copiar/pegar.
"""

    user_prompt = f"""
MODO_FUENTES: {req.modo_fuentes}
CANTIDAD_FUENTES: {len(sources)}

{context_header}
{context_body}

TAREA:
1) Detectá problemas/mejoras (claridad, secuenciación, evaluación, actividades, tiempos).
2) Proponé una versión 2026 mejorada.
3) Si el formato_salida es plan_clase: incluir objetivos, inicio-desarrollo-cierre, actividad práctica, evidencias, evaluación/rúbrica breve, materiales.
4) Si cronograma: tabla por clase/semana con entregables.
5) Si guia_practica: consigna + pasos + criterios de logro + checklist.
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI generation failed: {e}")

    output_text = getattr(resp, "output_text", None) or str(resp)

    return ImproveResponse(
        used_sources=sources,
        output=output_text
    )