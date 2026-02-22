# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

app = FastAPI(title="ISDM RAG API", version="0.2.0")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
    # contexto de consulta
    query: str = Field(..., min_length=2)
    anio: str | None = None
    materia: str | None = None
    unidad: str | None = None
    k: int = Field(10, ge=3, le=20)

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

    resp = client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=req.query,
        max_num_results=req.k,
        filters=filters if filters else None
    )

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
    # 1) buscar material relevante
    sreq = SearchRequest(
        query=req.query,
        k=req.k,
        anio=req.anio,
        materia=req.materia,
        unidad=req.unidad
    )
    sresp: SearchResponse = search(sreq)

    # 2) armar contexto SOLO con lo recuperado
    sources = []
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
        })
        context_blocks.append(
            f"[FUENTE {i}] "
            f"{meta.get('source_name','(sin nombre)')} | "
            f"{meta.get('materia','?')} {meta.get('anio','?')} {meta.get('unidad','')}\n"
            f"{r.text}\n"
        )

    context = "\n\n".join(context_blocks)

    # 3) pedir al modelo que PROPONGA mejoras basadas SOLO en ese contexto
    # Nota: usamos responses API para generación; podés cambiar modelo si querés.
    system_instructions = f"""
Sos un asistente pedagógico-profesional para nivel {req.nivel}.
REGLA: basate únicamente en el CONTEXTO provisto (fragmentos recuperados). 
Si falta información, decilo y proponé qué documento haría falta.
Objetivo del usuario: {req.objetivo}
Formato de salida: {req.formato_salida}
Tono: {req.tono}
Restricciones: {req.restricciones or "ninguna"}
Entregá propuestas concretas, mejoradas, reutilizables para 2026.
Incluí: mejoras, reestructuración, y un borrador listo para copiar/pegar.
"""

    user_prompt = f"""
CONTEXTO (material 2025 recuperado):
{context}

TAREA:
1) Detectá problemas/mejoras (claridad, secuenciación, evaluación, actividades, tiempos).
2) Proponé una versión 2026 mejorada.
3) Si el formato_salida es plan_clase: incluir objetivos, inicio-desarrollo-cierre, actividad práctica, evidencias, evaluación/rúbrica breve, materiales.
4) Si cronograma: tabla por clase/semana con entregables.
5) Si guia_practica: consigna + pasos + criterios de logro + checklist.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ],
    )

    # la salida viene en resp.output_text en SDK nuevo; fallback por si cambia
    output_text = getattr(resp, "output_text", None) or str(resp)

    return ImproveResponse(
        used_sources=sources,
        output=output_text
    )