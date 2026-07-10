"""API REST Puls-Events : expose la chaîne RAG via FastAPI.

Endpoints :
  - GET  /health    : vérifie que le serveur répond
  - GET  /metadata  : informations sur le corpus et la configuration
  - POST /ask       : pose une question, renvoie {answer, sources}
  - POST /rebuild   : re-collecte + re-vectorise + recharge la chaîne

Lancement local : python -m scripts.run_api
Documentation interactive (Swagger) : http://127.0.0.1:8000/docs
"""
import os

# Doit être posé avant tout import déclenchant FAISS (conflit libomp sur macOS).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.collect import collect_events
from src.vectorize import build_index, load_events
from src.rag_chain import (
    DEFAULT_INDEX_DIR,
    DEFAULT_K,
    EMBEDDING_MODEL,
    LLM_MODEL,
    build_chain,
    load_vectorstore,
)

SOURCE_LABEL = "Open Agenda — agrégateur Île-de-France (uid 56500817)"
GEO_SCOPE = "Île-de-France"


# ---------------------------------------------------------------------------
# Modèles d'entrée / sortie (validés et documentés automatiquement par FastAPI)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question en langage naturel (ex. : concerts de jazz à Paris ce week-end).",
    )

    @field_validator("question")
    @classmethod
    def question_non_vide(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La question ne peut pas être vide.")
        return v


class Source(BaseModel):
    uid: int | None = None
    title: str = ""
    location_name: str = ""
    city: str = ""
    date_range: str = ""
    image_url: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _index_built_at() -> str | None:
    """Date de dernière écriture de l'index servi (mtime du fichier FAISS)."""
    index_file = DEFAULT_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None
    return datetime.fromtimestamp(index_file.stat().st_mtime).isoformat(timespec="seconds")


def _docs_to_sources(docs) -> list[dict]:
    """Réduit les Documents récupérés aux champs utiles à un consommateur externe."""
    return [
        {
            "uid": doc.metadata.get("uid"),
            "title": doc.metadata.get("title") or "",
            "location_name": doc.metadata.get("location_name") or "",
            "city": doc.metadata.get("city") or "",
            "date_range": doc.metadata.get("date_range") or "",
            "image_url": doc.metadata.get("image_url") or "",
        }
        for doc in docs
    ]


def _load_state(app: FastAPI) -> None:
    """Charge l'index, construit la chaîne et mémorise les métadonnées du corpus.

    Appelée une fois au démarrage (lifespan) et après chaque /rebuild réussi.
    """
    vectorstore = load_vectorstore()
    app.state.chain = build_chain()
    app.state.event_count = int(vectorstore.index.ntotal)
    app.state.index_built_at = _index_built_at()


# ---------------------------------------------------------------------------
# Cycle de vie : on charge l'index et la chaîne UNE SEULE FOIS au démarrage,
# puis chaque requête réutilise l'objet en mémoire (app.state).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    _load_state(app)
    yield


app = FastAPI(
    title="Puls-Events API",
    description=(
        "API REST de recommandation d'événements culturels d'Île-de-France, "
        "appuyée sur un système RAG (Mistral + FAISS)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Vérifie que le serveur répond")
def health():
    """Liveness check basique, sans accès à l'index."""
    return {"status": "ok"}


@app.get("/metadata", summary="Informations sur le corpus et la configuration")
def metadata(request: Request):
    """Renvoie l'état du corpus servi et la configuration du système RAG."""
    return {
        "corpus": {
            "event_count": request.app.state.event_count,
            "source": SOURCE_LABEL,
            "geographic_scope": GEO_SCOPE,
            "index_built_at": request.app.state.index_built_at,
        },
        "config": {
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_k": DEFAULT_K,
        },
    }


@app.post("/ask", response_model=AskResponse, summary="Pose une question au système RAG")
def ask(payload: AskRequest, request: Request):
    """Récupère les événements pertinents et génère une réponse augmentée.

    Renvoie la réponse générée et la liste des événements sources. Si aucun
    événement n'est pertinent, la réponse l'indique et `sources` peut être vide
    (ce n'est pas une erreur : code 200).
    """
    try:
        result = request.app.state.chain.invoke(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la génération de la réponse : {exc}"
        )

    return {
        "answer": result["answer"],
        "sources": _docs_to_sources(result["docs"]),
    }


@app.post("/rebuild", summary="Reconstruit la base vectorielle à la demande")
def rebuild(request: Request):
    """Re-collecte les événements, re-vectorise, et recharge la chaîne en mémoire.

    Opération synchrone : la requête reste ouverte le temps de la collecte et de
    l'embedding. En cas d'échec, la chaîne en mémoire n'est pas modifiée et
    l'index précédent continue d'être servi.
    """
    start = time.perf_counter()
    try:
        path = collect_events()           # nouveau JSON daté dans data/raw/
        events = load_events(path)
        build_index(events, DEFAULT_INDEX_DIR)   # save_local en dernier (sûr)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Échec du rebuild : {exc}")

    # Hot-swap : sans ça, /ask continuerait de servir l'ancien index en mémoire.
    _load_state(request.app)

    return {
        "status": "rebuilt",
        "event_count": request.app.state.event_count,
        "index_built_at": request.app.state.index_built_at,
        "duration_seconds": round(time.perf_counter() - start, 1),
    }