"""
Module de vectorisation des événements Open Agenda.

Construit un index FAISS à partir des événements collectés, en utilisant
mistral-embed pour générer les vecteurs sémantiques.
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_events(path: Path) -> list[dict]:
    """Charge la liste d'événements depuis un fichier JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_text(event: dict) -> str:
    """Construit le texte concaténé qui sera vectorisé pour un événement."""
    location = event.get("location") or {}
    keywords = event.get("keywords") or []

    parts = [
        f"Titre: {event.get('title', '')}",
        f"Description: {event.get('description', '')}",
        f"Détails: {event.get('long_description', '')}",
        f"Mots-clés: {', '.join(keywords)}" if keywords else "",
        f"Lieu: {location.get('name', '')}, {location.get('city', '')}, {location.get('department', '')}",
        f"Date: {event.get('date_range', '')}",
    ]

    # Filtre les lignes vides ou réduites à un préfixe sans valeur
    return "\n".join(p for p in parts if p and not p.endswith(": "))


def build_metadata(event: dict) -> dict:
    """Extrait les métadonnées à conserver pour chaque événement indexé."""
    location = event.get("location") or {}

    return {
        "uid": event.get("uid"),
        "title": event.get("title", ""),
        "description": event.get("description", ""),
        "city": location.get("city", ""),
        "department": location.get("department", ""),
        "location_name": location.get("name", ""),
        "date_range": event.get("date_range", ""),
        "next_timing": event.get("next_timing", ""),
        "image_url": event.get("image_url", ""),
        "attendance_mode": event.get("attendance_mode", ""),
    }


def build_index(events: list[dict], output_dir: Path) -> FAISS:
    """Vectorise les événements et construit l'index FAISS persistant."""
    # Construction des Documents LangChain
    documents = [
        Document(
            page_content=build_text(event),
            metadata=build_metadata(event),
        )
        for event in events
    ]

    # Initialisation du modèle d'embedding
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    # Vectorisation + indexation
    print(f"Vectorisation de {len(documents)} événements via mistral-embed...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Persistance sur disque
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    print(f"Index sauvegardé dans : {output_dir}")

    return vectorstore