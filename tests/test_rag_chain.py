"""Tests unitaires de la chaîne RAG.

Portent sur `format_docs`, fonction pure : aucun appel API, aucun accès
à l'index FAISS. Rapides et déterministes.
"""
import pytest
from langchain_core.documents import Document

from src.rag_chain import format_docs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_docs() -> list[Document]:
    """Documents de test avec metadata représentatives d'événements Puls-Events."""
    return [
        Document(
            page_content="texte vectorisé brut, non utilisé par format_docs",
            metadata={
                "uid": 123,
                "title": "Festival de jazz de la Défense",
                "description": "Trois jours de concerts en plein air",
                "location_name": "Parc de la Défense",
                "city": "Puteaux",
                "department": "Hauts-de-Seine",
                "date_range": "du 15 au 17 juin 2026",
            },
        ),
        Document(
            page_content="texte vectorisé brut, non utilisé par format_docs",
            metadata={
                "uid": 456,
                "title": "Exposition Monet à l'Orangerie",
                "description": "Rétrospective sur les Nymphéas",
                "location_name": "Musée de l'Orangerie",
                "city": "Paris",
                "department": "Paris",
                "date_range": "du 1er au 31 juillet 2026",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tests unitaires - format_docs
# ---------------------------------------------------------------------------
def test_format_docs_returns_string(sample_docs):
    """format_docs doit retourner une string."""
    assert isinstance(format_docs(sample_docs), str)


def test_format_docs_numbers_events(sample_docs):
    """Chaque événement doit être préfixé par [Événement N]."""
    output = format_docs(sample_docs)
    assert "[Événement 1]" in output
    assert "[Événement 2]" in output


def test_format_docs_includes_metadata(sample_docs):
    """Les champs structurés metadata doivent apparaître dans la sortie."""
    output = format_docs(sample_docs)
    assert "Festival de jazz de la Défense" in output
    assert "Parc de la Défense" in output
    assert "Puteaux" in output
    assert "du 15 au 17 juin 2026" in output


def test_format_docs_handles_empty_list():
    """Une liste vide doit retourner une string vide."""
    assert format_docs([]) == ""


def test_format_docs_handles_missing_metadata():
    """Les champs metadata manquants ne doivent pas faire crasher."""
    doc = Document(page_content="", metadata={"title": "Titre uniquement"})
    output = format_docs([doc])
    assert "[Événement 1]" in output
    assert "Titre uniquement" in output
