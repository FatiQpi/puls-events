"""Tests de la chaîne RAG.

Deux catégories de tests :

- Tests unitaires purs (sans appel API, ni accès à l'index FAISS) :
  format_docs. Lancés par défaut avec `pytest`.

- Tests d'intégration (appel API Mistral + lecture de l'index sur disque) :
  marqués @pytest.mark.integration, skippés par défaut.
  Lancement explicite : `pytest -m integration`
"""
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from src.rag_chain import build_chain, format_docs


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


# ---------------------------------------------------------------------------
# Tests d'intégration - chaîne RAG complète
# Skippés par défaut, nécessitent MISTRAL_API_KEY + index FAISS sur disque
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_build_chain_returns_runnable():
    """build_chain doit retourner un objet Runnable.

    Vérifie également que l'index FAISS est chargeable depuis data/index.
    """
    chain = build_chain()
    assert isinstance(chain, Runnable)


@pytest.mark.integration
def test_chain_invoke_returns_dict_with_answer_and_docs():
    """L'invocation doit retourner un dict {answer, docs} non vides."""
    chain = build_chain()
    result = chain.invoke("concert de jazz à Paris")

    assert isinstance(result, dict)
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert isinstance(result["docs"], list)
    assert len(result["docs"]) > 0