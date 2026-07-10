"""Tests fonctionnels de l'API Puls-Events.

Exerce les endpoints via le TestClient de FastAPI (en mémoire, sans serveur à
lancer séparément). Le client est instancié en gestionnaire de contexte (`with`)
afin de déclencher le lifespan : l'index FAISS et la chaîne RAG sont chargés une
seule fois pour toute la session de test.

Deux niveaux :

- Tests par défaut : n'appellent PAS le LLM (health, structure de metadata,
  validation des entrées de /ask). Nécessitent tout de même l'index sur disque
  (data/index) et la clé Mistral dans .env, puisque le lifespan charge la chaîne.

- Tests d'intégration (@pytest.mark.integration, désélectionnés par défaut) :
  appellent réellement Mistral et/ou reconstruisent l'index.
  Lancement explicite : pytest -m integration
"""
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    # Le bloc `with` déclenche le lifespan (chargement de l'index + de la chaîne).
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Tests par défaut (sans appel au LLM)
# ---------------------------------------------------------------------------
def test_health(client):
    """/health répond 200 avec un statut ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_structure(client):
    """/metadata répond 200 avec les blocs corpus et config attendus."""
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()

    assert "corpus" in body
    assert "config" in body
    assert isinstance(body["corpus"]["event_count"], int)
    assert body["corpus"]["event_count"] > 0
    assert body["config"]["retrieval_k"] == 5


def test_ask_rejects_empty_question(client):
    """Une question vide est rejetée (422)."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_blank_question(client):
    """Une question réduite à des espaces est rejetée (422)."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_rejects_missing_field(client):
    """Une requête sans champ 'question' est rejetée (422)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests d'intégration (appellent Mistral / reconstruisent l'index)
# Désélectionnés par défaut. Lancement : pytest -m integration
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_ask_returns_answer_and_sources(client):
    """/ask répond 200 avec une réponse non vide et une liste de sources."""
    response = client.post("/ask", json={"question": "concert de jazz à Paris"})
    assert response.status_code == 200
    body = response.json()

    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert isinstance(body["sources"], list)


@pytest.mark.integration
def test_rebuild_reconstructs_index(client):
    """/rebuild répond 200 et renvoie un corpus reconstruit.

    ATTENTION : déclenche une collecte + un embedding COMPLETS (~1 min) et
    ÉCRASE l'index sur disque. Consomme du quota Mistral. À lancer
    délibérément, jamais dans un run de test routinier.
    """
    response = client.post("/rebuild")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "rebuilt"
    assert isinstance(body["event_count"], int)
    assert body["event_count"] > 0