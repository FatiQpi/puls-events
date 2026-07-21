"""Tests fonctionnels de l'API Puls-Events.

Exerce les endpoints via le TestClient de FastAPI (en mémoire, sans serveur à
lancer séparément). Le client est instancié en gestionnaire de contexte (`with`)
afin de déclencher le lifespan : l'index FAISS et la chaîne RAG sont chargés une
seule fois pour toute la session de test.

Ces tests n'appellent pas le LLM (health, structure de metadata, validation des
entrées de /ask). Ils nécessitent toutefois l'index sur disque (data/index) et la
clé Mistral dans .env, puisque le lifespan charge la chaîne au démarrage.
"""
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    # Le bloc `with` déclenche le lifespan (chargement de l'index + de la chaîne).
    with TestClient(app) as test_client:
        yield test_client


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