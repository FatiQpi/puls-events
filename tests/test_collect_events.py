"""Tests unitaires pour le script de collecte d'événements."""

import sys
from pathlib import Path

# Permettre l'import depuis scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from collect_events import transform_event, _extract_image_url


# --- Tests pour transform_event ---

def test_transform_event_valid():
    """Un événement valide est correctement transformé."""
    raw = {
        "uid": 123,
        "title": "Concert de jazz",
        "description": "Un super concert",
        "longDescription": "Description longue",
        "keywords": ["jazz", "musique"],
        "firstTiming": {"begin": "2026-05-15T20:00:00+02:00"},
        "location": {
            "name": "Le Sunset",
            "city": "Paris",
            "adminLevel1": "Île-de-France",
            "adminLevel2": "Paris",
        },
    }
    result = transform_event(raw)

    assert result is not None
    assert result["uid"] == 123
    assert result["title"] == "Concert de jazz"
    assert result["keywords"] == ["jazz", "musique"]
    assert result["first_timing"] == "2026-05-15T20:00:00+02:00"
    assert result["location"]["city"] == "Paris"
    assert result["location"]["region"] == "Île-de-France"


def test_transform_event_without_title_returns_none():
    """Un événement sans titre renvoie None."""
    raw = {"uid": 456, "description": "Sans titre"}
    assert transform_event(raw) is None


def test_transform_event_with_none_keywords():
    """Quand keywords est None, on renvoie une liste vide."""
    raw = {
        "uid": 789,
        "title": "Test",
        "keywords": None,
        "location": {},
    }
    result = transform_event(raw)
    assert result["keywords"] == []


# --- Tests pour _extract_image_url ---

def test_extract_image_url_with_full_variant():
    """Extrait l'URL de la variante 'full' en priorité."""
    image = {
        "base": "https://oa.example.com/",
        "filename": "img.base.jpg",
        "variants": [
            {"filename": "img.full.jpg", "type": "full"},
            {"filename": "img.thumb.jpg", "type": "thumbnail"},
        ],
    }
    assert _extract_image_url(image) == "https://oa.example.com/img.full.jpg"


def test_extract_image_url_fallback_to_base():
    """Si pas de variante 'full', utilise le filename de base."""
    image = {
        "base": "https://oa.example.com/",
        "filename": "img.base.jpg",
        "variants": [
            {"filename": "img.thumb.jpg", "type": "thumbnail"},
        ],
    }
    assert _extract_image_url(image) == "https://oa.example.com/img.base.jpg"


def test_extract_image_url_returns_none_when_empty():
    """Renvoie None si pas d'image."""
    assert _extract_image_url(None) is None
    assert _extract_image_url({}) is None