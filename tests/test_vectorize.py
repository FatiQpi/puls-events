"""
Tests unitaires pour le module src.vectorize.
Version minimale couvrant les fonctions pures du module.
"""

import json
import pytest

from src.vectorize import load_events, build_text, build_metadata


@pytest.fixture
def event_complet():
    """Un événement avec tous les champs renseignés."""
    return {
        "uid": 12345,
        "title": "Festival de Jazz",
        "description": "Concert en plein air",
        "long_description": "Une soirée musicale exceptionnelle",
        "keywords": ["jazz", "concert", "gratuit"],
        "location": {
            "name": "Parc Montsouris",
            "city": "Paris",
            "department": "Paris",
        },
        "date_range": "15-17 août 2026",
        "next_timing": "2026-08-15T19:00:00",
        "image_url": "https://oa.com/img.jpg",
        "attendance_mode": "offline",
    }


def test_build_text_cas_nominal(event_complet):
    """Tous les champs présents → texte structuré complet."""
    text = build_text(event_complet)

    assert "Titre: Festival de Jazz" in text
    assert "Description: Concert en plein air" in text
    assert "Mots-clés: jazz, concert, gratuit" in text
    assert "Lieu: Parc Montsouris, Paris, Paris" in text
    assert "Date: 15-17 août 2026" in text


def test_build_text_robuste_aux_champs_manquants():
    """La fonction ne crashe pas avec un événement minimal."""
    event_minimal = {
        "title": "Visite simple",
        "date_range": "demain",
    }
    text = build_text(event_minimal)

    assert "Titre: Visite simple" in text
    assert "Description:" not in text
    assert "Mots-clés:" not in text


def test_build_metadata_cas_nominal(event_complet):
    """Toutes les clés attendues sont présentes avec les bonnes valeurs."""
    meta = build_metadata(event_complet)

    assert meta["uid"] == 12345
    assert meta["title"] == "Festival de Jazz"
    assert meta["city"] == "Paris"
    assert meta["location_name"] == "Parc Montsouris"
    assert meta["date_range"] == "15-17 août 2026"


def test_load_events(tmp_path):
    """Charge un fichier JSON et retourne une liste de dicts."""
    fake_events = [
        {"uid": 1, "title": "Event 1"},
        {"uid": 2, "title": "Event 2"},
    ]
    json_path = tmp_path / "events.json"
    json_path.write_text(json.dumps(fake_events), encoding="utf-8")

    loaded = load_events(json_path)

    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert loaded[0]["uid"] == 1