"""Collecte des événements OpenAgenda (Île-de-France).

Logique métier importable : récupération via l'API, nettoyage, et sauvegarde
d'un export JSON daté dans data/raw/. Utilisée par le CLI
(scripts/collect_events.py) et par l'endpoint /rebuild de l'API.

Le chargement de .env est laissé aux points d'entrée, pas au module.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime

import requests

AGENDA_UID = 56500817  # Agenda OpenAgenda Île-de-France
API_BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"
PAGE_SIZE = 100          
REQUEST_DELAY = 0.3      # pause entre les pages
RAW_DIR = Path("data/raw")


def fetch_all_events() -> list[dict]:
    """Récupère tous les événements en cours et à venir via pagination."""
    api_key = os.getenv("OPENAGENDA_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAGENDA_API_KEY introuvable dans l'environnement")

    all_events = []
    after_cursor = None
    page_num = 1

    while True:
        params = {
            "relative[]": ["current", "upcoming"],
            "size": PAGE_SIZE,
            "monolingual": "fr",
            "detailed": 1,
        }
        if after_cursor:
            params["after[]"] = after_cursor

        print(f"Page {page_num}: requête en cours...", end=" ")
        response = requests.get(API_BASE_URL, headers={"key": api_key}, params=params)

        if response.status_code != 200:
            print(f"Erreur HTTP {response.status_code}")
            break

        data = response.json()
        events_page = data.get("events", [])
        all_events.extend(events_page)
        print(f"{len(events_page)} événements (total: {len(all_events)})")

        after_cursor = data.get("after")
        if not after_cursor or len(events_page) < PAGE_SIZE:
            break

        page_num += 1
        time.sleep(REQUEST_DELAY)

    return all_events


def _extract_image_url(image_data: dict | None) -> str | None:
    """Extrait l'URL complète de l'image principale, si disponible."""
    if not image_data:
        return None
    base_url = image_data.get("base", "")
    for variant in image_data.get("variants", []):
        if variant.get("type") == "full":
            return base_url + variant.get("filename", "")
    filename = image_data.get("filename", "")
    return base_url + filename if filename else None


def transform_event(raw_event: dict) -> dict | None:
    """Transforme un événement brut en structure propre. None si inexploitable."""
    title = raw_event.get("title")
    if not title:
        return None

    location = raw_event.get("location") or {}
    return {
        "uid": raw_event.get("uid"),
        "title": title,
        "description": raw_event.get("description"),
        "long_description": raw_event.get("longDescription"),
        "keywords": raw_event.get("keywords") or [],
        "conditions": raw_event.get("conditions"),
        "date_range": raw_event.get("dateRange"),
        "first_timing": (raw_event.get("firstTiming") or {}).get("begin"),
        "last_timing": (raw_event.get("lastTiming") or {}).get("end"),
        "next_timing": (raw_event.get("nextTiming") or {}).get("begin"),
        "attendance_mode": raw_event.get("attendanceMode"),
        "online_access_link": raw_event.get("onlineAccessLink"),
        "image_url": _extract_image_url(raw_event.get("image")),
        "location": {
            "name": location.get("name"),
            "address": location.get("address"),
            "city": location.get("city"),
            "postal_code": location.get("postalCode"),
            "department": location.get("adminLevel2"),
            "region": location.get("adminLevel1"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
        },
        "origin_agenda": (raw_event.get("originAgenda") or {}).get("title"),
    }


def save_events_to_json(events: list[dict], output_dir: Path = RAW_DIR) -> Path:
    """Sauvegarde les événements dans un JSON daté. Retourne son chemin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"events_idf_{timestamp}.json"
    file_path.write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return file_path


def collect_events() -> Path:
    """Orchestre la collecte : récupération, nettoyage, sauvegarde.

    Retourne le chemin du fichier JSON daté créé. Point d'entrée réutilisable,
    appelé par le CLI et par /rebuild.
    """
    raw_events = fetch_all_events()
    transformed = (transform_event(r) for r in raw_events)
    clean_events = [e for e in transformed if e is not None]

    if not clean_events:
        raise ValueError("Aucun événement valide collecté")

    return save_events_to_json(clean_events)