"""
Script de collecte des événements de l'agenda OpenAgenda en Île-de-France.

Récupère tous les événements en cours et à venir, les nettoie,
et les sauvegarde dans un fichier JSON pour la suite du pipeline RAG.
"""

import os
import time
import requests
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("OPENAGENDA_API_KEY")

if not API_KEY:
    raise EnvironmentError("OPENAGENDA_API_KEY introuvable dans .env")

AGENDA_UID = 56500817  # OpenAgenda en Île-de-France
API_BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"
PAGE_SIZE = 100  # max autorisé pour les événements
REQUEST_DELAY = 0.3  


def fetch_all_events() -> list[dict]:
    """
    Récupère tous les événements en cours et à venir via pagination.
    Retourne la liste brute des événements.
    """
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

        headers = {"key": API_KEY}

        print(f"Page {page_num}: requête en cours...", end=" ")
        response = requests.get(API_BASE_URL, headers=headers, params=params)

        if response.status_code != 200:
            print(f" Erreur HTTP {response.status_code}")
            break

        data = response.json()
        events_page = data.get("events", [])
        all_events.extend(events_page)

        print(f"{len(events_page)} événements (total: {len(all_events)})")

        after_cursor = data.get("after")
        if not after_cursor or len(events_page) < PAGE_SIZE:
            print("Fin de la pagination.")
            break

        page_num += 1
        time.sleep(REQUEST_DELAY)

    return all_events

def transform_event(raw_event: dict) -> dict | None:
    """
    Transforme un événement brut de l'API en structure propre.
    Retourne None si l'événement n'est pas exploitable.
    """
    # Exclure les événements sans titre
    title = raw_event.get("title")
    if not title:
        return None

    # Récupération sécurisée de la location 
    raw_location = raw_event.get("location") or {}

    # Structure propre
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
            "name": raw_location.get("name"),
            "address": raw_location.get("address"),
            "city": raw_location.get("city"),
            "postal_code": raw_location.get("postalCode"),
            "department": raw_location.get("adminLevel2"),
            "region": raw_location.get("adminLevel1"),
            "latitude": raw_location.get("latitude"),
            "longitude": raw_location.get("longitude"),
        },
        "origin_agenda": (raw_event.get("originAgenda") or {}).get("title"),
    }


def _extract_image_url(image_data: dict | None) -> str | None:
    """Extrait l'URL complète de l'image principale."""
    if not image_data:
        return None

    base_url = image_data.get("base", "")
    variants = image_data.get("variants", [])

    # Chercher la variante full
    for variant in variants:
        if variant.get("type") == "full":
            return base_url + variant.get("filename", "")

    # Utiliser le filename de base
    filename = image_data.get("filename", "")
    if filename:
        return base_url + filename

    return None

def save_events_to_json(events: list[dict], output_dir: str = "data/raw") -> Path:
    """
    Sauvegarde la liste des événements dans un fichier JSON daté.
    Retourne le chemin du fichier créé.
    """
    # Créer le dossier de destination s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Nom de fichier daté pour traçabilité
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"events_idf_{timestamp}.json"
    file_path = output_path / filename

    # Écriture du JSON
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    return file_path


def main():
    print(f"Collecte des événements de l'agenda {AGENDA_UID}...\n")

    raw_events = fetch_all_events()
    print(f"\n✓ {len(raw_events)} événements bruts collectés")

    # Diagnostic : qualité des champs essentiels
    events_without_title = [e for e in raw_events if not e.get("title")]
    events_without_description = [e for e in raw_events if not e.get("description")]
    events_without_long_description = [e for e in raw_events if not e.get("longDescription")]
    events_without_location = [e for e in raw_events if not e.get("location")]

    print(f"\n--- Diagnostic qualité ---")
    print(f"Événements sans 'title' : {len(events_without_title)}")
    print(f"Événements sans 'description' : {len(events_without_description)}")
    print(f"Événements sans 'longDescription' : {len(events_without_long_description)}")
    print(f"Événements sans 'location' : {len(events_without_location)}")

    # Transformation des événements vers la structure cible
    print("\n Transformation des événements...")
    transformed = [transform_event(e) for e in raw_events]
    clean_events = [e for e in transformed if e is not None]
    rejected_count = len(transformed) - len(clean_events)

    print(f" {len(clean_events)} événements valides")
    print(f" {rejected_count} événements rejetés (sans titre)")

    # Afficher le 1er événement transformé
    if clean_events:
        print("\n--- Exemple d'événement transformé ---")
        import json
        print(json.dumps(clean_events[0], indent=2, ensure_ascii=False))
    # Sauvegarde des événements dans data/raw/
    if clean_events:
        saved_path = save_events_to_json(clean_events)
        print(f"\n Données sauvegardées dans : {saved_path}")

if __name__ == "__main__":
    main()
