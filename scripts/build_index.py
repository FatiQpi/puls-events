"""
Construit l'index FAISS à partir du dernier export d'événements.

Usage : python scripts/build_index.py
"""

from pathlib import Path
from dotenv import load_dotenv

from src.vectorize import load_events, build_index


def main():
    load_dotenv()

    # Trouve le JSON le plus récent
    raw_dir = Path("data/raw")
    json_files = sorted(raw_dir.glob("events_idf_*.json"))
    if not json_files:
        raise FileNotFoundError(f"Aucun fichier events_idf_*.json dans {raw_dir}")

    latest = json_files[-1]
    print(f"Fichier source : {latest}")

    # Charge et indexe
    events = load_events(latest)
    print(f"Événements chargés : {len(events)}")

    output_dir = Path("data/index")
    build_index(events, output_dir)


if __name__ == "__main__":
    main()