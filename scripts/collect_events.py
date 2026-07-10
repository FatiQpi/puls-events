"""Collecte des événements OpenAgenda (Île-de-France) vers un export JSON daté.

Usage : python -m scripts.collect_events
"""
from dotenv import load_dotenv

from src.collect import collect_events


def main():
    load_dotenv()
    print("Collecte des événements OpenAgenda (Île-de-France)...\n")
    path = collect_events()
    print(f"\nDonnées sauvegardées dans : {path}")


if __name__ == "__main__":
    main()