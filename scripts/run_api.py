"""Lancement local de l'API Puls-Events.

Usage : python -m scripts.run_api
Puis ouvrir http://127.0.0.1:8000/docs pour la documentation interactive.
"""
import uvicorn


def main():
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()