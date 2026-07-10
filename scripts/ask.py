"""Interrogation de la chaîne RAG depuis la ligne de commande.

Usage : python -m scripts.ask "Je cherche un concert de jazz à Paris"
"""
import sys

from dotenv import load_dotenv

from src.rag_chain import build_chain


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage : python -m scripts.ask "votre question"')
        sys.exit(1)

    question = sys.argv[1]

    print("Initialisation de la chaîne RAG (chargement de l'index FAISS)...")
    chain = build_chain()

    print(f'\nQuestion : "{question}"')
    print("=" * 70)
    print()

    result = chain.invoke(question)
    print(result["answer"])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()