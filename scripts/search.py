"""
Recherche sémantique dans l'index FAISS construit par build_index.

Usage : python -m scripts.search "concert jazz à Paris"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    load_dotenv()

    # Récupère la requête depuis les arguments
    if len(sys.argv) < 2:
        print('Usage : python -m scripts.search "votre requête"')
        sys.exit(1)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Charge l'index existant
    index_dir = Path("data/index")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Recherche les k événements les plus proches
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f'\nRequête : "{query}"\n')
    print("=" * 70)
    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        print(f"\n#{rank}  (score: {score:.4f})  [uid: {meta.get('uid')}]")
        print(f"  Titre : {meta.get('title', '')}")
        print(f"  Lieu  : {meta.get('location_name', '')}, {meta.get('city', '')}")
        print(f"  Date  : {meta.get('date_range', '')}")
        desc = meta.get("description", "")
        if desc:
            print(f"  Desc  : {desc[:150]}{'...' if len(desc) > 150 else ''}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()