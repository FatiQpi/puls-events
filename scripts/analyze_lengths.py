"""Statistiques de longueur du corpus.

Mesure la longueur du texte réellement vectorisé pour chaque événement, afin
de faire un choix concernant le chunking. La fonction `build_text` est importée de
`src.vectorize`.

Lancement : python -m scripts.analyze_lengths
"""

import os

# Doit être posé avant l'import de FAISS (via src.vectorize) sur macOS.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

from src.vectorize import build_text

# Charge le JSON le plus récent
path = sorted(Path("data/raw").glob("events_idf_*.json"))[-1]
events = json.loads(path.read_text(encoding="utf-8"))

lengths = sorted(len(build_text(e)) for e in events)
n = len(lengths)

print(f"Fichier analysé : {path.name}")
print(f"Total événements : {n}")
print(f"Médiane : {lengths[n//2]} caractères")
print(f"P95     : {lengths[int(n*0.95)]} caractères")
print(f"Max     : {lengths[-1]} caractères")
print(f"> 3000  : {sum(1 for l in lengths if l > 3000)} ({100*sum(1 for l in lengths if l > 3000)/n:.1f}%)")