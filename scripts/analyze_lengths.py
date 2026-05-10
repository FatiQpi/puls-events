import json
from pathlib import Path

# Charge le JSON le plus récent
path = sorted(Path("data/raw").glob("events_idf_*.json"))[-1]
events = json.loads(path.read_text(encoding="utf-8"))

# Construit le texte concaténé pour chaque événement
def build_text(e):
    loc = e.get("location") or {}
    return "\n".join(filter(None, [
        f"Titre: {e.get('title', '')}",
        f"Description: {e.get('description', '')}",
        f"Détails: {e.get('long_description', '')}",
        f"Lieu: {loc.get('name', '')}, {loc.get('city', '')}",
        f"Date: {e.get('date_range', '')}",
    ]))

lengths = sorted(len(build_text(e)) for e in events)
n = len(lengths)

print(f"Total événements : {n}")
print(f"Médiane : {lengths[n//2]} caractères")
print(f"P95     : {lengths[int(n*0.95)]} caractères")
print(f"Max     : {lengths[-1]} caractères")
print(f"> 3000  : {sum(1 for l in lengths if l > 3000)} ({100*sum(1 for l in lengths if l > 3000)/n:.1f}%)")