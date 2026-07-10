# Image de base : Python 3.11 slim 
FROM python:3.11-slim

# Répertoire de travail à l'intérieur du conteneur.
WORKDIR /app

# 1) Dépendances d'abord. Cette couche n'est reconstruite que si requirements.txt change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code applicatif + index FAISS pré-construit. L'index doit exister AVANT le build (cf. README : collect puis build_index).
COPY src/ ./src/
COPY data/index/ ./data/index/

# Port de l'API 
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]