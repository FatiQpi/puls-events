# Puls-Events — Assistant RAG pour la recommandation d'événements culturels

Système de recommandation d'événements culturels basé sur une architecture RAG (Retrieval-Augmented Generation), exposé via une API REST et conteneurisé avec Docker. Projet réalisé dans le cadre de mes études AI Engineer.

## Contexte

Puls-Events est une plateforme spécialisée dans les recommandations culturelles personnalisées. Dans le cadre de ce projet, l'objectif est de livrer un POC démontrant la faisabilité technique, la pertinence métier et la performance d'un système RAG appliqué à la recommandation d'événements culturels.

J'ai choisi de restreindre le périmètre à l'Île-de-France. Ce cadrage n'était pas imposé : Open Agenda y expose un agenda régional agrégé unique, ce qui fournit un corpus cohérent sans travail d'agrégation multi-sources. Cela m'a permis de concentrer l'effort sur l'architecture RAG plutôt que sur la collecte.

## Objectifs du POC

- Collecter et structurer des données d'événements culturels depuis l'API Open Agenda
- Indexer ces événements dans une base vectorielle permettant une recherche sémantique rapide
- Générer des réponses en langage naturel aux questions des utilisateurs, appuyées sur les événements indexés
- Exposer le système via une API REST utilisable par les équipes produit et marketing
- Conteneuriser la solution pour garantir la portabilité et la reproductibilité

## Fonctionnement (pipeline RAG)

```
Question ─▶ embedding (Mistral) ─▶ recherche de similarité (FAISS)
                                          │
                          événements les plus proches (k=5)
                                          │
        prompt augmenté ─▶ LLM (Mistral) ─▶ réponse + sources
```

L'indexation suit le même principe en amont : chaque événement collecté est vectorisé une fois (un vecteur par événement, sans chunking) et stocké dans un index FAISS persisté sur disque.

Le choix de ne pas découper les textes découle d'une mesure sur le corpus (`scripts/analyze_lengths.py`, 1 769 événements) : le texte concaténé d'un événement fait 949 caractères en médiane, 2 447 au 95e centile, 5 141 au maximum, et seuls 2,1 % des événements dépassent 3 000 caractères — très en deçà de la fenêtre d'entrée de `mistral-embed`. Le chunking répond par ailleurs à deux besoins que je n'ai pas ici : faire tenir un document trop long dans la fenêtre du modèle, et cibler le passage pertinent au sein d'un texte traitant de plusieurs sujets. Mes événements sont courts et déjà autonomes — titre, lieu, date et description forment un ensemble cohérent. Les découper reviendrait à casser cette unité sans rien gagner.

## Stack technique

| Brique | Technologie | Rôle |
|---|---|---|
| Langage | Python 3.11 | Langage d'implémentation du projet |
| Framework d'orchestration | LangChain (LCEL) | Assemble les composants du pipeline RAG |
| Base vectorielle | FAISS (CPU) | Stockage et recherche de similarité sémantique |
| Modèle d'embedding | Mistral (`mistral-embed`) | Vectorisation des événements et des questions |
| Modèle de génération | Mistral (`mistral-small-latest`) | Génération des réponses en langage naturel |
| API REST | FastAPI + Uvicorn | Exposition du système via endpoints HTTP |
| Conteneurisation | Docker | Portabilité et reproductibilité du déploiement |
| Tests | pytest | Tests unitaires et fonctionnels |
| Évaluation | RAGAS | Mesure de la qualité du RAG (faithfulness, context recall) |
| Source de données | API Open Agenda | Récupération des événements culturels (agrégateur Île-de-France, uid 56500817) |
| Gestion des secrets | python-dotenv | Chargement des clés API depuis `.env` |

## Structure du projet

```
puls-events/
├── src/                    # Logique métier importable
│   ├── collect.py          #   collecte Open Agenda
│   ├── vectorize.py        #   construction de l'index FAISS
│   ├── rag_chain.py        #   chaîne RAG (LCEL)
│   └── api.py              #   API REST FastAPI
├── scripts/                # Points d'entrée (orchestrateurs)
│   ├── collect_events.py   #   lance la collecte
│   ├── build_index.py      #   construit l'index depuis le dernier export
│   ├── run_api.py          #   lance l'API en local (dev)
│   ├── ask.py              #   CLI pour interroger la chaîne
│   ├── evaluate_ragas.py   #   évaluation RAGAS
│   └── analyze_lengths.py  #   statistiques de longueur du corpus
├── tests/                  # Tests pytest
├── data/
│   ├── raw/                # Exports JSON datés (git-ignoré)
│   └── index/              # Index FAISS (git-ignoré)
├── Dockerfile
├── .dockerignore
├── .env.example
├── requirements.txt        # dépendances runtime (installées dans l'image Docker)
├── requirements-dev.txt    # + ragas, pour l'évaluation
└── README.md
```

> Les dossiers `data/raw/` et `data/index/` sont git-ignorés : ils sont générés localement (voir ci-dessous), pas versionnés.

## Prérequis

- **Python 3.11** installé sur la machine (testé avec Python 3.11.5)
- **Git** pour cloner le dépôt
- Une clé API **Mistral** et une clé API **Open Agenda**
- **Docker** (uniquement pour le déploiement conteneurisé)

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/FatiQpi/puls-events.git
cd puls-events
```

### 2. Créer et activer l'environnement virtuel (macOS / Linux)

```bash
python3 -m venv env
source env/bin/activate
```

Sur Windows :
```bash
env\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer les clés API

```bash
cp .env.example .env
```

Puis éditez `.env` et renseignez vos clés (format strict `NOM=valeur`, sans espace autour du `=`) :

```
MISTRAL_API_KEY=votre_cle_mistral
OPENAGENDA_API_KEY=votre_cle_openagenda
```

Le fichier `.env` est git-ignoré et n'est jamais commité ni copié dans l'image Docker.

### 5. Vérifier l'installation

```bash
python -c "import faiss; from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings; print('Environnement OK')"
```

## Utilisation en local (sans Docker)

```bash
# 1. Collecter les événements depuis Open Agenda → data/raw/
python -m scripts.collect_events

# 2. Construire l'index FAISS depuis le dernier export → data/index/
python -m scripts.build_index

# 3. Lancer l'API
python -m scripts.run_api
```

L'API démarre sur http://127.0.0.1:8000. Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

## Déploiement local avec Docker

L'API est conteneurisée pour un lancement local reproductible. L'index FAISS est **inclus dans l'image** : il doit donc être construit sur votre machine *avant* le build de l'image (collecte et embedding nécessitent vos clés API et un accès internet).

### 1. Préparer les données et l'index (une fois, avant le build)

Depuis l'environnement virtuel avec les dépendances installées (étapes d'installation ci-dessus) :

```bash
python -m scripts.collect_events   # collecte Open Agenda → data/raw/
python -m scripts.build_index      # construit l'index FAISS → data/index/
```

### 2. Construire l'image

```bash
docker build -t puls-events .
```

### 3. Lancer le conteneur

```bash
docker run --env-file .env -p 8000:8000 puls-events
```

L'API est disponible sur http://localhost:8000 — Swagger sur http://localhost:8000/docs

> **Rafraîchir les données :** `POST /rebuild` re-collecte et ré-indexe à chaud dans le conteneur en cours. Ce rafraîchissement ne survit pas à la suppression du conteneur (l'image reste immuable) ; pour figer un index frais dans l'image, relancez l'étape 1 puis reconstruisez l'image.

## Endpoints de l'API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Vérifie que le serveur répond |
| GET | `/metadata` | Informations sur le corpus (nombre d'événements, source) et la configuration (modèles, k de recherche) |
| POST | `/ask` | Pose une question en langage naturel, renvoie `{answer, sources}` |
| POST | `/rebuild` | Re-collecte, ré-indexe et recharge la chaîne à chaud |

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Je cherche un concert de jazz"}'
```

## Tests

```bash
pytest
```

`pytest` exécute 20 tests : 15 tests unitaires sur les fonctions pures (transformation des événements, construction du texte et des métadonnées, formatage du contexte) et 5 tests fonctionnels sur l'API, qui vérifient les contrats des endpoints via le `TestClient` de FastAPI.

Aucun test n'appelle le LLM : ils sont rapides et déterministes. Les tests d'API instancient l'application, donc son cycle de démarrage : l'index (`data/index`) et le fichier `.env` doivent être présents.

## Évaluation

L'évaluation nécessite une dépendance supplémentaire, absente de l'image Docker :

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Le système est évalué avec RAGAS 0.4.3 (`python -m scripts.evaluate_ragas`) sur un jeu de questions de référence. Résultats obtenus : **faithfulness 0.826**, **context_recall 0.713**. La métrique `answer_relevancy` n'a pas pu être obtenue (timeout lié au rate limit du tier gratuit).

## Limites et perspectives

- **Rate limit** : le tier gratuit Mistral limite les gros volumes d'embedding (rebuild, évaluation) et empêche notamment d'obtenir `answer_relevancy`. Contournement : exécution des métriques RAGAS séparément, à faible concurrence. Solution : passage à un tier Mistral payant (non rate-limité).
- **Persistance** : un `/rebuild` en conteneur ne persiste pas après suppression du conteneur (conséquence du choix d'un index inclus dans l'image).
