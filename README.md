# Puls-Events — Assistant RAG pour la recommandation d'événements culturels

Système de recommandation d'événements culturels basé sur une architecture RAG (Retrieval-Augmented Generation), exposé via une API REST. Projet réalisé dans le cadre de mes études AI Engineer.

## Contexte

Puls-Events est une plateforme spécialisée dans les recommandations culturelles personnalisées. Dans le cadre de ce projet, l'objectif est de livrer un POC démontrant la faisabilité technique, la pertinence métier et la performance d'un système RAG appliqué à la recommandation d'événements culturels.

## Objectifs du POC

- Collecter et structurer des données d'événements culturels depuis l'API Open Agenda
- Indexer ces événements dans une base vectorielle permettant une recherche sémantique rapide
- Générer des réponses en langage naturel aux questions des utilisateurs, appuyées sur les événements indexés
- Exposer le système via une API REST utilisable par les équipes produit et marketing
- Conteneuriser la solution pour garantir la portabilité et la reproductibilité

## Stack technique

| Brique | Technologie | Rôle |
|---|---|---|
| Langage | Python 3.11 | Langage d'implémentation du projet |
| Framework d'orchestration | LangChain | Assemble les composants du pipeline RAG |
| Base vectorielle | FAISS (CPU) | Stockage et recherche de similarité sémantique |
| Modèle d'embedding | Mistral | Vectorisation des événements et des questions |
| Modèle de génération | Mistral (LLM) | Génération des réponses en langage naturel |
| Source de données | API Open Agenda | Récupération des événements culturels |
| Gestion des secrets | python-dotenv | Chargement des clés API depuis `.env` |
| Manipulation des données | pandas | Nettoyage et exploration des données |

## Prérequis

- **Python 3.11** installé sur la machine (testé avec Python 3.11.5)
- **Git** pour cloner le dépôt

## Installation

### 1. Cloner le dépôt

```bash
git clone <URL_DU_DEPOT>
cd Puls-Events
```

### 2. Créer et activer l'environnement virtuel (mac)

```bash
python3 -m venv env
source env/bin/activate
```

Sur Windows, remplacer la commande d'activation par :
```bash
env\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python -c "import faiss; from langchain_community.vectorstores import FAISS; from langchain_huggingface import HuggingFaceEmbeddings; from mistralai.client.sdk import Mistral; print('Environnement OK')"
```
