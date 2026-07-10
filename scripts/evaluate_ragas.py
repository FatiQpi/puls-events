"""Évaluation de la chaîne RAG Puls-Events avec RAGAS.

Usage : python -m scripts.evaluate_ragas

Le LLM-juge et les embeddings sont ceux du projet (Mistral), afin de rester sur
le tier gratuit sans dépendre d'OpenAI.

Métriques disponibles (jugées par le LLM) :
  - faithfulness    : fidélité de la réponse au contexte récupéré (anti-hallucination)
  - relevancy       : pertinence de la réponse vis-à-vis de la question
  - context_recall  : couverture du contexte récupéré par rapport à la référence

Contrainte connue : le tier gratuit Mistral (Experiment) limite fortement le
débit. La faithfulness sur l'ensemble du jeu passe sans souci ; l'ajout de
plusieurs métriques sature le quota (timeouts). Le bloc CONFIG ci-dessous
permet d'ajuster le nombre de questions et de métriques en conséquence.
"""
import os

# Doit être posé avant l'import de FAISS (conflit libomp sur macOS).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# Les métriques sont importées depuis ragas.metrics :
# l'API "collections" exige de passer le LLM au constructeur de chaque métrique,
# alors que l'API historique le reçoit globalement via evaluate(). On conserve
# cette dernière, plus simple.
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

from src.rag_chain import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    build_chain,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TEST_SET_PATH = Path("data/eval/test_set.json")
RESULTS_DIR = Path("data/eval")

# Nombre de questions à évaluer (None = toutes).
MAX_QUESTIONS = None

# Métriques à calculer, parmi : "faithfulness", "relevancy", "context_recall".
# Par défaut : faithfulness seule, seule configuration fiable sur le tier gratuit.
USE_METRICS = ["faithfulness"]

# Concurrence des appels au LLM-juge. 1 = le plus doux pour le rate limit.
MAX_WORKERS = 1


def select_metrics(names):
    """Retourne les objets métriques RAGAS correspondant aux noms demandés."""
    available = {
        "faithfulness": Faithfulness(),
        "relevancy": ResponseRelevancy(),
        "context_recall": LLMContextRecall(),
    }
    return [available[name] for name in names]


def build_samples(test_set, chain):
    """Construit les échantillons RAGAS à partir du jeu de test.

    Pour chaque question, capture :
      - user_input        : la question
      - response          : la réponse générée par la chaîne RAG
      - retrieved_contexts: le texte des documents récupérés par la chaîne
      - reference         : la réponse de référence annotée (pour context_recall)

    La chaîne renvoie désormais {answer, docs, question}, ce qui permet de
    récupérer réponse et contextes en un seul appel (plus de double récupération).
    """
    samples = []
    for item in test_set:
        question = item["question"]
        reference = item.get("reference_answer", "")

        result = chain.invoke(question)
        retrieved_contexts = [doc.page_content for doc in result["docs"]]
        response = result["answer"]

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
        )
        print(f"  [{item['id']}] préparé")

    return samples


def main():
    load_dotenv()

    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    if MAX_QUESTIONS is not None:
        test_set = test_set[:MAX_QUESTIONS]

    print(f"Jeu de test : {len(test_set)} questions")
    print(f"Métriques : {USE_METRICS}   |   max_workers : {MAX_WORKERS}\n")

    # Chaîne RAG (elle renvoie désormais réponse + documents récupérés)
    chain = build_chain()

    print("Préparation des échantillons...")
    samples = build_samples(test_set, chain)
    dataset = EvaluationDataset(samples=samples)

    # LLM-juge et embeddings : Mistral (tier gratuit, pas d'OpenAI requis)
    evaluator_llm = LangchainLLMWrapper(ChatMistralAI(model=LLM_MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(MistralAIEmbeddings(model=EMBEDDING_MODEL))

    run_config = RunConfig(
        max_workers=MAX_WORKERS,
        max_retries=15,
        max_wait=90,
        timeout=600,
    )

    print("\nLancement de l'évaluation RAGAS (peut prendre plusieurs minutes)...\n")
    result = evaluate(
        dataset=dataset,
        metrics=select_metrics(USE_METRICS),
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
    )

    print("\n" + "=" * 70)
    print("RÉSULTATS RAGAS")
    print("=" * 70)
    print(result)
    print("=" * 70)

    # Sauvegarde du détail par question (exploitable pour le rapport).
    # Le nom du fichier dépend des métriques pour ne pas écraser les runs précédents.
    suffix = "_".join(USE_METRICS)
    results_path = RESULTS_DIR / f"ragas_{suffix}.csv"
    result.to_pandas().to_csv(results_path, index=False)
    print(f"\nDétail par question sauvegardé dans : {results_path}")


if __name__ == "__main__":
    main()