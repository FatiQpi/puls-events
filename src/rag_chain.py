"""Module d'assemblage de la chaîne RAG pour Puls-Events.

Charge l'index FAISS existant, configure le LLM Mistral, et compose une
chaîne LCEL qui prend une question (str) en entrée et retourne un dict
{answer, docs, question} en sortie : la réponse générée ET les documents
récupérés (pour exposer les sources via l'API).

Flux : question -> retriever (docs conservés) -> format_docs -> prompt -> llm
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings



# Configuration par défaut

DEFAULT_INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = "mistral-embed" 
LLM_MODEL = "mistral-small-latest"
DEFAULT_K = 5
DEFAULT_TEMPERATURE = 0  


SYSTEM_PROMPT = """Tu es un assistant culturel pour la plateforme Puls-Events,
qui couvre les événements culturels de toute l'Île-de-France (Paris et sa région).

À partir de la liste d'événements fournie en contexte, tu recommandes ceux qui
correspondent le mieux à la demande de l'utilisateur.

Règles :
- Ne recommande QUE des événements présents dans le contexte ci-dessous ; n'invente jamais.
- Le périmètre couvre toute l'Île-de-France : interprète les demandes géographiques
  avec souplesse. Si l'utilisateur demande « à Paris » mais que les événements
  disponibles sont en banlieue, propose-les en précisant clairement leur ville.
- Si vraiment aucun événement du contexte n'est pertinent, dis-le franchement.
- Pour chaque recommandation, cite le titre, le lieu et la date.
- Reproduis les titres, lieux, villes et dates exactement tels qu'ils figurent
  dans le contexte : ne les modifie pas, ne les complète pas, n'ajoute jamais
  une année ou une information absente.
- Réponds en français, sur un ton factuel et concis."""

# Chargement de l'index
def load_vectorstore(index_dir: Path = DEFAULT_INDEX_DIR) -> FAISS:
    """Charge l'index FAISS persistant depuis le disque.

    Le modèle d'embedding doit être identique à celui utilisé lors de
    l'indexation (cf. src/vectorize.py).
    """
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        # Le flag est requis par LangChain par sécurité (pickle peut
        # exécuter du code arbitraire au chargement).
        allow_dangerous_deserialization=True,
    )


# Formatage des documents pour le prompt
def format_docs(docs: list[Document]) -> str:
    """Reconstruit un texte structuré pour le LLM à partir des metadata.

    Le page_content des Documents est optimisé pour l'embedding (concaténation
    brute) ; ici, on exploite les champs metadata structurés pour produire un
    texte que le LLM peut lire et citer plus facilement.
    """
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        lines.append(
            f"[Événement {i}]\n"
            f"Titre : {meta.get('title', '')}\n"
            f"Lieu : {meta.get('location_name', '')}, {meta.get('city', '')}\n"
            f"Date : {meta.get('date_range', '')}\n"
            f"Description : {meta.get('description', '')}"
        )
    return "\n\n".join(lines)


# Assemblage de la chaîne LCEL
def build_chain(
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: str = LLM_MODEL,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Runnable:
    """Construit la chaîne RAG complète.

    Retourne un Runnable[str, dict] : prend une question en entrée, retourne
    un dict {"docs": [...], "question": "...", "answer": "..."}. Les documents
    récupérés sont conservés pour permettre à l'API d'exposer les sources.
    """
    vectorstore = load_vectorstore(index_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatMistralAI(model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Contexte (événements pertinents) :\n{context}\n\nQuestion : {question}",
            ),
        ]
    )

    # Sous-chaîne de génération : à partir de {docs, question}, formate le
    # contexte puis génère la réponse textuelle.
    generate_answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    # On récupère les docs et on garde la question, puis on ajoute 'answer'
    # sans perdre les docs (qui deviendront les sources côté API).
    return (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=generate_answer)
    )