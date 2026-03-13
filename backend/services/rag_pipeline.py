"""
RAG (Retrieval-Augmented Generation) pipeline.

1. Load and chunk a corpus of pricing documents
2. Embed chunks with sentence-transformers
3. Store in FAISS index
4. Retrieve top-K chunks for a query
5. Synthesize an answer via Nemotron NIM
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_index = None
_chunks: list[str] = []
_embedder = None

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers embedder")
        except Exception as e:
            logger.warning(f"Could not load embedder: {e}")
            _embedder = "fallback"
    return _embedder


def _build_corpus() -> list[str]:
    """Build corpus from documents + generated module summaries."""
    documents = []

    documents.extend([
        "Price elasticity of demand measures how sensitive consumer purchases are to changes "
        "in price. For retail locations, elasticity typically ranges from -0.3 to -2.0 depending on "
        "location type. Motorway locations tend to have lower elasticity (customers are captive) "
        "while urban locations face higher elasticity due to more competitor options nearby.",

        "SHAP (SHapley Additive exPlanations) values decompose a model prediction into individual "
        "feature contributions. In pricing, a SHAP waterfall chart shows how each factor "
        "(price gap to competitors, time of day, weather, crude oil price) pushes the predicted "
        "volume up or down from the average. Positive SHAP values indicate the feature increases "
        "the prediction; negative values decrease it.",

        "The MTS-K (Markttransparenzstelle für Kraftstoffe) is Germany's fuel price transparency "
        "agency. Since 2013, it has collected real-time price data from all ~14,500 retail locations "
        "(e.g. fuel stations for diesel, petrol) in Germany. Station operators must report every "
        "price change within 5 minutes. This data is publicly available and forms the basis of "
        "competitive pricing analysis.",

        "Crude oil price pass-through to retail prices is not instantaneous. Research shows "
        "a 'rockets and feathers' asymmetry: retail prices rise faster when crude increases "
        "(rockets) but fall more slowly when crude decreases (feathers). This lag ranges from "
        "2-8 days depending on market competition intensity.",

        "Isolation Forest is an unsupervised anomaly detection algorithm well-suited for sales "
        "volume anomalies. It works by building random trees that isolate observations. Anomalies "
        "are isolated earlier (shorter path lengths) because they differ significantly from normal "
        "patterns. In pricing, anomalies may indicate: pump failures, local events driving "
        "unusual traffic, competitor price wars, or data quality issues.",

        "Multi-armed bandits provide an online learning framework for pricing experiments. "
        "Thompson Sampling maintains a posterior distribution over the reward (profit) for each "
        "price level and samples from it to select actions. This naturally balances exploration "
        "(trying uncertain price points) with exploitation (choosing known profitable prices).",

        "Q-Learning is a model-free reinforcement learning algorithm that learns an optimal "
        "pricing policy through trial and error. The agent observes the pricing state (price gap "
        "to competitors, volume level, time of day, competitor trend) and learns which pricing "
        "action (raise, lower, maintain) maximizes long-term cumulative profit.",

        "The FT-Transformer (Feature Tokenizer + Transformer) applies self-attention to tabular "
        "data by converting each feature into a token. For pricing, this allows the model "
        "to discover complex interactions between features like temperature, competition, and "
        "time-of-day that simpler models might miss.",

        "A pricing governance framework ensures responsible deployment of AI pricing models. "
        "Key elements include: (1) human-in-the-loop for price changes above a threshold, "
        "(2) A/B testing before full rollout, (3) fairness audits to prevent discriminatory "
        "pricing, (4) model monitoring for drift detection, (5) rollback procedures.",

        "Walk-forward validation is critical for time series models in pricing. Unlike random "
        "cross-validation, it respects temporal ordering: the model is trained on historical data "
        "and tested on future data, then the window slides forward. This prevents data leakage "
        "and gives realistic out-of-sample performance estimates.",
    ])

    if CORPUS_DIR.exists():
        for fp in CORPUS_DIR.glob("*.md"):
            documents.append(fp.read_text())
        for fp in CORPUS_DIR.glob("*.txt"):
            documents.append(fp.read_text())

    return documents


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_index(force: bool = False) -> int:
    """Build or rebuild the FAISS index from the corpus."""
    global _index, _chunks

    if _index is not None and not force:
        return len(_chunks)

    documents = _build_corpus()
    _chunks = []
    for doc in documents:
        _chunks.extend(_chunk_text(doc))

    embedder = _get_embedder()
    if embedder == "fallback":
        logger.warning("Using random embeddings (embedder not available)")
        embeddings = np.random.default_rng(42).random((len(_chunks), 384)).astype("float32")
    else:
        embeddings = embedder.encode(_chunks, show_progress_bar=False, convert_to_numpy=True)

    try:
        import faiss
        dim = embeddings.shape[1]
        _index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        _index.add(embeddings)
        logger.info(f"Built FAISS index with {len(_chunks)} chunks, dim={dim}")
    except ImportError:
        logger.warning("FAISS not available, using numpy cosine similarity")
        _index = embeddings

    return len(_chunks)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-K relevant chunks for a query."""
    if _index is None:
        build_index()

    embedder = _get_embedder()
    if embedder == "fallback":
        q_emb = np.random.default_rng(hash(query) % 2**32).random((1, 384)).astype("float32")
    else:
        q_emb = embedder.encode([query], convert_to_numpy=True)

    try:
        import faiss
        if hasattr(_index, "search"):
            faiss.normalize_L2(q_emb)
            scores, indices = _index.search(q_emb, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(_chunks):
                    results.append({"chunk": _chunks[idx], "score": float(score), "index": int(idx)})
            return results
    except (ImportError, AttributeError):
        pass

    if isinstance(_index, np.ndarray):
        from numpy.linalg import norm
        q_norm = q_emb / (norm(q_emb) + 1e-8)
        idx_norm = _index / (norm(_index, axis=1, keepdims=True) + 1e-8)
        scores = (idx_norm @ q_norm.T).ravel()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{"chunk": _chunks[i], "score": float(scores[i]), "index": int(i)}
                for i in top_indices]

    return [{"chunk": c, "score": 0.5, "index": i} for i, c in enumerate(_chunks[:top_k])]


def query_rag(question: str, top_k: int = 5, temperature: float = 0.5) -> dict:
    """Full RAG pipeline: retrieve + synthesize."""
    retrieved = retrieve(question, top_k=top_k)
    context = "\n\n---\n\n".join([r["chunk"] for r in retrieved])

    from services.llm_client import pricing_chat
    answer = pricing_chat(question, context=context, temperature=temperature)

    return {
        "answer": answer,
        "sources": retrieved,
        "query": question,
        "n_chunks_used": len(retrieved),
    }
