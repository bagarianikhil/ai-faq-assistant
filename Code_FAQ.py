# Databricks notebook source
import huggingface_hub, sentence_transformers, transformers
print("huggingface_hub:", huggingface_hub.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("transformers:", transformers.__version__)

from sentence_transformers import SentenceTransformer
# Should download or load from cache without import errors:
_ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ============================================================
# MAGIC ###  FAQ Assistant
# MAGIC
# MAGIC What you get:
# MAGIC -    Load & clean FAQs (columns: prompt, response)
# MAGIC -    Embedding retriever (Sentence-Transformers) + FAISS index
# MAGIC -    Optional OpenAI RAG answer constrained to retrieved context
# MAGIC -    Baseline TF-IDF retriever (for comparison)
# MAGIC -    Evaluation: self-retrieval on prompts
# MAGIC
# MAGIC Setup:
# MAGIC - Optional Set OPENAI_API_KEY in your environment/secret scopes, then USE_OPENAI = True
# MAGIC
# MAGIC ============================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Packages

# COMMAND ----------

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from joblib import dump, load

# Baseline TF-IDF (optional; useful for evaluation/comparison)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

# Optional .env support for OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ### Config

# COMMAND ----------

# -----------------------------
# Config — adjust as needed
# -----------------------------
CSV_PATH = "faqs.csv"      # e.g., "Filepath  - faqs.csv"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & accurate for FAQs
TOP_K = 5
USE_OPENAI = False                    # True if OPENAI_API_KEY is set

print("Config ->", {
    "CSV_PATH": CSV_PATH,
    "EMBED_MODEL_NAME": EMBED_MODEL_NAME,
    "TOP_K": TOP_K,
    "USE_OPENAI": USE_OPENAI
})


# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Normalization

# COMMAND ----------

# -----------------------------
# Text normalization
# -----------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _URL_RE.sub(" ", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s)
    return s

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Structures

# COMMAND ----------

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class FAQItem:
    prompt: str
    response: str

@dataclass
class FAQCorpus:
    items: List[FAQItem]
    df: pd.DataFrame  # with columns: prompt, response, prompt_norm, response_norm

@dataclass
class EmbeddingIndex:
    index: faiss.Index
    dim: int
    id_to_row: List[int]
    model_name: str

@dataclass
class TfidfIndex:
    vectorizer: TfidfVectorizer
    matrix: any
    id_to_row: List[int]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Clean Data

# COMMAND ----------

# -----------------------------
# Load & clean CSV
# -----------------------------
def load_faqs(csv_path: str) -> FAQCorpus:
    df = pd.read_csv(
        csv_path,
        encoding="ISO-8859-1",
        engine="python",
        on_bad_lines="skip",
        escapechar="\\"
    )
    expected = {"prompt", "response"}
    lower_cols = set(df.columns.str.lower())
    missing = expected - lower_cols
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.dropna(subset=["prompt", "response"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    # Remove duplicate prompts
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    # Normalized copies for indexing
    df["prompt_norm"] = df["prompt"].apply(normalize_text)
    df["response_norm"] = df["response"].apply(normalize_text)
    items = [FAQItem(r["prompt"], r["response"]) for _, r in df.iterrows()]
    return FAQCorpus(items=items, df=df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embeddings

# COMMAND ----------

# -----------------------------
# Embeddings + FAISS
# -----------------------------
def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    if normalize:
        faiss.normalize_L2(embs)  # enables cosine via inner product
    return embs

def build_faiss_index(embeddings: np.ndarray, model_name: str) -> EmbeddingIndex:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP on normalized vectors = cosine
    index.add(embeddings)
    id_to_row = list(range(embeddings.shape[0]))
    return EmbeddingIndex(index=index, dim=dim, id_to_row=id_to_row, model_name=model_name)

def search_faiss(eindex: EmbeddingIndex, embedder: SentenceTransformer, corpus: FAQCorpus, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
    q = normalize_text(query)
    q_emb = embedder.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = eindex.index.search(q_emb, top_k)
    scores = scores[0]; idxs = idxs[0]
    mapped: List[Tuple[int, float]] = []
    for i, score in zip(idxs.tolist(), scores.tolist()):
        if i == -1:
            continue
        row_idx = corpus.df.index[eindex.id_to_row[i]]
        mapped.append((row_idx, float(score)))
    return mapped

# COMMAND ----------

# MAGIC %md
# MAGIC ### TF-IDF

# COMMAND ----------

# -----------------------------
# TF-IDF baseline (for comparison)
# -----------------------------
def build_tfidf_index(corpus: FAQCorpus, ngram_max: int = 2, stop_words: str = "english") -> TfidfIndex:
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=1, stop_words=stop_words)
    mat = vec.fit_transform(corpus.df["prompt_norm"].tolist())
    id_to_row = list(range(len(corpus.df)))
    return TfidfIndex(vectorizer=vec, matrix=mat, id_to_row=id_to_row)

def search_tfidf(tindex: TfidfIndex, corpus: FAQCorpus, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
    q = normalize_text(query)
    q_vec = tindex.vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tindex.matrix)[0]
    top_ids = np.argsort(-sims)[:top_k]
    return [(corpus.df.index[tindex.id_to_row[i]], float(sims[i])) for i in top_ids]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Answering

# COMMAND ----------

# -----------------------------
# Answering
# -----------------------------
def format_context(corpus: FAQCorpus, hits: List[Tuple[int, float]]) -> List[Dict]:
    ctx = []
    for row_idx, score in hits:
        r = corpus.df.iloc[row_idx]
        ctx.append({"prompt": r["prompt"], "response": r["response"], "score": round(score, 4)})
    return ctx

def answer_direct(corpus: FAQCorpus, hits: List[Tuple[int, float]]) -> str:
    if not hits:
        return "Sorry, I couldn't find an answer to that."
    top_row, _ = hits[0]
    return str(corpus.df.iloc[top_row]["response"])

def answer_rag_openai(corpus: FAQCorpus, hits: List[Tuple[int, float]], user_query: str,
                      model: str = "gpt-4o", api_key: Optional[str] = None, temperature: float = 0.2) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return answer_direct(corpus, hits)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        context_blocks = format_context(corpus, hits)
        system_prompt = (
            "You are a concise FAQ assistant. Answer ONLY from the provided context. "
            "If the answer is not present, say you do not know."
        )
        user_msg = (
            f"User question:\n{user_query}\n\n"
            "Context (top matches):\n" + json.dumps(context_blocks, ensure_ascii=False, indent=2)
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error: {e}]\n" + answer_direct(corpus, hits)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Engine

# COMMAND ----------

# -----------------------------
# Engine
# -----------------------------
class Engine:
    def __init__(self, corpus: FAQCorpus,
                 retriever: str = "faiss",
                 embedder: Optional[SentenceTransformer] = None,
                 eindex: Optional[EmbeddingIndex] = None,
                 tindex: Optional[TfidfIndex] = None):
        self.corpus = corpus
        self.retriever = retriever
        self.embedder = embedder
        self.eindex = eindex
        self.tindex = tindex
        assert retriever in {"faiss", "tfidf"}, "retriever must be 'faiss' or 'tfidf'"

    def _search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.retriever == "faiss":
            return search_faiss(self.eindex, self.embedder, self.corpus, query, top_k=top_k)
        else:
            return search_tfidf(self.tindex, self.corpus, query, top_k=top_k)

    def query(self, text: str, top_k: int = 5, use_openai: bool = False) -> Dict:
        hits = self._search(text, top_k=top_k)
        contexts = format_context(self.corpus, hits)
        ans = answer_rag_openai(self.corpus, hits, text) if use_openai else answer_direct(self.corpus, hits)
        return {"query": text, "answer": ans, "matches": contexts}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation

# COMMAND ----------

# -----------------------------
# Evaluation:
# -----------------------------
def evaluate_retriever(corpus: FAQCorpus,
                       engine: Engine,
                       top_k_list: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Self-retrieval evaluation: for each prompt, query the retriever and check
    whether the ground-truth row is in top-K. Also compute MRR@K (with max K).
    """
    max_k = max(top_k_list)
    n = len(corpus.df)
    hits_per_k = {k: 0 for k in top_k_list}
    rr_list: List[float] = []

    for i in range(n):
        q = corpus.df.iloc[i]["prompt"]
        true_row_idx = corpus.df.index[i]
        hits = engine._search(q, top_k=max_k)
        rank = None
        for pos, (row_idx, _) in enumerate(hits, start=1):
            if row_idx == true_row_idx:
                rank = pos
                break
        rr_list.append(1.0 / rank if rank is not None else 0.0)
        for k in top_k_list:
            if rank is not None and rank <= k:
                hits_per_k[k] += 1

    metrics = {f"Hit@{k}": round(hits_per_k[k] / n, 4) for k in top_k_list}
    metrics[f"MRR@{max_k}"] = round(float(np.mean(rr_list)), 4)
    metrics["queries"] = n
    return pd.DataFrame([metrics])


# COMMAND ----------

# -----------------------------
# Build everything & quick runs
# -----------------------------
corpus = load_faqs(CSV_PATH)
print(f"Loaded FAQs: {len(corpus.df)} rows")

# Build Embedding + FAISS
embedder = load_embedder(EMBED_MODEL_NAME)
embeddings = embed_texts(embedder, corpus.df["prompt_norm"].tolist(), batch_size=64, normalize=True)
eindex = build_faiss_index(embeddings, EMBED_MODEL_NAME)
faiss_engine = Engine(corpus=corpus, retriever="faiss", embedder=embedder, eindex=eindex)

# Baseline TF-IDF (optional)
tfidf_index = build_tfidf_index(corpus)
tfidf_engine = Engine(corpus=corpus, retriever="tfidf", tindex=tfidf_index)

# -----------------------------
# Helper: ask()
# -----------------------------
def ask(q: str, top_k: int = TOP_K, use_openai: bool = USE_OPENAI, retriever: str = "faiss", show_matches: bool = True):
    engine = faiss_engine if retriever == "faiss" else tfidf_engine
    result = engine.query(q, top_k=top_k, use_openai=use_openai)
    print(f"Retriever: {retriever.upper()} | Question: {result['query']}")
    print("\nAnswer:\n" + result["answer"])
    if show_matches:
        print("\nTop matches:")
        for i, m in enumerate(result["matches"], start=1):
            print(f"{i}. score={m['score']:.4f} | Q: {m['prompt']}")
            print(f"   A: {m['response'][:160]}{'...' if len(m['response'])>160 else ''}")

# -----------------------------
# Smoke tests (Can modify/remove)
# -----------------------------
print("-"*80); ask("Do you provide any job assistance?", retriever="faiss", show_matches=False); print()
print("-"*80); ask("Does Power BI work on Mac?", retriever="faiss", show_matches=False); print()
print("-"*80); ask("Is there lifetime access?", retriever="faiss", show_matches=False); print()

# -----------------------------
# Evaluation (both retrievers)
# -----------------------------
print("\nEvaluating FAISS retriever (this runs over the full dataset)...")
faiss_eval_df = evaluate_retriever(corpus, faiss_engine, top_k_list=[1,3,5,10])
try:
    display(faiss_eval_df)
except Exception:
    print(faiss_eval_df)

print("\nEvaluating TF-IDF retriever (baseline)...")
tfidf_eval_df = evaluate_retriever(corpus, tfidf_engine, top_k_list=[1,3,5,10])
try:
    display(tfidf_eval_df)
except Exception:
    print(tfidf_eval_df)