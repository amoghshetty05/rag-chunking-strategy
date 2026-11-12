"""
RAG Chunking & Embedding Comparison - live demo
Run from the project folder:  streamlit run src/app.py
"""
import csv
import pickle
import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODELS = {"minilm": "all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}
MODEL_LABELS = {"minilm": "all-MiniLM-L6-v2 (384-d, fast)", "bge": "BGE-small-en-v1.5 (384-d)"}
CHUNKINGS = ["fixed", "sentence", "semantic"]
CHUNK_LABELS = {
    "fixed": "Fixed-size (500 chars, 50 overlap)",
    "sentence": "Sentence-level (5 sentences, 1 overlap)",
    "semantic": "Semantic (meaning-shift split)",
}

st.set_page_config(page_title="RAG Chunking Comparison", page_icon="🔎", layout="wide")


# ---------- cached loaders (load once, reuse on every click) ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model(key):
    m = SentenceTransformer(MODELS[key])
    m.encode(["warm up"])
    return m


@st.cache_resource(show_spinner="Loading chunks...")
def load_chunks(cname):
    with open(f"results/chunks_{cname}.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index(cname, mkey):
    return faiss.read_index(f"results/indexes/{cname}_{mkey}.faiss")


@st.cache_data
def load_questions():
    try:
        with open("data/questions/questions_all.csv", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


@st.cache_data
def load_summary():
    try:
        return pd.read_csv("results/retrieval_summary.csv")
    except FileNotFoundError:
        return None


def retrieve(question, cname, mkey, k):
    model, chunks, index = load_model(mkey), load_chunks(cname), load_index(cname, mkey)
    t0 = time.perf_counter()
    qv = model.encode([question], normalize_embeddings=True)
    scores, idx = index.search(np.asarray(qv, dtype="float32"), k)
    ms = (time.perf_counter() - t0) * 1000
    results = [dict(chunks[i], score=float(s)) for s, i in zip(scores[0], idx[0])]
    return results, ms


def hit_rank(results, expected):
    if not expected:
        return None
    for r, c in enumerate(results, 1):
        if c["source"] == expected["source"] and c["page"] == int(expected["page"]):
            return r
    return 0


# ---------- sidebar ----------
st.sidebar.title("Settings")
cname = st.sidebar.selectbox("Chunking strategy", CHUNKINGS, format_func=CHUNK_LABELS.get)
mkey = st.sidebar.selectbox("Embedding model", list(MODELS), format_func=MODEL_LABELS.get)
k = st.sidebar.slider("Top-K chunks", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Corpus: 8 course PDFs (1,099 pages) · Index: FAISS inner product on "
    "normalised embeddings · Evaluated on 69 questions."
)

# ---------- header ----------
st.title("🔎 Chunking & Embedding Comparison for RAG Retrieval")
st.write(
    "Type a question about the course notes, or pick one from the evaluation set. "
    "The app retrieves the most relevant chunks and shows where they came from."
)

questions = load_questions()
options = ["(type my own question)"] + [f"[{q['set']}] {q['question']}" for q in questions]
choice = st.selectbox("Evaluation question (optional)", options)
expected = None
if choice != options[0]:
    expected = questions[options.index(choice) - 1]
    default_q = expected["question"]
else:
    default_q = ""

question = st.text_input("Question", value=default_q,
                         placeholder="e.g. What is the kernel trick in SVMs?")

tab1, tab2, tab3 = st.tabs(["Retrieve", "Compare all 6 methods", "Benchmark results"])

# ---------- tab 1: single method ----------
with tab1:
    if question:
        results, ms = retrieve(question, cname, mkey, k)
        c1, c2, c3 = st.columns(3)
        c1.metric("Method", f"{cname} + {mkey}")
        c2.metric("Retrieval latency", f"{ms:.1f} ms")
        rank = hit_rank(results, expected)
        if rank is None:
            c3.metric("Expected page", "n/a")
        else:
            c3.metric("Expected page found at", f"rank {rank}" if rank else "not in top-K")
            st.caption(f"Expected answer location: {expected['source']}, page {expected['page']}")
        for r, c in enumerate(results, 1):
            is_hit = expected and c["source"] == expected["source"] and c["page"] == int(expected["page"])
            tag = " ✅ expected page" if is_hit else ""
            with st.expander(f"#{r} · {c['source']} · p{c['page']} · score {c['score']:.3f}{tag}",
                             expanded=(r == 1)):
                st.write(c["text"])
    else:
        st.info("Enter a question above to retrieve chunks.")

# ---------- tab 2: compare all ----------
with tab2:
    if question:
        rows = []
        for mk in MODELS:
            for cn in CHUNKINGS:
                res, ms = retrieve(question, cn, mk, k)
                rk = hit_rank(res, expected)
                rows.append({
                    "method": f"{cn} + {mk}",
                    "top-1 source": f"{res[0]['source']} p{res[0]['page']}",
                    "top-1 score": round(res[0]["score"], 3),
                    "expected page rank": ("-" if rk is None else (rk if rk else "miss")),
                    "latency (ms)": round(ms, 1),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Same question, all six chunking × embedding combinations, top-K as set in the sidebar.")
    else:
        st.info("Enter a question above to compare all methods.")

# ---------- tab 3: benchmark ----------
with tab3:
    df = load_summary()
    if df is None:
        st.warning("results/retrieval_summary.csv not found. Run src/evaluate_retrieval.py first.")
    else:
        df = df.copy()
        df["method"] = df["chunking"] + " + " + df["embedding"]
        st.subheader("Retrieval benchmark (69 questions, K = 5)")
        st.dataframe(df.drop(columns=["method"]), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("hit@1 / hit@5 / MRR")
            st.bar_chart(df.set_index("method")[["hit@1", "hit@5", "MRR"]])
        with c2:
            st.caption("Mean query latency (ms)")
            st.bar_chart(df.set_index("method")[["latency_ms"]])
        st.markdown(
            "**Takeaways:** hit@5 is 0.90–0.94 for every method and the 95% confidence intervals "
            "overlap, so no method is clearly better at finding the right page. Fixed-size + MiniLM "
            "ranks the right page first most often (hit@1 0.78, MRR 0.84) and is among the fastest. "
            "BGE costs about 70% more latency without a clear gain."
        )
