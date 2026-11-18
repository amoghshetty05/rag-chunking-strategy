"""
RAG Chunking & Embedding Comparison - live demo
Run from the project folder:  streamlit run src/app.py
Includes the base-paper retriever (Lewis et al. 2020: 100-word passages + DPR) as a baseline;
its passages and index are built by src/baseline_lewis_rag.py.
"""
from pathlib import Path
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

BASE_CHUNKS = "results/chunks_lewis100.pkl"
BASE_INDEX = "results/indexes/lewis_dpr.faiss"
DPR_Q_MODEL = "facebook/dpr-question_encoder-multiset-base"
HAS_BASELINE = Path(BASE_CHUNKS).exists() and Path(BASE_INDEX).exists()
RECENT = {
    "amiri_r100": "Amiri & Bocklitz 2025 (100 tokens, no overlap)",
    "stabler_s512": "Stabler et al. 2025 (sentences, 512-token window, 200 overlap)",
}
HAS_RECENT = all(Path(f"results/indexes/{c}_{m}.faiss").exists() for c in RECENT for m in ("minilm", "bge"))
RECENT_SHORT = {"amiri_r100": "Amiri & Bocklitz 2025", "stabler_s512": "Stabler et al. 2025"}


def label(cname, mkey):
    if cname == "lewis":
        return "Lewis 2020 baseline (100-word + DPR)"
    if cname in RECENT_SHORT:
        return f"{RECENT_SHORT[cname]} + {mkey}"
    return f"{cname} + {mkey}"


st.set_page_config(page_title="RAG Chunking Comparison", page_icon="🔎", layout="wide")


# ---------- cached loaders (load once, reuse on every click) ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model(key):
    m = SentenceTransformer(MODELS[key])
    m.encode(["warm up"])
    return m


@st.cache_resource(show_spinner="Loading DPR question encoder (base paper)...")
def load_dpr():
    import torch
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
    torch.set_grad_enabled(False)
    tok = DPRQuestionEncoderTokenizerFast.from_pretrained(DPR_Q_MODEL)
    enc = DPRQuestionEncoder.from_pretrained(DPR_Q_MODEL).eval()
    enc(**tok(["warm up"], return_tensors="pt"))
    return tok, enc


@st.cache_resource(show_spinner="Loading chunks...")
def load_chunks(cname):
    path = BASE_CHUNKS if cname == "lewis" else f"results/chunks_{cname}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index(cname, mkey):
    path = BASE_INDEX if cname == "lewis" else f"results/indexes/{cname}_{mkey}.faiss"
    return faiss.read_index(path)


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
        df = pd.read_csv("results/retrieval_summary.csv")
    except FileNotFoundError:
        return None
    df["method"] = df["chunking"] + " + " + df["embedding"]
    if Path("results/baseline_recent_summary.csv").exists():
        r = pd.read_csv("results/baseline_recent_summary.csv")
        r["method"] = [label(c, m) for c, m in zip(r["chunking"], r["embedding"])]
        df = pd.concat([df, r], ignore_index=True)
    if Path("results/baseline_lewis_summary.csv").exists():
        b = pd.read_csv("results/baseline_lewis_summary.csv")
        b["method"] = "Lewis 2020 baseline"
        df = pd.concat([df, b], ignore_index=True)
    return df


def retrieve(question, cname, mkey, k):
    chunks, index = load_chunks(cname), load_index(cname, mkey)
    if cname == "lewis":
        tok, enc = load_dpr()
        t0 = time.perf_counter()
        qv = enc(**tok([question], truncation=True, max_length=64, return_tensors="pt")).pooler_output.numpy()
    else:
        model = load_model(mkey)
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
modes = (["Our pipelines"] + (["Recent papers (2025)"] if HAS_RECENT else [])
         + (["Base paper: Lewis et al. 2020"] if HAS_BASELINE else []))
mode = st.sidebar.radio("Pipeline", modes)
if mode == "Our pipelines":
    cname = st.sidebar.selectbox("Chunking strategy", CHUNKINGS, format_func=CHUNK_LABELS.get)
    mkey = st.sidebar.selectbox("Embedding model", list(MODELS), format_func=MODEL_LABELS.get)
elif mode == "Recent papers (2025)":
    cname = st.sidebar.selectbox("Paper's best chunking setting", list(RECENT), format_func=RECENT.get)
    mkey = st.sidebar.selectbox("Embedding model", list(MODELS), format_func=MODEL_LABELS.get)
    st.sidebar.info(
        "Best chunking settings reported by two 2025 studies, run on our corpus and questions "
        "with our embedding models: Amiri & Bocklitz (arXiv:2506.17277) and Stabler et al. (COINS 2025)."
    )
else:
    cname, mkey = "lewis", "dpr"
    st.sidebar.info(
        "Original RAG retriever (Lewis et al., NeurIPS 2020): disjoint 100-word passages, "
        "DPR question and passage encoders (BERT-base, 768-d), FAISS inner-product search."
    )
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
if expected and question.strip() != expected["question"].strip():
    expected = None  # user edited the question, so the expected page no longer applies

tab1, tab2, tab3 = st.tabs(["Retrieve", "Compare all pipelines", "Benchmark results"])

# ---------- tab 1: single method ----------
with tab1:
    if question:
        results, ms = retrieve(question, cname, mkey, k)
        c1, c2, c3 = st.columns(3)
        c1.metric("Method", label(cname, mkey))
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
        combos = [(cn, mk) for mk in MODELS for cn in CHUNKINGS]
        if HAS_RECENT:
            combos += [(cn, mk) for mk in MODELS for cn in RECENT]
        if HAS_BASELINE:
            combos.append(("lewis", "dpr"))
        for cn, mk in combos:
                res, ms = retrieve(question, cn, mk, k)
                rk = hit_rank(res, expected)
                rows.append({
                    "method": label(cn, mk),
                    "top-1 source": f"{res[0]['source']} p{res[0]['page']}",
                    "top-1 score": round(res[0]["score"], 3),
                    "expected page rank": ("-" if rk is None else (str(rk) if rk else "miss")),
                    "latency (ms)": round(ms, 1),
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Same question through our six pipelines, the 2025 paper settings and the Lewis et al. 2020 baseline (where built), top-K as set in the sidebar.")
    else:
        st.info("Enter a question above to compare all methods.")

# ---------- tab 3: benchmark ----------
with tab3:
    df = load_summary()
    if df is None:
        st.warning("results/retrieval_summary.csv not found. Run src/evaluate_retrieval.py first.")
    else:
        df = df.copy()
        st.subheader("Retrieval benchmark (69 questions, K = 5)")
        st.dataframe(df.drop(columns=["method"]), width="stretch", hide_index=True)
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
            "BGE costs about 66% more latency without a clear gain. Of the 2025 paper settings, Amiri & Bocklitz's "
            "100-token chunks with MiniLM reach the highest hit@5 (0.957, significant vs Fixed + MiniLM, p = 0.03) "
            "with similar ranking quality. The Lewis et al. 2020 baseline (100-word passages + DPR) scores far lower "
            "(hit@5 0.594, MRR 0.423) and is about 4.7× slower. Latencies of the 2025 settings were measured in a "
            "separate run and are not directly comparable."
        )
