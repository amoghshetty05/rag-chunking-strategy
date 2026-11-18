"""
Baselines from two 2025 papers, run on our corpus and our 69 questions with our two embedding models.

  1. Amiri & Bocklitz (2025), "Chunk Twice, Embed Once" (arXiv:2506.17277)
     Best configuration "R100-0": recursive token-based chunking, 100 tokens, no overlap.
  2. Stabler et al. (2025), "The Impact of Chunking Strategies on Domain-Specific
     Information Retrieval in RAG Systems" (COINS 2025)
     Best configuration: sentence splitting into 512-token windows with 200-token overlap.

Both are applied page by page (so every chunk keeps one page label, as in our pipelines),
embedded with all-MiniLM-L6-v2 and BGE-small-en-v1.5, indexed with FAISS IndexFlatIP on
normalised vectors and scored exactly like evaluate_retrieval.py.
Tokens are counted with the MiniLM (BERT WordPiece) tokenizer.

Run from the project root:  python src/baseline_recent_2025.py
"""
import csv
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from chunking import load_all

K = 5
MODELS = {"minilm": "all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}
CONFIGS = {
    "amiri_r100": "Amiri & Bocklitz 2025: recursive 100 tokens, 0 overlap",
    "stabler_s512": "Stabler et al. 2025: sentences, 512-token window, 200 overlap",
}
Path("results/indexes").mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
ntok = lambda s: len(tok.encode(s, add_special_tokens=False))


def meta(d):
    return {"source": Path(d.metadata["source"]).name, "page": int(d.metadata["page"]) + 1}


def amiri_r100(docs):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tok, chunk_size=100, chunk_overlap=0)
    out = []
    for d in docs:
        for t in splitter.split_text(d.page_content):
            if t.strip():
                out.append({"text": t, **meta(d)})
    return out


def stabler_s512(docs, window=512, overlap=200):
    out = []
    for d in docs:
        sents = [s for s in sent_tokenize(d.page_content) if s.strip()]
        lens = [ntok(s) for s in sents]
        i = 0
        while i < len(sents):
            j, total = i, 0
            while j < len(sents) and (total + lens[j] <= window or j == i):
                total += lens[j]
                j += 1
            out.append({"text": " ".join(sents[i:j]), **meta(d)})
            if j >= len(sents):
                break
            # next window starts with the trailing sentences that fit in the overlap
            k, back = j, 0
            while k - 1 > i and back + lens[k - 1] <= overlap:
                back += lens[k - 1]
                k -= 1
            i = k
    return out


# ---------- 1) chunk (cached) ----------
docs = None
chunk_sets = {}
for name, fn in [("amiri_r100", amiri_r100), ("stabler_s512", stabler_s512)]:
    cache = Path(f"results/chunks_{name}.pkl")
    if cache.exists():
        chunk_sets[name] = pickle.load(open(cache, "rb"))
    else:
        docs = docs or load_all()
        chunk_sets[name] = fn(docs)
        pickle.dump(chunk_sets[name], open(cache, "wb"))
    lens = [len(c["text"]) for c in chunk_sets[name]]
    print(f"{name}: {len(lens)} chunks, median {int(np.median(lens))} chars")

# ---------- 2) embed + index (cached) ----------
models = {}
for mkey, mname in MODELS.items():
    models[mkey] = SentenceTransformer(mname)
    models[mkey].encode(["warm up"])
    for cname, chunks in chunk_sets.items():
        out = Path(f"results/indexes/{cname}_{mkey}.faiss")
        if out.exists():
            continue
        print("Embedding:", cname, "x", mkey)
        emb = models[mkey].encode([c["text"] for c in chunks], normalize_embeddings=True,
                                  batch_size=64, show_progress_bar=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(np.asarray(emb, dtype="float32"))
        faiss.write_index(index, str(out))

# ---------- 3) evaluate on the same 69 questions ----------
questions = list(csv.DictReader(open("data/questions/questions_all.csv", encoding="utf-8-sig")))
summary, details = [], []
for mkey in MODELS:
    for cname, chunks in chunk_sets.items():
        index = faiss.read_index(f"results/indexes/{cname}_{mkey}.faiss")
        ranks, precs, times = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            qv = models[mkey].encode([q["question"]], normalize_embeddings=True)
            _, idx = index.search(np.asarray(qv, dtype="float32"), K)
            times.append((time.perf_counter() - t0) * 1000)
            hits = [chunks[i]["source"] == q["source"] and chunks[i]["page"] == int(q["page"]) for i in idx[0]]
            rank = hits.index(True) + 1 if any(hits) else 0
            ranks.append(rank)
            precs.append(sum(hits) / K)
            top = chunks[idx[0][0]]
            details.append({"combo": f"{cname}_{mkey}", "question": q["question"],
                            "expected": f'{q["source"]} p{q["page"]}', "first_hit_rank": rank,
                            "top1_got": f'{top["source"]} p{top["page"]}', "set": q.get("set", "")})
        n = len(questions)
        summary.append({"chunking": cname, "embedding": mkey, "chunks": len(chunks),
                        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
                        "hit@3": round(sum(1 <= r <= 3 for r in ranks) / n, 3),
                        "hit@5": round(sum(r >= 1 for r in ranks) / n, 3),
                        "MRR": round(sum(1 / r for r in ranks if r) / n, 3),
                        "precision@5": round(float(np.mean(precs)), 3),
                        "latency_ms": round(float(np.mean(times)), 1)})

pd.DataFrame(summary).to_csv("results/baseline_recent_summary.csv", index=False)
pd.DataFrame(details).to_csv("results/baseline_recent_details.csv", index=False)

# ---------- 4) everything side by side ----------
frames = [pd.read_csv("results/retrieval_summary.csv"), pd.DataFrame(summary)]
if Path("results/baseline_lewis_summary.csv").exists():
    frames.append(pd.read_csv("results/baseline_lewis_summary.csv"))
table = pd.concat(frames, ignore_index=True).sort_values("MRR", ascending=False)
pd.set_option("display.width", 200)
print("\nOur 6 pipelines vs 2025 paper settings vs Lewis et al. 2020 (69 questions, K = 5):\n")
print(table.to_string(index=False))
print("\nSaved: results/baseline_recent_summary.csv and results/baseline_recent_details.csv")
