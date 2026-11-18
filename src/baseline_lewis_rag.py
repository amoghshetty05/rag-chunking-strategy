"""
Baseline: the original RAG retriever setup of Lewis et al. (NeurIPS 2020),
run on our corpus and our 69 questions so it can be compared with our 6 pipelines.

What Lewis et al. used, and how it is reproduced here:
  - Documents split into disjoint 100-word passages   -> same (per page, so page labels still apply)
  - DPR bi-encoder: separate BERT-base question and    -> facebook/dpr-question_encoder-multiset-base
    passage encoders, 768-d, trained on NQ + TriviaQA     facebook/dpr-ctx_encoder-multiset-base
  - Maximum inner product search with FAISS            -> exact IndexFlatIP on raw (unnormalised) vectors
  - Top-k passages passed to the generator             -> top-5 scored with the same metrics as our pipelines

Run from the project root:  python src/baseline_lewis_rag.py
First run downloads two ~440 MB models and encodes all passages (several minutes on CPU); later runs use the cache.
"""
import csv
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizerFast,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast)

from chunking import load_all

K = 5
WORDS = 100
CTX_MODEL = "facebook/dpr-ctx_encoder-multiset-base"
Q_MODEL = "facebook/dpr-question_encoder-multiset-base"
CHUNKS = Path("results/chunks_lewis100.pkl")
INDEX = Path("results/indexes/lewis_dpr.faiss")
Path("results/indexes").mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)


def hundred_word_passages(docs):
    """Disjoint 100-word passages, as in Lewis et al., kept within each page."""
    out = []
    for d in docs:
        words = d.page_content.split()
        for i in range(0, len(words), WORDS):
            text = " ".join(words[i:i + WORDS])
            if text.strip():
                out.append({"text": text,
                            "source": Path(d.metadata["source"]).name,
                            "page": int(d.metadata["page"]) + 1})
    return out


# ---------- 1) passages + index (cached) ----------
if CHUNKS.exists() and INDEX.exists():
    chunks = pickle.load(open(CHUNKS, "rb"))
    index = faiss.read_index(str(INDEX))
    print(f"Loaded cached passages: {len(chunks)}")
else:
    chunks = hundred_word_passages(load_all())
    print(f"Passages (100 words): {len(chunks)} - encoding with DPR, this takes a few minutes...")
    c_tok = DPRContextEncoderTokenizerFast.from_pretrained(CTX_MODEL)
    c_enc = DPRContextEncoder.from_pretrained(CTX_MODEL).eval()
    vecs = []
    for i in range(0, len(chunks), 32):
        batch = [c["text"] for c in chunks[i:i + 32]]
        inp = c_tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        vecs.append(c_enc(**inp).pooler_output.numpy())
        if (i // 32) % 20 == 0:
            print(f"  encoded {min(i + 32, len(chunks))}/{len(chunks)}")
    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    pickle.dump(chunks, open(CHUNKS, "wb"))
    faiss.write_index(index, str(INDEX))

# ---------- 2) evaluate on the same 69 questions ----------
q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(Q_MODEL)
q_enc = DPRQuestionEncoder.from_pretrained(Q_MODEL).eval()
q_enc(**q_tok(["warm up"], return_tensors="pt"))

questions = list(csv.DictReader(open("data/questions/questions_all.csv", encoding="utf-8-sig")))
ranks, precs, times, details = [], [], [], []
for q in questions:
    t0 = time.perf_counter()
    qv = q_enc(**q_tok([q["question"]], truncation=True, max_length=64, return_tensors="pt")).pooler_output.numpy()
    _, idx = index.search(qv.astype("float32"), K)
    times.append((time.perf_counter() - t0) * 1000)
    hits = [chunks[i]["source"] == q["source"] and chunks[i]["page"] == int(q["page"]) for i in idx[0]]
    rank = hits.index(True) + 1 if any(hits) else 0
    ranks.append(rank)
    precs.append(sum(hits) / K)
    top = chunks[idx[0][0]]
    details.append({"combo": "lewis2020_dpr", "question": q["question"],
                    "expected": f'{q["source"]} p{q["page"]}', "first_hit_rank": rank,
                    "top1_got": f'{top["source"]} p{top["page"]}', "set": q.get("set", "")})

n = len(questions)
row = {"chunking": "100-word (Lewis 2020)", "embedding": "DPR multiset", "chunks": len(chunks),
       "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
       "hit@3": round(sum(1 <= r <= 3 for r in ranks) / n, 3),
       "hit@5": round(sum(r >= 1 for r in ranks) / n, 3),
       "MRR": round(sum(1 / r for r in ranks if r) / n, 3),
       "precision@5": round(float(np.mean(precs)), 3),
       "latency_ms": round(float(np.mean(times)), 1)}
pd.DataFrame([row]).to_csv("results/baseline_lewis_summary.csv", index=False)
pd.DataFrame(details).to_csv("results/baseline_lewis_details.csv", index=False)

# ---------- 3) side-by-side with our six pipelines ----------
ours = pd.read_csv("results/retrieval_summary.csv")
table = pd.concat([pd.DataFrame([row]), ours], ignore_index=True).sort_values("MRR", ascending=False)
pd.set_option("display.width", 200)
print("\nBaseline (Lewis et al. 2020 retriever) vs our pipelines, 69 questions, K = 5:\n")
print(table.to_string(index=False))

# per-set breakdown for the baseline
d = pd.DataFrame(details)
for s in ["A", "B"]:
    sub = d[d["set"] == s]
    if len(sub):
        print(f"Baseline set {s}: hit@5 = {(sub.first_hit_rank > 0).mean():.3f}, "
              f"MRR = {sub.first_hit_rank.apply(lambda r: 1 / r if r else 0).mean():.3f}")
print("\nSaved: results/baseline_lewis_summary.csv and results/baseline_lewis_details.csv")
