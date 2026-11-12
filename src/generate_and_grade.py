"""
Answer generation + groundedness/correctness grading with a LOCAL model (Ollama).
No API key, no quota. Needs: Ollama installed, `ollama pull qwen2.5:3b`, `pip install ollama`.

For every chunking x embedding combination, the same sample of questions is:
  1. retrieved (top-K chunks from the FAISS index)
  2. answered by the local LLM using ONLY those chunks
  3. graded by the same LLM: faithful (supported by context) and correct (matches reference)
Progress is cached, so you can stop (Ctrl+C) and rerun at any time.
"""
import os, json, time, pickle, csv, random, re
import numpy as np
import faiss
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer

LLM = "qwen2.5:3b"          # use "qwen2.5:7b" if you have 16 GB+ RAM (better, slower)
N_PER_SET = 15              # questions per set (A and B) -> 30 total
K = 5

MODELS = {"minilm": "all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}
CHUNKINGS = ["fixed", "sentence", "semantic"]
CACHE = "results/llm_cache_ollama.json"
SUMMARY = "results/llm_summary.csv"


def ask(prompt, json_mode=False):
    for attempt in range(3):
        try:
            r = ollama.chat(
                model=LLM,
                messages=[{"role": "user", "content": prompt}],
                format="json" if json_mode else "",
                options={"temperature": 0, "seed": 42},
            )
            return r["message"]["content"]
        except Exception as e:
            print(f"  ollama error ({e}); is the Ollama app running? retrying in 5s")
            time.sleep(5)
    raise RuntimeError("Ollama not responding. Open the Ollama app and rerun.")


def to01(v):
    """Small models sometimes return "1", true, "yes" etc. Normalise to 0/1."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v >= 1 else 0
    return 1 if str(v).strip().lower() in {"1", "true", "yes"} else 0


def grade(question, reference, ctx, answer):
    prompt = (
        "You are grading a retrieval-augmented answer. Return JSON with two integer fields.\n"
        "faithful: 1 if every claim in the ANSWER is supported by the CONTEXT "
        "(an answer that says it was not found counts as faithful), else 0.\n"
        "correct: 1 if the ANSWER agrees with the REFERENCE answer, else 0.\n\n"
        f"QUESTION: {question}\nREFERENCE: {reference}\n\n"
        f"CONTEXT:\n{ctx}\n\nANSWER: {answer}\n\n"
        'Return only: {"faithful": 0 or 1, "correct": 0 or 1}'
    )
    raw = ask(prompt, json_mode=True)
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        v = json.loads(m.group(0)) if m else {}
    return to01(v.get("faithful", 0)), to01(v.get("correct", 0))


# same fixed sample of questions for every combination
qs = list(csv.DictReader(open("data/questions/questions_all.csv", encoding="utf-8-sig")))
rng = random.Random(1)
sample = []
for s in ["A", "B"]:
    pool = [q for q in qs if q["set"] == s]
    sample += rng.sample(pool, min(N_PER_SET, len(pool)))
print(f"LLM: {LLM} | questions used: {len(sample)}")

cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

for mkey, mname in MODELS.items():
    emb_model = SentenceTransformer(mname)
    for cname in CHUNKINGS:
        combo = f"{cname}_{mkey}"
        chunks = pickle.load(open(f"results/chunks_{cname}.pkl", "rb"))
        index = faiss.read_index(f"results/indexes/{combo}.faiss")
        for n, q in enumerate(sample, 1):
            key = f"{combo}|{q['question']}"
            if key in cache:
                continue
            t0 = time.time()
            qv = emb_model.encode([q["question"]], normalize_embeddings=True)
            _, idx = index.search(np.asarray(qv, dtype="float32"), K)
            ctx = "\n\n".join(f"[{i+1}] {chunks[j]['text']}" for i, j in enumerate(idx[0]))

            answer = ask(
                "Answer the question using ONLY the context below. If the context "
                "does not contain the answer, reply exactly: Not found in context.\n\n"
                f"Context:\n{ctx}\n\nQuestion: {q['question']}"
            ).strip()

            faithful, correct = grade(q["question"], q["answer"], ctx, answer)
            cache[key] = {
                "combo": combo, "set": q["set"], "question": q["question"],
                "answer": answer, "faithful": faithful, "correct": correct,
                "not_found": int("not found in context" in answer.lower()),
                "llm": LLM,
            }
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1, ensure_ascii=False)
            print(f"{combo} {n}/{len(sample)} faithful={faithful} correct={correct} "
                  f"({time.time() - t0:.0f}s)")

# summary: only rows from this LLM and this question sample
sample_qs = {q["question"] for q in sample}
df = pd.DataFrame(cache.values())
df = df[(df["llm"] == LLM) & (df["question"].isin(sample_qs))]
summary = df.groupby("combo")[["faithful", "correct", "not_found"]].mean().round(3)
summary["n"] = df.groupby("combo").size()
by_set = df.groupby(["combo", "set"])[["faithful", "correct"]].mean().round(3).unstack()
summary.to_csv(SUMMARY)
print("\nOverall:\n", summary)
print("\nBy question set:\n", by_set)
print(f"\nSaved: {SUMMARY} and {CACHE}")