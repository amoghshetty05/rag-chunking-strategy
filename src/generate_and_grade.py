import os, json, time, pickle, csv, random
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"   # if you get a 404 error, see the note below
N_PER_SET = 10              # questions per set (A and B) -> 30 total
K = 5
DELAY = 15                    # seconds between calls (free-tier rate limit)

MODELS = {"minilm": "all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}
CHUNKINGS = ["fixed", "sentence", "semantic"]
CACHE = "results/llm_cache.json"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


import time, random

FALLBACKS = ["gemini-flash-latest", "gemini-flash-lite-latest"]  # tried if MODEL is overloaded

def ask(prompt, json_mode=False):
    models = [MODEL] + FALLBACKS
    for attempt in range(4):
        m = models[attempt % len(models)] if attempt >= 3 else MODEL
        try:
            r = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json" if json_mode else "text/plain",
                ),
            )
            return r.text
        except Exception as e:
            wait = min(90, 5 * (2 ** min(attempt, 4))) + random.random() * 3
            print(f"  retry {attempt+1} ({m}) - {str(e)[:60]} ... waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("Gemini kept failing")


# fixed sample of questions, the same for every combination
qs = list(csv.DictReader(open("data/questions/questions_all.csv", encoding="utf-8-sig")))
rng = random.Random(1)
sample = []
for s in ["A", "B"]:
    pool = [q for q in qs if q["set"] == s]
    sample += rng.sample(pool, min(N_PER_SET, len(pool)))
print("Questions used:", len(sample))

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

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
            qv = emb_model.encode([q["question"]], normalize_embeddings=True)
            _, idx = index.search(np.asarray(qv, dtype="float32"), K)
            ctx = "\n\n".join(f"[{i+1}] {chunks[j]['text']}" for i, j in enumerate(idx[0]))

            answer = ask(
                "Answer the question using ONLY the context below. If the context "
                "does not contain the answer, reply exactly: Not found in context.\n\n"
                f"Context:\n{ctx}\n\nQuestion: {q['question']}"
            ).strip()

            verdict = json.loads(ask(
                "You are grading a retrieval-augmented answer. Return JSON with two integer fields.\n"
                "faithful: 1 if every claim in the ANSWER is supported by the CONTEXT "
                "(an answer that says it was not found counts as faithful), else 0.\n"
                "correct: 1 if the ANSWER agrees with the REFERENCE answer, else 0.\n\n"
                f"QUESTION: {q['question']}\nREFERENCE: {q['answer']}\n\n"
                f"CONTEXT:\n{ctx}\n\nANSWER: {answer}\n\n"
                'Return only: {"faithful": 0 or 1, "correct": 0 or 1}',
                json_mode=True,
            ))
            cache[key] = {"combo": combo, "set": q["set"], "question": q["question"],
                          "answer": answer, "faithful": int(verdict["faithful"]),
                          "correct": int(verdict["correct"])}
            json.dump(cache, open(CACHE, "w"), indent=1)
            print(f"{combo} {n}/{len(sample)} faithful={verdict['faithful']} correct={verdict['correct']}")

df = pd.DataFrame(cache.values())
summary = df.groupby("combo")[["faithful", "correct"]].mean().round(3)
by_set = df.groupby(["combo", "set"])[["faithful", "correct"]].mean().round(3).unstack()
summary.to_csv("results/llm_summary.csv")
print("\nOverall:\n", summary)
print("\nBy question set:\n", by_set)