import csv
import time
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

MODELS = {"minilm": "all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}
CHUNKINGS = ["fixed", "sentence", "semantic"]
K = 5

questions = list(csv.DictReader(open("data/questions/questions_all.csv", encoding="utf-8-sig")))
print("Questions loaded:", len(questions))

summary, details = [], []

for mkey, mname in MODELS.items():
    model = SentenceTransformer(mname)
    model.encode(["warm up"])  # so the first timing isn't unfairly slow

    for cname in CHUNKINGS:
        chunks = pickle.load(open(f"results/chunks_{cname}.pkl", "rb"))
        index = faiss.read_index(f"results/indexes/{cname}_{mkey}.faiss")
        assert index.ntotal == len(chunks), f"Index/chunk mismatch for {cname}_{mkey}"

        ranks, precs, times = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            qv = model.encode([q["question"]], normalize_embeddings=True)
            _, idx = index.search(np.asarray(qv, dtype="float32"), K)
            times.append((time.perf_counter() - t0) * 1000)

            hits = [
                chunks[i]["source"] == q["source"] and chunks[i]["page"] == int(q["page"])
                for i in idx[0]
            ]
            rank = hits.index(True) + 1 if any(hits) else 0
            ranks.append(rank)
            precs.append(sum(hits) / K)
            top = chunks[idx[0][0]]
            details.append({
                "combo": f"{cname}_{mkey}",
                "question": q["question"],
                "expected": f'{q["source"]} p{q["page"]}',
                "first_hit_rank": rank,
                "top1_got": f'{top["source"]} p{top["page"]}',
            })

        n = len(questions)
        summary.append({
            "chunking": cname,
            "embedding": mkey,
            "chunks": len(chunks),
            "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
            "hit@3": round(sum(1 for r in ranks if 1 <= r <= 3) / n, 3),
            "hit@5": round(sum(1 for r in ranks if r >= 1) / n, 3),
            "MRR": round(sum(1 / r for r in ranks if r) / n, 3),
            "precision@5": round(float(np.mean(precs)), 3),
            "latency_ms": round(float(np.mean(times)), 1),
        })

df = pd.DataFrame(summary).sort_values("hit@5", ascending=False)
df.to_csv("results/retrieval_summary.csv", index=False)
pd.DataFrame(details).to_csv("results/retrieval_details.csv", index=False)

pd.set_option("display.width", 200)
print()
print(df.to_string(index=False))
print("\nSaved: results/retrieval_summary.csv and results/retrieval_details.csv")