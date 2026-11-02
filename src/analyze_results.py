import numpy as np
import pandas as pd

d = pd.read_csv("results/retrieval_details.csv")
d["hit5"] = (d.first_hit_rank >= 1).astype(int)
d["rr"] = d.first_hit_rank.apply(lambda r: 1 / r if r else 0)

piv = d.pivot(index="question", columns="combo", values="hit5")
expected = d.groupby("question").expected.first()

print("Questions missed by ALL 6 combos:")
for q in piv.index[piv.sum(axis=1) == 0]:
    print(" -", expected[q], "|", q)

print("\nQuestions missed by some combos:")
for q in piv.index[(piv.sum(axis=1) > 0) & (piv.sum(axis=1) < 6)]:
    print(" -", expected[q], "| missed by", int(6 - piv.loc[q].sum()), "combos |", q)

rng = np.random.default_rng(0)
n = len(piv)
boot = [rng.integers(0, n, n) for _ in range(2000)]
rr = d.pivot(index="question", columns="combo", values="rr")

print("\nBootstrap 95% confidence intervals (same resamples for every combo):")
for combo in piv.columns:
    h = [piv[combo].values[i].mean() for i in boot]
    m = [rr[combo].values[i].mean() for i in boot]
    print(f"{combo:18} hit@5 {piv[combo].mean():.3f} [{np.percentile(h,2.5):.2f}-{np.percentile(h,97.5):.2f}]"
          f"   MRR {rr[combo].mean():.3f} [{np.percentile(m,2.5):.2f}-{np.percentile(m,97.5):.2f}]")

qs = pd.read_csv("data/questions/questions_all.csv", encoding="utf-8-sig")
d["set"] = d.question.map(dict(zip(qs.question, qs.set)))
print("\nhit@5 by question set:")
print(d.groupby(["combo", "set"]).hit5.mean().unstack().round(3))
print("\nMRR by question set:")
print(d.groupby(["combo", "set"]).rr.mean().unstack().round(3))          