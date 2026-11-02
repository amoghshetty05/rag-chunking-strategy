import csv
import random
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

random.seed(42)
rows = []
for pdf in sorted(Path("data/raw").glob("*.pdf")):
    pages = PyPDFLoader(str(pdf)).load()
    good = [p for p in pages if len(p.page_content) > 600]
    picks = random.sample(good, min(4, len(good)))
    for p in sorted(picks, key=lambda x: x.metadata["page"]):
        rows.append({
            "source": pdf.name,
            "page": p.metadata["page"] + 1,
            "question": "",
            "answer": "",
            "page_text": p.page_content[:1500].replace("\n", " "),
        })

with open("data/questions/questions_draft.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["source", "page", "question", "answer", "page_text"])
    w.writeheader()
    w.writerows(rows)

print("Wrote", len(rows), "rows")