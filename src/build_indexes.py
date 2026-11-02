import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from chunking import load_all, fixed_chunks, sentence_chunks, semantic_chunks

Path("results/indexes").mkdir(parents=True, exist_ok=True)

MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}

# 1) Build the 3 chunk sets (semantic is slow, so we cache it)
docs = load_all()
chunk_sets = {}
for name, fn in [("fixed", fixed_chunks), ("sentence", sentence_chunks), ("semantic", semantic_chunks)]:
    cache = Path(f"results/chunks_{name}.pkl")
    if cache.exists():
        chunk_sets[name] = pickle.load(open(cache, "rb"))
    else:
        print("Chunking:", name)
        chunks = fn(docs)
        chunk_sets[name] = [
            {"text": c.page_content, "source": Path(c.metadata["source"]).name, "page": c.metadata["page"] + 1}
            for c in chunks
        ]
        pickle.dump(chunk_sets[name], open(cache, "wb"))
    print(name, "->", len(chunk_sets[name]), "chunks")

# 2) Embed each chunk set with each model and save a FAISS index
for mkey, mname in MODELS.items():
    model = SentenceTransformer(mname)
    for cname, chunks in chunk_sets.items():
        out = Path(f"results/indexes/{cname}_{mkey}.faiss")
        if out.exists():
            continue
        print("Embedding:", cname, "x", mkey)
        emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(np.asarray(emb, dtype="float32"))
        faiss.write_index(index, str(out))

print("DONE. Built indexes:")
for f in sorted(Path("results/indexes").glob("*.faiss")):
    print(" ", f.name)