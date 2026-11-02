from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

nltk.download("punkt_tab", quiet=True)


def load_all():
    docs = []
    for pdf in Path("data/raw").glob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs


def fixed_chunks(docs, size=500, overlap=50):
    """Cut every 500 characters, no matter what."""
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def sentence_chunks(docs, per_chunk=5, overlap=1):
    """Group 5 sentences per chunk, overlapping by 1."""
    out = []
    step = per_chunk - overlap
    for d in docs:
        sents = sent_tokenize(d.page_content)
        for i in range(0, len(sents), step):
            text = " ".join(sents[i:i + per_chunk])
            if text.strip():
                out.append(Document(page_content=text, metadata=d.metadata))
    return out

from sentence_transformers import SentenceTransformer

_model = None


def semantic_chunks(docs, threshold=0.5, max_sents=10):
    """Start a new chunk when meaning shifts (similarity drops below threshold)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    out = []
    for d in docs:
        sents = [s for s in sent_tokenize(d.page_content) if s.strip()]
        if not sents:
            continue
        if len(sents) == 1:
            out.append(Document(page_content=sents[0], metadata=d.metadata))
            continue
        emb = _model.encode(sents, normalize_embeddings=True, batch_size=64)
        sims = (emb[:-1] * emb[1:]).sum(axis=1)
        cur = [sents[0]]
        for i, s in enumerate(sents[1:]):
            if (sims[i] < threshold and len(" ".join(cur)) >= 250) or len(cur) >= max_sents:
                out.append(Document(page_content=" ".join(cur), metadata=d.metadata))
                cur = [s]
            else:
                cur.append(s)
        out.append(Document(page_content=" ".join(cur), metadata=d.metadata))
    return out


if __name__ == "__main__":
    docs = load_all()
    print("Fixed-size chunks:", len(fixed_chunks(docs)))
    print("Sentence chunks:", len(sentence_chunks(docs)))

    test = docs[:50]  # only 50 pages for now, to keep the test quick
    sem = semantic_chunks(test)
    print("Semantic chunks (first 50 pages):", len(sem))
    print("--- Semantic sample ---")
    print(sem[10].page_content)
    print(sem[10].metadata)