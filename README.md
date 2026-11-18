# Chunking Strategy or Embedding Model? RAG Retrieval Benchmark

A controlled comparison of **3 chunking strategies × 2 embedding models** for retrieval-augmented generation (RAG), evaluated on 1,099 pages of university course notes.

Mini project for *Prompt Engineering and Optimization for Generative AI* (UE24AM343AB5), PES University.

**Team:** Amogh Shetty (PES1UG24AM034) · Madhumitha Dhondi (PES1UG24AM923)
**Guide:** Dr. Chetana Srinivas

---

## Research question

Every RAG system has to decide how to cut documents into chunks and which embedding model to use. Both choices are usually copied from tutorials. On the same corpus, index and question set, **how much do they actually change retrieval quality and speed?**

## Pipeline

```
8 course PDFs (1,099 pages)
   → chunk:  fixed-size | sentence-level | semantic
   → embed:  all-MiniLM-L6-v2 | BAAI/bge-small-en-v1.5
   → index:  FAISS IndexFlatIP (cosine on normalised vectors)
   → retrieve top-5 for 69 page-labelled questions
   → score:  hit@1/3/5, MRR, precision@5, latency + bootstrap statistics
```

| Chunking | Rule | Chunks |
|---|---|---|
| Fixed-size | 500 characters, 50 overlap | 2,028 |
| Sentence | 5 sentences, 1 overlap (NLTK Punkt) | 1,980 |
| Semantic | new chunk when adjacent-sentence similarity < 0.5 (min 250 chars, max 10 sentences) | 2,238 |

Chunking is done page by page, so every chunk carries one source page. A retrieved chunk is a **hit** when its source file and page match the question's label.

## Results (69 questions, K = 5)

| Pipeline | hit@1 | hit@3 | hit@5 | MRR | P@5 | Latency (ms) |
|---|---|---|---|---|---|---|
| **Fixed + MiniLM** | **0.783** | 0.884 | 0.899 | **0.837** | **0.386** | 14.4 |
| Semantic + MiniLM | 0.725 | 0.884 | 0.899 | 0.801 | 0.342 | 15.5 |
| Sentence + MiniLM | 0.638 | 0.855 | 0.899 | 0.744 | 0.351 | 14.3 |
| Fixed + BGE | 0.667 | 0.884 | 0.899 | 0.769 | 0.354 | 24.7 |
| Sentence + BGE | 0.667 | 0.870 | 0.899 | 0.773 | 0.313 | 23.7 |
| Semantic + BGE | 0.652 | 0.899 | **0.942** | 0.770 | 0.330 | 25.1 |

**Key findings**

- **Top-5 recall is a tie.** hit@5 is 0.90–0.94 for all six pipelines and the 95% bootstrap intervals overlap.
- **Ranking differs.** Fixed + MiniLM puts the correct page first 78% of the time. In paired bootstrap tests its hit@1 is significantly higher than 4 of the other 5 pipelines (p < 0.05).
- **BGE is ~66% slower** per query (≈24.5 ms vs ≈14.7 ms) with no measurable quality gain on this corpus.
- **Paraphrased questions were not harder** than near-verbatim ones.
- **Error analysis:** only 1 question was missed by every pipeline. Several misses are correct answers on a neighbouring page, a limit of strict page-level scoring.

## Comparison with published work

The best settings from three papers were run on the same corpus and 69 questions:

| Setting (paper) | Model | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| **Ours: Fixed 500 chars** | MiniLM | **0.783** | 0.899 | **0.837** |
| 100 tokens, no overlap (Amiri & Bocklitz 2025) | MiniLM | 0.710 | **0.957** | 0.812 |
| 100 tokens, no overlap (Amiri & Bocklitz 2025) | BGE | 0.667 | 0.928 | 0.775 |
| 512-token sentence windows, 200 overlap (Stabler et al. 2025) | MiniLM | 0.681 | 0.884 | 0.761 |
| 512-token sentence windows, 200 overlap (Stabler et al. 2025) | BGE | 0.667 | 0.855 | 0.746 |
| 100-word passages + DPR (Lewis et al. 2020, original RAG retriever) | DPR | 0.304 | 0.594 | 0.423 |

- 100-token chunks reach the highest top-5 recall (significant vs Fixed + MiniLM, p = 0.03); ranking (hit@1, MRR) is not significantly different.
- Long 512-token windows do worse, and the original 2020 RAG retriever does far worse on course notes (likely domain shift).

Scripts: `src/baseline_recent_2025.py` and `src/baseline_lewis_rag.py`. Both appear in the demo app's sidebar once run.

## Repository structure

```
src/
  load_docs.py            load the PDFs and check page counts
  chunking.py             fixed, sentence and semantic chunkers
  build_indexes.py        build and cache chunks + 6 FAISS indexes
  sample_pages.py         sample pages for question set A (seed 42)
  sample_pages2.py        sample pages for question set B (seed 7)
  evaluate_retrieval.py   run all questions on all 6 pipelines
  analyze_results.py      misses, bootstrap confidence intervals, per-set scores
  generate_and_grade.py   optional: answer + grade with a local LLM (Ollama)
  baseline_lewis_rag.py   baseline: Lewis et al. 2020 retriever (100-word + DPR)
  baseline_recent_2025.py baselines: best chunking settings of two 2025 papers
  app.py                  Streamlit demo
data/
  raw/                    course PDFs (not included, see below)
  questions/              questions_all.csv (69 questions: source, page, question, answer, set)
results/
  retrieval_summary.csv   one row per pipeline
  retrieval_details.csv   one row per question × pipeline
```

## Setup

Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install langchain langchain-community langchain-text-splitters pypdf nltk sentence-transformers faiss-cpu pandas numpy streamlit
python -c "import nltk; nltk.download('punkt_tab')"
```

Put the course PDFs in `data/raw/`. They are not in this repository because they are university course material.

## How to run

Run from the project root, in this order:

```bash
python src/build_indexes.py        # chunks all PDFs and builds the 6 indexes (semantic chunking takes a few minutes)
python src/evaluate_retrieval.py   # writes results/retrieval_summary.csv and retrieval_details.csv
python src/analyze_results.py      # misses, bootstrap CIs, scores by question set
streamlit run src/app.py           # interactive demo
```

### Demo app

- **Retrieve:** pick a chunking strategy and embedding model, ask a question, see the top chunks with source, page, score and latency.
- **Compare all 6:** one question through every pipeline side by side, with the rank of the expected page.
- **Benchmark:** the results table and charts.

### Optional: answer-level grading

`generate_and_grade.py` answers each question from the retrieved chunks and grades faithfulness and correctness with a local model through [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:3b
pip install ollama
python src/generate_and_grade.py
```

## Question set

- **Set A (32 questions):** 4 random pages per PDF, questions close to the page wording.
- **Set B (37 questions):** 5 further pages per PDF, deliberately paraphrased.

Each question is labelled with its source PDF and page. Questions were drafted with AI assistance from the page text and reviewed manually against the PDFs.

## Limitations

- 69 questions give wide confidence intervals; only differences of roughly 0.1 or more are detectable.
- One labelled page per question, so correct answers on adjacent pages count as misses.
- One corpus, two small (384-d) embedding models, one setting per chunker, no re-ranking or hybrid search.

## License

Code is for academic use as part of the PES University course project.
