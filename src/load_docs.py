from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

docs = []
for pdf in Path("data/raw").glob("*.pdf"):
    pages = PyPDFLoader(str(pdf)).load()
    docs.extend(pages)
    print(pdf.name, "->", len(pages), "pages")

print("Total pages loaded:", len(docs))
print("--- Sample ---")
print(docs[0].page_content[:300])