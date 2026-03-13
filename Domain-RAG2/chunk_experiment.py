import os
import time
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import DOCS_FOLDER

CONFIGS = [
    {"size": 300, "overlap": 0,   "label": "300 no-overlap"},
    {"size": 300, "overlap": 50,  "label": "300 overlap-50"},
    {"size": 500, "overlap": 50,  "label": "500 overlap-50"},
    {"size": 800, "overlap": 0,   "label": "800 no-overlap"},
    {"size": 800, "overlap": 100, "label": "800 overlap-100"},
]

MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

TEST_QUERIES = [
    "What is the leave policy?",
    "How do I request time off?",
    "What are the working hours?",
    "What is the dress code?",
    "How is performance evaluated?",
]


def load_docs():
    docs = []
    for fname in sorted(os.listdir(DOCS_FOLDER)):
        path = os.path.join(DOCS_FOLDER, fname)
        ext  = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif ext == ".txt":
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def run():
    docs = load_docs()
    if not docs:
        print(f"No documents in {DOCS_FOLDER}/ — add PDFs or TXTs first.")
        return

    print("=" * 65)
    print("CHUNK & EMBEDDING EXPERIMENT")
    print("=" * 65)

    for model_name in MODELS:
        print(f"\nModel: {model_name}")
        print("─" * 65)
        encoder = SentenceTransformer(model_name)

        for cfg in CONFIGS:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["size"], chunk_overlap=cfg["overlap"]
            )
            chunks = splitter.split_documents(docs)
            texts  = [c.page_content for c in chunks]

            vecs = np.array(encoder.encode(texts, show_progress_bar=False), dtype=np.float32)
            idx  = faiss.IndexFlatL2(vecs.shape[1])
            idx.add(vecs)

            print(f"\n  [{cfg['label']}] — {len(texts)} chunks")
            total_dist = total_ms = 0

            for q in TEST_QUERIES:
                qv  = np.array(encoder.encode([q], show_progress_bar=False), dtype=np.float32)
                t0  = time.perf_counter()
                D, I = idx.search(qv, k=3)
                ms  = (time.perf_counter() - t0) * 1000
                avg = float(np.mean([d for d, i in zip(D[0], I[0]) if i != -1] or [0]))
                total_dist += avg
                total_ms   += ms
                print(f"    {q[:48]:<48} dist={avg:.4f} {ms:.1f}ms")

            n = len(TEST_QUERIES)
            print(f"    → avg dist={total_dist/n:.4f}  avg time={total_ms/n:.1f}ms")

    print("\n" + "=" * 65)
    print("Done. Lower distance = better match.")


if __name__ == "__main__":
    run()
