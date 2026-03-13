from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import threading

from config import DOCS_FOLDER, WORDS_PER_CHUNK, OVERLAP_WORDS, EMBED_MODEL, FETCH_COUNT

_encoder  = None
_index    = None
_all_text = None
_origins  = None
_mutex    = threading.Lock()


def _read_docs(folder):
    loaded = []
    if not os.path.isdir(folder):
        return loaded
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        ext   = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            loaded.extend(PyPDFLoader(fpath).load())
        elif ext == ".txt":
            loaded.extend(TextLoader(fpath, encoding="utf-8").load())
    return loaded


def _build_index():
    global _encoder, _index, _all_text, _origins

    if _index is not None:
        return

    with _mutex:
        if _index is not None:
            return

        docs = _read_docs(DOCS_FOLDER)
        if not docs:
            raise RuntimeError(f"No documents found in '{DOCS_FOLDER}' folder.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=WORDS_PER_CHUNK,
            chunk_overlap=OVERLAP_WORDS
        )
        pieces = splitter.split_documents(docs)

        _encoder  = SentenceTransformer(EMBED_MODEL)
        _all_text = [p.page_content for p in pieces]
        _origins  = [p.metadata.get("source", "unknown") for p in pieces]

        vecs = np.array(_encoder.encode(_all_text, show_progress_bar=False), dtype=np.float32)
        _index = faiss.IndexFlatL2(vecs.shape[1])
        _index.add(vecs)


def retrieve(question, k=FETCH_COUNT):
    _build_index()
    q_vec = np.array(_encoder.encode([question], show_progress_bar=False), dtype=np.float32)
    dists, idxs = _index.search(q_vec, k=k)
    hits = []
    for d, i in zip(dists[0], idxs[0]):
        if i != -1:
            hits.append({"text": _all_text[i], "source": _origins[i], "distance": float(d)})
    return hits


def reset():
    global _encoder, _index, _all_text, _origins
    _encoder = _index = _all_text = _origins = None
