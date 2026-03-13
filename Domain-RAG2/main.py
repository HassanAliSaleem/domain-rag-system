import os
import time
import logging
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_PATH, LOG_LEVEL
from rag_pipeline import retrieve
from llm_generator import generate_answer, generate_answer_stream

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = FastAPI(title="Domain RAG System", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

ui = Jinja2Templates(directory="templates")
history = deque(maxlen=100)


@app.middleware("http")
async def track_latency(req: Request, call_next):
    t = time.perf_counter()
    res = await call_next(req)
    ms = (time.perf_counter() - t) * 1000
    res.headers["X-Latency-Ms"] = f"{ms:.2f}"
    log.info("%s %s — %.2f ms", req.method, req.url.path, ms)
    return res


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return ui.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "powered_by": "Claude"}


@app.get("/ask")
def ask(query: str):
    if not query.strip():
        return {"error": "Question cannot be empty."}

    start   = time.perf_counter()
    hits    = retrieve(query)
    result  = generate_answer(query, hits)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    entry = {
        "query":      query,
        "answer":     result["answer"],
        "citations":  result["citations"],
        "confidence": result["confidence"],
        "latency_ms": elapsed,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    history.append(entry)
    log.info("Q: %s | conf=%s | %s ms", query[:60], result["confidence"], elapsed)

    return entry


@app.get("/ask/stream")
def ask_stream(query: str):
    if not query.strip():
        return {"error": "Question cannot be empty."}
    hits = retrieve(query)

    def stream():
        for token in generate_answer_stream(query, hits):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/history")
def get_history():
    return {"total": len(history), "history": list(history)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
