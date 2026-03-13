# Domain-Specific RAG System
### Mini ChatGPT for Company Documents — Powered by Claude

## Features
- PDF and TXT document support
- FAISS vector search
- Citations and confidence scoring
- Query history tracking
- Latency measurement
- Streaming responses
- Simple frontend UI

## Project Structure
```
Domain-RAG/
├── main.py              # FastAPI app
├── rag_pipeline.py      # Document loading + FAISS retrieval
├── llm_generator.py     # Claude answer generation
├── config.py            # All settings in one place
├── evaluation.py        # 30 test questions evaluation
├── chunk_experiment.py  # Chunk size comparison
├── requirements.txt
├── docs/                # Put your PDF/TXT files here
└── templates/
    └── index.html       # Frontend UI
```

## How to Run

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your documents to docs/ folder

# 3. Set API key
$env:ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run
python main.py
```

Open http://127.0.0.1:8000

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Frontend UI |
| `GET /ask?query=...` | Ask a question |
| `GET /ask/stream?query=...` | Streaming answer |
| `GET /history` | Query history |
| `GET /health` | Health check |
