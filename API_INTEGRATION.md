# CareerLens RAG HTTP adapter

This adapter exposes the existing functions under `utils/` to the CareerLens web app. It does not replace the manual RAG pipeline.

## Run locally

```bash
cp .env.example .env
# Set GEMINI_API_KEY and RAG_API_KEY in .env
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
```

The website sends `POST /analyze` with the uploaded CV as base64 PDF content and the job description as text. The adapter calls the existing document loader, chunker, embeddings, Chroma retrieval, grounded generation, skill-gap, resume-coach, and tailored-resume functions.

For a deployed service, configure the web project with:

- `RAG_SERVICE_URL=https://your-rag-service.example.com`
- `RAG_SERVICE_API_KEY=<same value as RAG_API_KEY>`

Keep both values server-side and never expose them as `VITE_*` variables.
