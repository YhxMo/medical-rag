# Medical Imaging RAG Follow-up Assistant

A LangChain-based retrieval augmented generation (RAG) demo for pulmonary nodule guideline question answering and structured follow-up recommendations.

## Features

- PDF ingestion and chunking for a local guideline knowledge base
- Chroma vector retrieval with HuggingFace embeddings
- DeepSeek/OpenAI-compatible chat completion backend
- Tool-style workflow: retrieve guideline evidence, classify nodule type, generate structured follow-up recommendation, and self-check citations
- Gradio web UI for RAG Q&A and follow-up recommendation generation
- Pytest coverage for ingestion, retrieval, generation prompts, tool wrappers, and configuration safety

## Public Repository Notes

This public version intentionally does not include API keys, PDF source files, generated vector databases, model checkpoints, or local runtime outputs.

Place your own guideline PDFs under `data/pdfs/`, then build the vector database from the app or by running `python ingest.py`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` locally and set:

```text
DEEPSEEK_API_KEY=your-deepseek-api-key
```

The `.env` file is ignored by git.

## Run

```powershell
python app.py
```

Open http://127.0.0.1:7860.

## Test

```powershell
python -m pytest -q
```

## Safety

Generated answers are for education and demonstration only. They are not medical advice and cannot replace professional clinical judgment.
