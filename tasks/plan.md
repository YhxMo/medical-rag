# Plan: Medical Multimodal RAG MVP

## Scope

Build a lightweight, local-first multimodal RAG MVP for the 4 medical imaging textbooks in `data/`.

The MVP must support:
- PDF text extraction with page metadata.
- OCR candidate detection: if extracted page text has fewer than 50 characters, mark the page as `needs_ocr`.
- Configurable chunking strategies instead of fixed chunks only.
- Exercise book structuring: question section extraction, answer/explanation extraction, chapter + question number matching, and one-question-one-chunk evidence.
- PDF page/image extraction with local VLM-generated Chinese captions.
- Unified evidence indexing for text chunks and image captions.
- Hybrid retrieval with FAISS dense search + BM25 keyword search.
- Optional lightweight reranking.
- DashScope-backed answer generation with source citations.
- Gradio UI and CLI commands.
- Semi-automatic evaluation dataset generation and repeatable evaluation runs.

## Recommended Model Stack

| Module | MVP Choice | Reason |
|---|---|---|
| Local image understanding | `Qwen/Qwen3-VL-2B-Instruct` first, fallback `Qwen/Qwen2.5-VL-3B-Instruct` | Fits RTX 4060 Laptop better than 7B+ VLMs; used offline for caption/OCR-like evidence extraction. |
| Chunking | `layout_heading` default, `fixed` baseline, `parent_child` next | Preserves textbook sections, figure captions, tables, and diagnostic lists better than fixed windows. |
| Exercise chunks | `exercise_qa_pair` | Handles exercise books where questions and answers are stored in separate PDF sections. |
| Text embedding | `BAAI/bge-small-zh-v1.5` | Lightweight Chinese embedding; simple local deployment through sentence-transformers. |
| Keyword retrieval | BM25 + jieba | Strong exact-match behavior for Chinese medical terms and textbook phrasing. |
| Reranker | `BAAI/bge-reranker-base`, optional | Improves top-k quality without a heavy multimodal reranker. |
| Generator | DashScope Qwen chat model | User has API key; keeps generation off local GPU so the GPU can be reserved for image captioning. |

## Architecture

```text
PDF files
  |
  +--> PDF text loader
  |      |
  |      +--> page documents + text length + needs_ocr flag
  |              |
  |              +--> configurable chunker
  |                      |
  |                      +--> layout/heading-aware chunks
  |                      +--> parent-child chunks
  |                      +--> fixed baseline chunks
  |
  +--> Exercise book parser
         |
         +--> question section extraction
         +--> answer/explanation section extraction
         +--> chapter + question number matching
         +--> one-question-one-chunk evidence
  |
  +--> PDF image/page renderer
         |
         +--> local VLM captioner / OCR fallback for needs_ocr pages
                 |
                 +--> image caption evidence

text chunks + image caption evidence
  |
  +--> embedding provider
  |      |
  |      +--> FAISS dense index
  |
  +--> tokenizer
         |
         +--> BM25 index

query
  |
  +--> hybrid retriever
  |
  +--> optional reranker
  |
  +--> DashScope generator
  |
  +--> answer with citations
```

## Implementation Order

1. Bootstrap project skeleton, dependencies, config loader, and shared data models.
2. Implement PDF text extraction and chunking strategy interfaces with page-level metadata.
3. Implement exercise book question-answer structuring and one-question-one-chunk evidence.
4. Implement image/page extraction and cache extracted assets under an ignored generated directory.
5. Implement local VLM captioning behind a provider interface, with a deterministic stub for tests.
6. Implement local embedding, FAISS persistence, BM25 persistence, and index build command.
7. Implement hybrid retrieval and optional reranker.
8. Implement DashScope generation with strict citation formatting.
9. Implement CLI commands: `index`, `query`, `serve`, `evaluate`, `generate-eval`.
10. Implement Gradio MVP UI.
11. Implement semi-automatic evaluation dataset generation and metric reporting.

## Dependency Graph

```text
config + models
  -> document loader
  -> chunker
  -> exercise parser
  -> image extractor
  -> captioner
  -> evidence store
  -> embedding/indexer
  -> retriever
  -> reranker
  -> generator
  -> pipeline
  -> CLI/UI/evaluation
```

## Verification Checkpoints

| Checkpoint | Command | Expected Result |
|---|---|---|
| Config loads | `pytest tests/test_config.py -v` | YAML resolves environment variables and typed config. |
| PDF parsing works | `pytest tests/test_document_loader.py -v` | Text pages preserve source + page, record text length, and mark pages under 50 chars as `needs_ocr`. |
| Chunking works | `pytest tests/test_chunker.py -v` | `layout_heading` preserves headings/page metadata; `fixed` remains available as baseline. |
| Exercise structuring works | `pytest tests/test_exercise_parser.py -v` | Questions and answers are paired by chapter + question number; one question produces one chunk. |
| Captioning interface works | `pytest tests/test_captioner.py -v` | Stub captions produce image evidence without loading a VLM. |
| Index builds | `python -m src.cli index --config config.yaml` | FAISS/BM25/index metadata are written locally. |
| Query works | `python -m src.cli query "脑出血的CT表现是什么？" --config config.yaml` | Answer includes cited textbook page evidence. |
| Evaluation works | `python -m src.cli evaluate --config config.yaml --output experiments/results` | Metrics JSON/CSV are generated. |
| UI works | `python -m src.cli serve --config config.yaml` | Gradio launches and completes one chat turn. |

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Local VLM may exceed RTX 4060 Laptop VRAM | Use 4-bit loading first, limit image resolution, batch size 1, and cache captions so captioning is offline. |
| PDF image extraction may include decorative or low-value images | Store page-level screenshots plus extracted images; filter tiny images; keep source/page metadata. |
| PyMuPDF may extract too little text from scanned/image-heavy pages | Mark pages with extracted text length under 50 as `needs_ocr` and send them to the local VLM/OCR fallback path. |
| Medical image captions may be inaccurate | Treat captions as retrieval hints, not final diagnosis; final answer must cite source text/page and avoid unsupported claims. |
| Naive chunking may damage textbook semantics | Default to layout/heading-aware chunking; record chunk strategy in every evidence item and evaluation result. |
| Exercise question/answer sections may not align perfectly | Keep unmatched questions/answers in an extraction report instead of dropping them. |
| DashScope model names may differ by account availability | Keep model names configurable in `config.yaml`; fail with a clear error message. |
| Evaluation data may be noisy if fully generated | Semi-automatic generation plus manual review flag in `data/test_questions.json`. |

## Parallel Work

Can be developed in parallel after shared models/config exist:
- PDF text loader and chunker.
- Exercise parser once the shared document schema exists.
- Image extraction and captioner interface.
- Evaluation dataset schema.
- Gradio UI shell.

Must be sequential:
- Retrieval depends on evidence schema and index persistence.
- Exercise evidence depends on question-answer pairing output.
- Generation depends on retrieval result shape.
- Evaluation depends on query pipeline outputs.

## Phase Gate

After this plan is approved, implement tasks from `tasks/todo.md` in order. Each task should leave the project in a runnable or testable state.
