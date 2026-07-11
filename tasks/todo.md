# Tasks: Medical Multimodal RAG MVP

- [x] Task 1: Bootstrap Python project skeleton and dependencies
  - Acceptance: `src/`, `tests/`, `config.yaml`, `requirements.txt`, and basic package files exist.
  - Verify: `python -m src.cli --help` runs and `pytest tests/test_config.py -v` passes.
  - Files: `src/cli.py`, `src/config/settings.py`, `config.yaml`, `requirements.txt`, `tests/test_config.py`

- [x] Task 2: Define shared data models
  - Acceptance: Text pages, text chunks, image assets, image captions, retrieval hits, and generated answers have typed dataclasses.
  - Verify: `pytest tests/test_models.py -v`
  - Files: `src/schema.py`, `tests/test_models.py`

- [x] Task 3: Implement PDF text loader
  - Acceptance: Each PDF page becomes a document object with source file, page number, extracted text, text length, and `needs_ocr=true` when extracted text length is under 50 characters.
  - Verify: `pytest tests/test_document_loader.py -v`
  - Files: `src/document/loader.py`, `tests/test_document_loader.py`

- [x] Task 4: Implement configurable chunking strategies
  - Acceptance: `layout_heading` is the default strategy, `fixed` is available as a baseline, and each chunk records strategy, heading path, page range, nearby image IDs, and metadata.
  - Verify: `pytest tests/test_chunker.py -v`
  - Files: `src/document/chunker.py`, `tests/test_chunker.py`

- [x] Task 5: Implement exercise book question-answer structuring
  - Acceptance: Exercise PDFs are split into question sections and answer/explanation sections, chapter + question number are recognized, `question_id` is built, answers are paired, and each question becomes exactly one chunk.
  - Verify: `pytest tests/test_exercise_parser.py -v`
  - Files: `src/document/exercise_parser.py`, `tests/test_exercise_parser.py`

- [x] Task 6: Implement PDF image/page extraction
  - Acceptance: Page screenshots and extracted images are cached under a generated artifact directory with source/page metadata.
  - Verify: `pytest tests/test_image_extractor.py -v`
  - Files: `src/document/image_extractor.py`, `tests/test_image_extractor.py`, `.gitignore`

- [x] Task 7: Implement local VLM captioner interface
  - Acceptance: A stub captioner works in tests, a local Qwen-VL provider can be enabled by config for real indexing, and `needs_ocr` pages can be sent through an OCR-style prompt.
  - Verify: `pytest tests/test_captioner.py -v`
  - Files: `src/vision/base.py`, `src/vision/stub.py`, `src/vision/qwen_vl.py`, `tests/test_captioner.py`

- [x] Task 8: Build unified evidence preparation
  - Acceptance: Text chunks from the selected chunk strategy, exercise QA chunks, and image captions are converted into one evidence list for indexing.
  - Verify: `pytest tests/test_evidence.py -v`
  - Files: `src/indexer/evidence.py`, `tests/test_evidence.py`

- [x] Task 9: Implement embedding provider
  - Acceptance: `BAAI/bge-small-zh-v1.5` provider embeds evidence text and query text; tests use a fake provider.
  - Verify: `pytest tests/test_embedding.py -v`
  - Files: `src/embedding/base.py`, `src/embedding/bge.py`, `tests/test_embedding.py`

- [x] Task 10: Implement FAISS and BM25 index persistence
  - Acceptance: Index build writes dense index, BM25 index, and evidence metadata to disk; reload returns same evidence IDs.
  - Verify: `pytest tests/test_index_store.py -v`
  - Files: `src/indexer/store.py`, `src/indexer/indexer.py`, `tests/test_index_store.py`

- [x] Task 11: Implement hybrid retriever
  - Acceptance: Query combines dense and BM25 hits with reciprocal rank fusion and returns cited evidence.
  - Verify: `pytest tests/test_retrieval.py -v`
  - Files: `src/retrieval/hybrid.py`, `src/retrieval/base.py`, `tests/test_retrieval.py`

- [x] Task 12: Implement optional lightweight reranker
  - Acceptance: Reranker can be enabled or disabled by config; fake reranker covers tests.
  - Verify: `pytest tests/test_reranker.py -v`
  - Files: `src/reranker/base.py`, `src/reranker/bge.py`, `tests/test_reranker.py`

- [x] Task 13: Implement DashScope generator
  - Acceptance: Generator builds a citation-constrained prompt and returns answer text with referenced evidence IDs.
  - Verify: `pytest tests/test_generator.py -v`
  - Files: `src/generator/generator.py`, `tests/test_generator.py`

- [x] Task 14: Implement end-to-end RAG pipeline
  - Acceptance: Pipeline runs retrieve -> rerank -> generate and returns answer plus source list.
  - Verify: `pytest tests/test_pipeline.py -v`
  - Files: `src/pipeline/pipeline.py`, `tests/test_pipeline.py`

- [x] Task 15: Implement CLI commands
  - Acceptance: `index`, `query`, `serve`, `evaluate`, and `generate-eval` commands exist.
  - Verify: `python -m src.cli --help`
  - Files: `src/cli.py`, `tests/test_cli.py`

- [x] Task 16: Implement Gradio UI
  - Acceptance: UI lets the user ask a question, inspect citations, and choose retrieval/reranker settings.
  - Verify: `python -m src.cli serve --config config.yaml`
  - Files: `src/ui/app.py`, `src/cli.py`

- [x] Task 17: Implement semi-automatic evaluation dataset generation
  - Acceptance: `generate-eval` creates draft questions from retrieved page/chunk evidence and marks them for review.
  - Verify: `python -m src.cli generate-eval --config config.yaml --output data/test_questions.json`
  - Files: `src/evaluation/dataset.py`, `src/evaluation/generator.py`, `tests/test_evaluation_dataset.py`

- [x] Task 18: Implement evaluation runner
  - Acceptance: Evaluation loads reviewed questions, runs the pipeline, and writes JSON/CSV metrics to `experiments/results/`, including the active chunk strategy in each run record.
  - Verify: `python -m src.cli evaluate --config config.yaml --output experiments/results`
  - Files: `src/evaluation/evaluator.py`, `tests/test_evaluator.py`

- [x] Task 19: Write README MVP workflow
  - Acceptance: README explains setup, environment variables, indexing, querying, UI, evaluation, and GPU notes for RTX 4060 Laptop.
  - Verify: Follow README commands from a fresh environment.
  - Files: `README.md`
