# Medical Imaging Learning RAG

Chinese medical-imaging retrieval-augmented generation project for learning,
review, and source-grounded search over local radiology textbooks and exercise
books.

This is an educational study and retrieval assistant. It is not a clinical
diagnosis system, does not read patient studies, and should not be used for
medical decision-making.

## Portfolio Highlights

- Built a multimodal RAG pipeline over 4 local PDF books with 5,410 indexed
  evidence records: 547 textbook text chunks, 3,080 structured exercise QA
  chunks, and 1,783 image-caption evidence records.
- Implemented PDF loading, RapidOCR fallback/cache, heading-aware chunking,
  structured exercise question-answer pairing, contextual image captioning,
  unified evidence serialization, Qdrant dense retrieval, BM25 sparse retrieval,
  reciprocal-rank fusion, and optional BGE reranking.
- Parsed 2 exercise books into 3,080 QA chunks with an estimated 94.9% pairing
  rate in the current parser report.
- Current source-grounded evaluation uses 30 reviewed questions and reports
  29/30 hits, recall@5 0.9667, MRR 0.9667, and NDCG@5 0.9667.
- Retrieval ablation on the same 30-question source-grounded set shows the
  hybrid + BGE reranker run at 30/30 hits with recall@5, MRR, and NDCG@5 all
  equal to 1.0. Treat this as a pipeline/regression signal, not a real-world
  clinical benchmark.
- A saved web-style radiology evaluation is harder and less source-shaped:
  22 mapped questions, 18/22 hits, recall@5 0.8182, MRR 0.5659, NDCG@5 0.5532.
- Test suite status in the project environment: 67 passed, 1 skipped.

## Current Index Snapshot

Current main index file: `artifacts/index/evidence.jsonl`.

| Evidence type | Count |
|---|---:|
| `text` | 547 |
| `exercise_qa` | 3,080 |
| `image_caption` | 1,783 |
| **Total** | **5,410** |

Source distribution:

| Source file | Evidence records |
|---|---:|
| `医学影像学.pdf` | 1,032 |
| `医学影像诊断学.pdf` | 1,298 |
| `医学影像学学习指导与习题集.pdf` | 1,852 |
| `医学影像诊断学习题集.pdf` | 1,228 |

Image-caption distribution:

| Image kind | Count |
|---|---:|
| Page image | 874 |
| Embedded image | 909 |

All current image captions were generated through the OpenAI-compatible vision
captioner path with model metadata `qwen3-vl-32b-instruct`.

Exercise parsing report:

| Exercise PDF | Chunks | Unmatched questions | Unmatched answers | Estimated pairing rate |
|---|---:|---:|---:|---:|
| `医学影像学学习指导与习题集.pdf` | 1,852 | 85 | 71 | 91.9% |
| `医学影像诊断学习题集.pdf` | 1,228 | 3 | 2 | 99.6% |
| **Total** | **3,080** | **88** | **73** | **94.9%** |

Index hygiene: the current `artifacts/index/evidence.jsonl` contains 5,410
unique evidence IDs and 0 duplicate evidence-id groups. Repeated same-chapter
question numbers and case subquestions receive deterministic `occurrenceN`
suffixes.

## Evaluation Results

### Smoke Tests

Smoke tests answer "does the pipeline run?" They should not be presented as
semantic retrieval quality.

- `pytest` validates code paths and unit-level behavior.
- A hash-embedding smoke index/query can run without downloading local BGE
  models.
- A smoke query only checks that retrieval and extractive answer formatting
  complete successfully.

### Source-Grounded Evaluation

Dataset: `data/test_questions.json`, 30 reviewed questions generated from
known evidence sources and then reviewed. This is useful for regression checks
and source-grounding sanity, but it is easier than open-ended user questions.

Latest saved result:
`experiments/results/final_current_eval/evaluation_result.json`

| Total | Hit | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 30 | 29 | 0.9667 | 0.1933 | 0.9667 | 0.9667 |

Retrieval ablation:
`experiments/results/retrieval_ablation.json`

| Run | Evidence | Hit / Total | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only | 5,410 | 30 / 30 | 1.0000 | 0.2000 | 0.6417 | 0.7323 |
| BGE dense only | 5,410 | 25 / 30 | 0.8333 | 0.1667 | 0.6522 | 0.6976 |
| BGE + BM25 hybrid | 5,410 | 29 / 30 | 0.9667 | 0.1933 | 0.8011 | 0.8427 |
| Hybrid + BGE reranker | 5,410 | 30 / 30 | 1.0000 | 0.2000 | 1.0000 | 1.0000 |
| Hybrid without image captions | 3,627 | 30 / 30 | 1.0000 | 0.2000 | 1.0000 | 1.0000 |

The last row means the current 30-question source-grounded set does not isolate
image-caption value; it does not mean image captions are unnecessary.

### Web-Style Evaluation

Saved result: `experiments/results/web_rad_eval_result.json`

This set is closer to web/radiology-style questions and is intentionally
reported separately from the source-grounded set. The saved failure analysis
notes that this result and later rebuilt indexes are not perfectly
version-aligned.

| Total mapped questions | Hit | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 22 | 18 | 0.8182 | 0.1818 | 0.5659 | 0.5532 |

Related type-weight experiments on the same 22-question web-style set:

| Setting | Hit / Total | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| All evidence type weights = 1.0 | 13 / 22 | 0.5909 | 0.1182 | 0.3265 | 0.3579 |
| `exercise_qa=0.92`, others 1.0 | 14 / 22 | 0.6364 | 0.1273 | 0.3742 | 0.3884 |

## Setup

Recommended interpreter:

```powershell
conda activate all-in-rag
$PY = 'python'
```

Install dependencies:

```powershell
& $PY -m pip install -r requirements.txt
```

Create local config:

```powershell
Copy-Item config.example.yaml config.yaml
```

`config.yaml` is ignored by git and may contain local model paths or API
settings. Do not commit real API keys. The template defaults to `hash`
embedding and `none` reranker for lightweight smoke tests. For the current main
index, use local BGE models:

- `models/bge-small-zh-v1.5`
- `models/bge-reranker-base`

The default vector-store backend is embedded Qdrant, persisted in
`artifacts/qdrant/`; rebuild the index after switching from the legacy FAISS
files. To use a Qdrant server instead, set `vector_store.qdrant.url` in
`config.yaml` and provide `QDRANT_API_KEY` when the server requires one.
Set `active.vector_store: faiss` to keep using the legacy file-backed baseline.

## Common Commands

Run tests:

```powershell
& $PY -m pytest
```

Verify the current index statistics, duplicate-ID status, test total, and the
published README/HTML/PDF status:

```powershell
& $PY scripts/project_status_snapshot.py --check-docs --run-tests
```

Index commands write to `paths.index_dir` from `config.yaml`. Point that value
to a temporary directory if you want to preserve the current main index.

Build a lightweight smoke index without image captioning:

```powershell
& $PY -m src.cli index --config config.yaml --embedding hash --image-mode none --captioner stub --ocr auto
```

Build a BGE text + exercise index without image captioning:

```powershell
& $PY -m src.cli index --config config.yaml --embedding bge --image-mode none --captioner stub --ocr auto
```

Build the multimodal index with cached/generated captions:

```powershell
& $PY -m src.cli index --config config.yaml --embedding bge --image-mode all-pages --captioner dashscope --ocr auto
```

Query the current BGE main index:

```powershell
& $PY -m src.cli query '脑出血的CT表现是什么？' --config config.yaml --embedding bge --reranker none --generator extractive --top-k 3
```

Generate draft source-grounded evaluation questions:

```powershell
& $PY -m src.cli generate-eval --config config.yaml --output data/test_questions.json
```

Run source-grounded evaluation:

```powershell
& $PY -m src.cli evaluate --config config.yaml --embedding bge --reranker bge --output experiments/results/final_current_eval
```

Run retrieval ablation:

```powershell
& $PY scripts/run_retrieval_ablation.py --config config.yaml --dataset data/test_questions.json --output-dir experiments/results --top-k 5 --candidate-top-k 20
```

Skip the temporary "without image_caption" ablation index:

```powershell
& $PY scripts/run_retrieval_ablation.py --config config.yaml --dataset data/test_questions.json --output-dir experiments/results --top-k 5 --candidate-top-k 20 --skip-no-image-caption
```

Start the Gradio UI:

```powershell
& $PY -m src.cli serve --config config.yaml --embedding bge --reranker bge --generator extractive
```

## Architecture

1. PDF ingestion
   - `PDFLoader` extracts embedded PDF text.
   - `RapidOCRPDFLoader` OCRs low-text pages and caches page OCR under
     `artifacts/ocr/`.

2. Evidence construction
   - Textbooks use layout/heading-aware chunking.
   - Exercise books use `ExerciseParser` to produce one-question-one-chunk
     `exercise_qa` records with question text, options, answer, explanation,
     page metadata, and parser audit output.
   - Page and embedded images can be extracted and captioned, then enriched
     with page number, image kind, nearby text, figure-caption candidates, and
     heading context.

3. Retrieval
   - Dense retrieval: Qdrant (embedded locally by default, or a remote Qdrant service).
   - Sparse retrieval: BM25 with jieba tokenization.
   - Fusion: reciprocal-rank fusion.
   - Optional reranking: BGE CrossEncoder.
   - Evidence type weighting currently keeps `text` and `image_caption` at
     `1.0` and applies `exercise_qa=0.92`.

4. Answer generation
   - `extractive` generator returns cited snippets from retrieved evidence.
   - `dashscope` generator uses an OpenAI-compatible chat API and is instructed
     to answer only from retrieved evidence.

## Known Limitations / Next Steps

- Not a clinical system. The project is for medical-imaging learning and
  retrieval-assisted review only.
- Published status numbers can drift after an index rebuild or test-suite
  change. Run `scripts/project_status_snapshot.py --check-docs --run-tests`
  before publishing updated project or resume materials.
- The 30-question source-grounded evaluation is small and source-shaped. It is
  good for regression testing, but it should not be advertised as real-world
  medical QA accuracy.
- The saved web-style evaluation is more realistic, but it has only 22 mapped
  questions and known index-version drift. Expand it with reviewed expert
  questions.
- Image captions are present at scale, but the current reviewed evaluation set
  does not measure image-caption-specific retrieval value. Add a dedicated
  image-grounded benchmark.
- Exercise parsing still has unmatched question/answer fragments. Continue
  improving section/header handling and answer pairing.
- Full multimodal rebuilds depend on cached captions or a working
  OpenAI-compatible vision model API. Local VLM throughput and API permissions
  remain operational constraints.
- Reranker behavior is promising on the source-grounded set but mixed on
  web-style trials; keep it as an evaluated option rather than a blanket claim.
