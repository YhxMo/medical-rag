# Medical Imaging Textbook Learning Assistant

[简体中文](README.zh-CN.md) · [Setup and experiments](docs/resume-v2/README.md) · [Verification status](docs/resume-v2/status.json)

A source-grounded RAG application for textbook learning: hybrid retrieval, document figures, screenshot questions, citations, and reproducible evaluation. Not a patient image diagnosis system.

The public corpus contains **2,809 text evidence records from three textbooks**. Resume-v2 has built two isolated indexes containing these records plus **60 real model-generated visual descriptions**, with registered originals, source pages and chapter metadata.

The shared CLI/Gradio query service supports English retrieval expressions for Chinese questions, screenshot interpretation, optional original-image answers, evidence sufficiency checks and citation validation. Retrieval uses local BGE/ONNX, Qdrant, BM25 and RRF, with chapter-aware and reranking ablations. A shared budget ledger caps additional model spending at CNY 30.

**Implemented interfaces are not evidence of successful live multimodal evaluation.** See the verification status for completed local checks and outstanding API-dependent experiments. Historical results are not reused as current accuracy claims.

```bash
python -m pytest -q -p no:cacheprovider --tb=short
python scripts/build_resume_v2.py --mode text
python -m src.cli query 'What determines axial resolution in ultrasound?' --config config.resume-v2.yaml --json
python -m src.cli serve --config config.resume-v2.yaml
```

Use Python 3.12 and the dependency snapshot in `docs/resume-v2/requirements.lock.txt`. A fresh checkout requires the source PDFs and public-v1 base evidence first; see the setup guide. Textbooks, image assets, model weights, credentials and full generated datasets are not distributed in Git.

The **80-task AI-generated dataset is frozen**, grouped by source into 40 development and 40 held-out tasks. In the 10-task development text retrieval comparison, reranking increased labelled Recall@5 from 0.80 to 0.90 and NDCG@5 from 0.706 to 0.755, with a 1.85-second retrieval P95; it is the frozen default. This small-sample retrieval result is not answer accuracy. AI judgments, source-review disagreements, human review and clinical validation are reported separately. Previously inspected questions remain regression data.

Historical Chinese-corpus statistics and documentation are preserved in [the historical README](docs/history/README.pre-resume-v2.en.md). Do not quote their 100% retrieval score as current medical answer accuracy.
