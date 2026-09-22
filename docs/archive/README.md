# 归档说明

本目录收纳当前实验链路不再使用的历史材料，仅供追溯，不再维护。顶层 README 不引用本目录。

## 重要声明

中文 4 教材原始语料、中文索引与配套题库的来源 PDF 已丢失，`chinese-mvp/` 下材料**仅作历史记录，不可复现**。其中的证据标签、页码与评分不能作为当前系统（三本公开英文教材语料）的能力证据。

## 归档清单

| 原路径 | 现路径 | 说明 |
|---|---|---|
| `spec.md` | `chinese-mvp/spec.md` | 初始规格（结构章节已过时，列出的多个模块从未实现） |
| `tasks/` | `chinese-mvp/tasks/` | 中文 MVP 计划与已完成任务清单 |
| `医学评估题目/` | `chinese-mvp/医学评估题目/` | 中文语料时代人工评估候选与题集 |
| `评估指标记录/` | `chinese-mvp/评估指标记录/` | 2026-07-06 混合检索基线评估（215 条证据，旧语料） |
| `config.example.yaml` | `chinese-mvp/config.example.yaml` | 中文语料配置模板 |
| `docs/history/` | `README.pre-resume-v2.{en,zh-CN}.md` | resume-v2 之前的双语 README |
| `docs/demo/` | `demo/` | 旧演示说明（可复现案例已吸收进 `docs/resume-v2/`） |
| `docs/evaluation/requirements-offline-test.txt` | `requirements-offline-test.txt` | 历史测试环境快照 |
| `docs/evaluation/requirements-experiment-v1.txt` | `requirements-experiment-v1.txt` | 历史实验环境快照 |
| `experiments/` | `experiments/` | 旧轮次用量记录 |
| `scripts/` 下 17 个历史脚本 | `scripts/` | 诊断（diag_*×4）、中文题库工具、public-corpus 旧轮次实验与复核包脚本 |

本地未入库的旧轮次评测数据（`data/evaluation/` 下除 `resume-v2/` 外的目录）因被冻结链路脚本按固定路径引用（如 `scripts/prepare_resume_v2.py` 的回归排除表），在实验收尾前保持原位，Git 忽略。

## 指纹迁移记录

`scripts/verify_resume_v2.py` 对 `src/`、`scripts/`、`tests/` 全部 `.py` 的相对路径与内容字节做 SHA-256 指纹。本次归档移动了 17 个历史脚本，指纹范围随之变化，**其余 `.py` 文件路径与内容未做任何修改**：

| | 指纹 | 范围内 `.py` 数 |
|---|---|---|
| 迁移前（tag `pause-2026-09-22-holdout-4-60`，`docs/resume-v2/status.json` 记录） | `078f026120fed0b310da293bcf099c72fdb1c467a8c82340b8c6904753690aa8` | 125 |
| 迁移后（本 commit） | `7239074613f43b1b02d966a82d55f33df45c18ff7356cf196441d19c99814509` | 108 |

移出指纹范围的 17 个文件（字节内容未变）：

`scripts/diag_case_analysis.py`、`scripts/diag_case_answers.py`、`scripts/diag_exercise.py`、`scripts/diag_unmatched_q.py`、`scripts/build_manual_eval_candidates.py`、`scripts/validate_eval_candidate_ids.py`、`scripts/verify_evaluation_offline.py`、`scripts/experiment_context.py`、`scripts/build_evidence_revision.py`、`scripts/run_answer_quality.py`、`scripts/run_paired_answers.py`、`scripts/run_evidence_challenge.py`、`scripts/evaluate_public_candidates.py`、`scripts/run_retrieval_ablation.py`、`scripts/build_review_packet.py`、`scripts/serve_review_packet.py`、`scripts/apply_human_review.py` → 全部迁至 `docs/archive/scripts/`。

冻结实验的输入冻结（`run_manifest.json` 的 dataset/index/judge/selection 哈希）不受影响；续跑前请核对 `artifacts/resume-v2/selection.json` 与索引版本一致。
