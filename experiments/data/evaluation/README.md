# 本地评测数据目录

## 纳入版本管理的内容

- `resume-v2/`：80 道冻结题集（`dataset.json`，开发／留出各 40）、来源核验、配对回答、评分修复与来源复核记录、`run_manifest.json`。这是当前实验的冻结记录，失败与未完成条目如实保留。
- 本说明文件。

## 仅保留在本地、不入库的内容

- `data/raw/`：三份公开英文教材 PDF（版权文件，不随仓库分发；来源与 SHA-256 见 `docs/evaluation/` 改进记录）。
- 旧轮次实验数据目录（`public-v1/`、`public-v2-draft/`、`answer-quality-v1/`、`paired-answers-v1/`、`challenge-v1/`、`context-experiment-v1/`、`medical-review-v3/`）：属于历史实验输入／输出，被冻结链路脚本按固定路径引用（如 `scripts/prepare_resume_v2.py` 的回归排除表），在实验收尾前保持原位，Git 忽略。

## 可重建产物

`artifacts/` 下的索引、图片资产、模型缓存与 API 缓存不入库；除已跟踪的小型状态 JSON 外均可重建：

```bash
python scripts/prepare_resume_v2.py   # 来源核验 + 视觉页冻结（输出 artifacts/resume-v2/preparation.json）
python scripts/build_resume_v2.py --mode text
python scripts/build_resume_v2.py --mode multimodal
```

只记录汇总数量、技术配置和指标；不在文档中复制原始问题、答案、教材片段或凭据。
