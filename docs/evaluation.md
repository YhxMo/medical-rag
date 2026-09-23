# 检索评测

评测是独立命令，不参与用户问答。它验证“相关证据是否被召回并排在前面”，不评价医学正确性或生成答案质量。

## 准备数据

建库后，在 `artifacts/rag/index/evidence.jsonl` 中核对证据内容，为问题填写相关 `evidence_id`。格式如下；示例 ID 必须替换成当前索引里的真实 ID：

```json
[
  {
    "question_id": "q1",
    "question": "What determines axial resolution in ultrasound?",
    "expected_evidence_ids": ["replace-with-a-reviewed-evidence-id"],
    "reviewed": true
  }
]
```

只有 `reviewed: true` 的记录进入评测。该字段需要由标注者确认，不由模型或脚本自动设为通过。问题 ID 必须唯一，相关证据列表不能为空。重建索引后需重新核对标签。

```bash
uv run medical-rag evaluate questions.json --output experiments/runs/baseline.json
```

评测读取同一份配置，使用相同的 Qdrant + BM25 + RRF + 重排参数。在检索排序后计算 Top-k 指标，不包含问题翻译、截图解析、上下文字数截断和回答生成；默认英文索引应使用英文评测问题。

## 指标

| 指标 | 定义 |
|---|---|
| Recall@k | 前 k 个结果中命中的相关证据数 / 全部相关证据数，再对问题求平均 |
| Precision@k | 前 k 个结果中命中的相关证据数 / k；返回不足 k 项仍以 k 为分母 |
| Hit rate@k | 至少召回一条相关证据的问题比例 |
| MRR | 第一个相关结果的排名倒数的平均值 |
| NDCG@k | 按排名对二元相关性折扣，再除以理想排序的得分 |

重复证据在截取 Top-k 前去重，不能重复计分。每次评测保存汇总指标、逐题命中与排序、关键配置以及数据集和证据文件的 SHA-256。模型 API 密钥不写入报告。默认写入 `experiments/runs/<时间戳>.json`；指定 `--output` 时必须使用新文件，已有记录不会被覆盖。空数据集或无标签数据不会产生看似有效的零分报告。

## 验证与实验记录

自动化测试覆盖真实本地 Qdrant 的建库、重建、重新打开和混合检索，以及 PDF → 分块 → 入库、重排候选池、上下文预算、引用编号、截图路由、CLI/UI 共用服务和指标计算。模型响应在单元测试中使用测试替身；这些测试不代表真实模型回答质量。

实验配置、逐题记录和调参结论见 [experiments/](../experiments/README.md)。对照实验固定语料和标注集，比较检索策略、Top-k 与上下文配置，同时记录召回指标、延迟和失败案例。每轮结果与配置快照一起保存，便于追溯调参依据。
