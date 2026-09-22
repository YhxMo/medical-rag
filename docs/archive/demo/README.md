# 可复现演示说明

本项目定位为医学影像教材学习助手，展示来源检索、引用回答与证据评测工程。没有临床验证或人工专家认证。当前可验证语料为三份英文教材，共2,809条文本证据，未启用OCR/VLM，因此不能承诺读懂图像和曲线。

## 环境与输入

在项目根目录创建 Python 3.12 虚拟环境，安装项目依赖。`requirements.txt` 是功能依赖范围；`docs/evaluation/requirements-offline-test.txt` 是之前已验证的测试环境快照，不能冒充整个生产环境锁文件。本轮精确运行环境另存 `docs/evaluation/requirements-experiment-v1.txt`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/evaluation/requirements-experiment-v1.txt
python -m pytest -q -p no:cacheprovider --tb=short
```

本地已有语料、模型与Qdrant索引时可直接演示。全新机器需要按 `docs/evaluation/public_corpus_v1.json` 中的来源与SHA-256获取原文，再按 `docs/evaluation/public_v1_rebuild.md` 重建。生成数据和PDF未提交Git，不能声称克隆仓库就拥有全部数据。公开可访问不等于允许重新分发。不要同时启动多个进程占用同一个本地Qdrant目录。

## 先做零API演示

```bash
python -m src.cli query 'What is the purpose of maintaining radiographic screens, and what equipment is used for cleaning them?' --config config.public-corpus.yaml --top-k 10 --generator extractive
```

此命令展示检索片段及出处，不是模型生成的医学回答。本轮已执行并将输出放入 `data/evaluation/context-experiment-v1/offline_demo.txt`。第10条能定位到正确手册的PDF第183页；extractive模式只显示前180字符，完整清洁条目需看原页或 `data/evaluation/paired-answers-v1/public_v1_s040_ranked_10.json` 中保存的实际回答。

对照前5条：将 `--top-k 10` 改为 `--top-k 5`。可选实验参数 `--expand-context` 会给原始命中追加同书相邻文字块，最多10条/12,000字符，保留来源与独立引用编号；本轮收益小，未默认启用。

真实生成可将 `--generator extractive` 改为 `--generator deepseek`，通过运行时环境提供 `DEEPSEEK_API_KEY`，不要写进命令历史、配置、截图或报告。普通 query 不是预算受控的批量实验入口；批量评测应使用下面带费用上限的脚本。

## 复现实验

```bash
python scripts/experiment_context.py
python scripts/run_paired_answers.py
python scripts/run_evidence_challenge.py
```

第一项为本地检索实验，后两项会调用 DeepSeek 官方。密钥可在交互终端无回显输入。后续运行的配对脚本上限14元，挑战脚本上限1元，两者合计最多15元，分别记账。本轮已完成运行时配对任务本身设置15元，实际保守预留总计约9.85元、已知用量峰价估算约1.15元，未超过已告知的合并15元上限；历史报告保留当时的设置。

已保存答案会复用，并校验输入题集、上下文、索引和提示词版本，拒绝不同输入复用同一缓存目录。检索实验包含运行耗时，重新执行会改变上下文文件哈希，即使排序相同也可能触发保守的缓存拒绝；应采用新实验目录记录新运行，不删除旧结果来掩盖版本差异。在线API没有不可变模型权重标识，也不保证字面答案完全可重现。

## 讲清楚三个案例

1. **清洁维护题漏召回**：前5条含同主题但不完整的片段；关键证据位于第10条。扩至10条后实际回答包含厂家认可的清洁液、无绒布和刷子。代价是平均上下文约翻倍。
2. **IVC题缺条件**：原标注块截掉适用人群和后文限制。先修订题目与多条来源标签，再做固定版本对照，观察回答是否强调不能孤立决定补液。新增上下文也可能诱发不必要的数值阈值扩写，不能把模型高分等同于临床安全。
3. **尚未解决的章节混淆**：参考文献题检索到了别的章节书目，增加到10条仍未找到期望来源。保留为失败，后续考虑章节元数据和查询解析，并用开发集验证。

## 常见追问

- **为什么混合检索？** 语义向量与关键词匹配互补；本项目用Qdrant保存向量，BM25作稀疏检索，RRF合并名次。它不保证每题最优，需要消融数据支持。
- **为什么没有直接上线相邻块扩展？** 代码已经实现并测试，但本次数据上增加Top-K更有效；保留实验能力、默认不启用收益不明确的策略。
- **为什么不能说准确率98.2%？** 56/57是同模型评委对来源限定回归题的通过数；题集由AI生成并修改过，不是独立临床样本，也没有医学专家标注。
- **怎样处理隐私？** API只接收本轮授权的公开教材及派生题答，密钥只运行时读取；逐题材料在Git忽略目录，公开报告只保留技术汇总。
- **还缺什么？** 更大规模真实用户问题、独立评审、真正盲测的端到端难例，以及医学专业验证。目前的12题挑战采用受控证据，不测检索噪声下的拒答能力。
