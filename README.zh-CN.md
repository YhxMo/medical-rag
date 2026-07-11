# 医学影像学习 RAG

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个面向医学影像学学习、复习和本地教材溯源检索的中文多模态检索增强生成（RAG）项目，数据来源包括本地医学影像教材和习题集。

本项目仅用于教学、学习与资料检索，不是临床诊断系统，不能读取患者检查影像，也不应被用于医疗决策。

## 项目亮点

- 基于 4 本本地 PDF 构建多模态 RAG 流水线，共索引 5,410 条证据：547 条教材文本分块、3,080 条结构化习题问答分块，以及 1,783 条图片描述证据。
- 实现 PDF 加载、RapidOCR 兜底与缓存、标题感知分块、习题问答结构化配对、带上下文的图片描述、统一证据序列化、Qdrant 稠密检索、BM25 稀疏检索、倒数排名融合（RRF）和可选 BGE 重排序。
- 将 2 本习题集解析为 3,080 个问答分块，当前解析报告估算配对率为 94.9%。
- 当前溯源评估集包含 30 道人工审核题目，命中 29/30，Recall@5、MRR 和 NDCG@5 均为 0.9667。
- 在同一套 30 题溯源数据上，混合检索加 BGE 重排序的消融实验命中 30/30，Recall@5、MRR 和 NDCG@5 均为 1.0。该结果应被视为流水线回归信号，而非真实临床基准。
- 保存的网页风格医学影像评估难度更高、与本地原文相似度更低：22 道已映射题目命中 18/22，Recall@5 为 0.8182、MRR 为 0.5659、NDCG@5 为 0.5532。
- 项目环境中的测试状态：67 项通过，1 项跳过。

## 当前索引快照

当前主索引文件：`artifacts/index/evidence.jsonl`。

| 证据类型 | 数量 |
|---|---:|
| `text` | 547 |
| `exercise_qa` | 3,080 |
| `image_caption` | 1,783 |
| **合计** | **5,410** |

来源分布：

| 来源文件 | 证据数量 |
|---|---:|
| `医学影像学.pdf` | 1,032 |
| `医学影像诊断学.pdf` | 1,298 |
| `医学影像学学习指导与习题集.pdf` | 1,852 |
| `医学影像诊断学习题集.pdf` | 1,228 |

图片描述分布：

| 图片类型 | 数量 |
|---|---:|
| 页面图片 | 874 |
| PDF 内嵌图片 | 909 |

当前所有图片描述均通过兼容 OpenAI 接口的视觉模型路径生成，模型元数据为 `qwen3-vl-32b-instruct`。

习题解析报告：

| 习题 PDF | 分块数 | 未匹配题目 | 未匹配答案 | 估算配对率 |
|---|---:|---:|---:|---:|
| `医学影像学学习指导与习题集.pdf` | 1,852 | 85 | 71 | 91.9% |
| `医学影像诊断学习题集.pdf` | 1,228 | 3 | 2 | 99.6% |
| **合计** | **3,080** | **88** | **73** | **94.9%** |

索引完整性：当前 `artifacts/index/evidence.jsonl` 包含 5,410 个唯一证据 ID，不存在重复证据 ID 组。同一章节内重复的题号和病例子问题会获得确定性的 `occurrenceN` 后缀。

## 评估结果

### 冒烟测试

冒烟测试只回答“流水线能否运行”，不应将其表述为语义检索质量。

- `pytest` 用于验证代码路径和单元级行为。
- 使用哈希嵌入的轻量索引与查询无需下载本地 BGE 模型即可运行。
- 冒烟查询仅检查检索流程和抽取式答案格式能否正常完成。

### 溯源评估

数据集：`data/test_questions.json`。其中包含 30 道从已知证据来源生成并经过审核的问题，适合回归检查和溯源健全性验证，但比开放式用户问题简单。

最新保存结果：`experiments/results/final_current_eval/evaluation_result.json`

| 总题数 | 命中 | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 30 | 29 | 0.9667 | 0.1933 | 0.9667 | 0.9667 |

检索消融实验：`experiments/results/retrieval_ablation.json`

| 实验配置 | 证据数 | 命中/总数 | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| 仅 BM25 | 5,410 | 30 / 30 | 1.0000 | 0.2000 | 0.6417 | 0.7323 |
| 仅 BGE 稠密检索 | 5,410 | 25 / 30 | 0.8333 | 0.1667 | 0.6522 | 0.6976 |
| BGE + BM25 混合检索 | 5,410 | 29 / 30 | 0.9667 | 0.1933 | 0.8011 | 0.8427 |
| 混合检索 + BGE 重排序 | 5,410 | 30 / 30 | 1.0000 | 0.2000 | 1.0000 | 1.0000 |
| 混合检索，不使用图片描述 | 3,627 | 30 / 30 | 1.0000 | 0.2000 | 1.0000 | 1.0000 |

最后一行表示当前 30 题溯源数据集无法独立衡量图片描述的价值，并不意味着图片描述没有必要。

### 网页风格评估

保存结果：`experiments/results/web_rad_eval_result.json`

这套题目更接近网页和医学影像学问答风格，因此与溯源评估分开报告。已有失败分析指出，该结果与后来重建的索引并非完全版本对齐。

| 已映射题目 | 命中 | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 22 | 18 | 0.8182 | 0.1818 | 0.5659 | 0.5532 |

同一套 22 题网页风格数据上的证据类型权重实验：

| 配置 | 命中/总数 | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| 所有证据类型权重均为 1.0 | 13 / 22 | 0.5909 | 0.1182 | 0.3265 | 0.3579 |
| `exercise_qa=0.92`，其他为 1.0 | 14 / 22 | 0.6364 | 0.1273 | 0.3742 | 0.3884 |

## 环境安装

推荐使用 Conda 环境：

```powershell
conda activate all-in-rag
$PY = 'python'
```

安装依赖：

```powershell
& $PY -m pip install -r requirements.txt
```

创建本地配置：

```powershell
Copy-Item config.example.yaml config.yaml
```

`config.yaml` 已被 Git 忽略，可以保存本地模型路径或 API 配置。请勿提交真实 API Key。配置模板默认使用 `hash` 嵌入和 `none` 重排序器，便于执行轻量冒烟测试。若要使用当前主索引对应的模型，请准备：

- `models/bge-small-zh-v1.5`
- `models/bge-reranker-base`

默认向量数据库是以嵌入模式运行的 Qdrant，数据保存在 `artifacts/qdrant/`。从旧版 FAISS 文件切换后，需要重新构建索引。如需连接 Qdrant 服务，请在 `config.yaml` 中设置 `vector_store.qdrant.url`；如果服务需要鉴权，还需提供 `QDRANT_API_KEY`。若要继续使用旧版文件型基线，可设置 `active.vector_store: faiss`。

## 常用命令

运行测试：

```powershell
& $PY -m pytest
```

检查当前索引统计、重复 ID、测试数量以及已发布 README/HTML/PDF 的状态：

```powershell
& $PY scripts/project_status_snapshot.py --check-docs --run-tests
```

索引命令会写入 `config.yaml` 中的 `paths.index_dir`。如果希望保留当前主索引，请将该配置指向临时目录。

构建不含图片描述的轻量冒烟索引：

```powershell
& $PY -m src.cli index --config config.yaml --embedding hash --image-mode none --captioner stub --ocr auto
```

构建不含图片描述的 BGE 文本与习题索引：

```powershell
& $PY -m src.cli index --config config.yaml --embedding bge --image-mode none --captioner stub --ocr auto
```

使用缓存或新生成的图片描述构建多模态索引：

```powershell
& $PY -m src.cli index --config config.yaml --embedding bge --image-mode all-pages --captioner dashscope --ocr auto
```

查询当前 BGE 主索引：

```powershell
& $PY -m src.cli query '脑出血的CT表现是什么？' --config config.yaml --embedding bge --reranker none --generator extractive --top-k 3
```

生成溯源评估问题草稿：

```powershell
& $PY -m src.cli generate-eval --config config.yaml --output data/test_questions.json
```

运行溯源评估：

```powershell
& $PY -m src.cli evaluate --config config.yaml --embedding bge --reranker bge --output experiments/results/final_current_eval
```

运行检索消融实验：

```powershell
& $PY scripts/run_retrieval_ablation.py --config config.yaml --dataset data/test_questions.json --output-dir experiments/results --top-k 5 --candidate-top-k 20
```

跳过临时的“不含图片描述”消融索引：

```powershell
& $PY scripts/run_retrieval_ablation.py --config config.yaml --dataset data/test_questions.json --output-dir experiments/results --top-k 5 --candidate-top-k 20 --skip-no-image-caption
```

启动 Gradio 界面：

```powershell
& $PY -m src.cli serve --config config.yaml --embedding bge --reranker bge --generator extractive
```

## 系统架构

1. PDF 摄取
   - `PDFLoader` 提取 PDF 内嵌文本。
   - `RapidOCRPDFLoader` 对文本过少的页面执行 OCR，并将页面 OCR 缓存到 `artifacts/ocr/`。

2. 证据构建
   - 教材使用布局与标题感知分块。
   - 习题集由 `ExerciseParser` 生成一题一块的 `exercise_qa` 记录，包含题干、选项、答案、解析、页码元数据和解析审计结果。
   - 系统可以提取页面图片与 PDF 内嵌图片，生成图片描述，并补充页码、图片类型、邻近文本、图注候选和标题上下文。

3. 检索
   - 稠密检索：Qdrant，默认使用本地嵌入模式，也可连接远程 Qdrant 服务。
   - 稀疏检索：BM25 + jieba 分词。
   - 融合：倒数排名融合（RRF）。
   - 可选重排序：BGE CrossEncoder。
   - 当前证据类型权重中，`text` 和 `image_caption` 为 `1.0`，`exercise_qa` 为 `0.92`。

4. 答案生成
   - `extractive` 生成器返回带引用的检索证据片段。
   - `dashscope` 生成器调用兼容 OpenAI 的聊天接口，并被要求只根据检索证据作答。

## 已知局限与后续计划

- 本项目不是临床系统，仅用于医学影像学学习和检索辅助复习。
- 重建索引或修改测试套件后，README 中发布的状态数字可能发生变化。在发布更新后的项目材料前，请运行 `scripts/project_status_snapshot.py --check-docs --run-tests`。
- 30 题溯源评估集规模较小，而且问题与来源文本较接近。它适合回归测试，但不应被宣传为真实场景医学问答准确率。
- 网页风格评估更加贴近真实提问，但目前只有 22 道已映射题目，并存在索引版本偏移。后续应加入经过专家审核的问题。
- 系统已经大规模加入图片描述，但当前审核评估集无法独立测量图片描述的检索价值，需要新增专门的图片溯源基准。
- 习题解析仍存在未匹配的题目和答案片段，需要继续改进章节、页眉处理和问答配对逻辑。
- 完整多模态索引重建依赖已有图片描述缓存或可用的 OpenAI 兼容视觉模型 API。本地视觉语言模型吞吐量和 API 权限仍是运行约束。
- 重排序器在溯源评估中表现良好，但在网页风格实验中结果不完全稳定，应将其作为经过评估的可选能力，而不是笼统宣传其效果。
