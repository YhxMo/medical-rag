# Spec: Medical RAG — 医学影像学智能问答系统

## Objective

构建一个面向医学生的**医学影像学图文多模态 RAG（检索增强生成）系统**，基于 4 本教材 PDF 进行智能问答。核心目标是：
- 帮助医学生通过自然语言问答学习医学影像学知识
- 支持教材文本、页内医学影像图片、图注/标题等信息的联合检索
- 先交付轻量可运行 MVP，再逐步扩展为可对比不同模型与检索策略的实验平台

**用户画像：** 医学生（放射科方向），通过提问复习知识点、理解疑难概念。

## Tech Stack

| 层 | 选型 | 版本/说明 |
|---|---|---|
| 语言 | Python | 3.11+ |
| 生成模型 | DashScope Qwen | 第一版使用已有 DashScope API key，OpenAI 兼容格式 |
| 本地图文模型 | Qwen3-VL-2B-Instruct 或 Qwen2.5-VL-3B-Instruct | 用于 PDF 图片 caption/OCR 辅助，适配 RTX 4060 Laptop，优先 4-bit/8-bit 量化部署 |
| 文本 Embedding | BAAI/bge-small-zh-v1.5 | 轻量中文 embedding，本地运行 |
| 图文检索 | 图像 caption + 文本 embedding 为主，AltCLIP/JinaCLIP 作为后续可选实验 | MVP 避免重型多模态向量索引 |
| Reranker | BAAI/bge-reranker-base 或 bge-reranker-v2-m3 | 本地轻量 reranker；MVP 可先设为可选 |
| 向量存储 | FAISS + BM25 (rank-bm25 / jieba 分词) | 轻量、本地 |
| 文档解析 | PyMuPDF (fitz) | 提取文本、页图、图片位置、页码和版面线索 |
| OCR 触发规则 | page text length < 50 | 单页提取文本少于 50 字时标记为可能需要 OCR |
| Chunk 策略 | layout-aware / heading-aware / parent-child / semantic / fixed baseline | MVP 默认不使用纯固定 chunk；固定 chunk 只作为对照实验 |
| 习题集结构化 | question-answer pairing | 题目区和答案/解析区分别抽取，按章节 + 题号配对，一题一个 chunk |
| UI 框架 | Gradio | 简易聊天界面 |
| 评估框架 | RAGAS | 关注 Recall、Precision |
| 配置管理 | YAML (PyYAML) | 切换实验参数 |
| LLM 客户端 | openai Python SDK | 兼容千问 API |

## Model Recommendation

MVP 选型遵循“能在普通电脑上跑起来优先，效果其次可迭代”的原则。

| 模块 | 推荐 | 理由 | 备注 |
|---|---|---|---|
| 图片理解 / caption | `Qwen/Qwen3-VL-2B-Instruct`，若环境不兼容则 `Qwen/Qwen2.5-VL-3B-Instruct` | 2B/3B 级别比 ColQwen、7B VLM 更轻，适合 RTX 4060 Laptop 本地离线生成图片说明 | 优先用 4-bit；显存不足时切 8-bit/CPU offload 或降低图片分辨率 |
| 文本 embedding | `BAAI/bge-small-zh-v1.5` | 中文友好、体积小、CPU 可用 | 第一版把文本块和图片 caption 统一编码 |
| 关键词召回 | BM25 + jieba | 对医学术语、教材原文匹配稳定 | 与向量召回做 RRF 融合 |
| 重排 | `BAAI/bge-reranker-base` | 比多模态 reranker 轻，能先提升文本相关性 | MVP 中设为可选开关，跑不动时关闭 |
| 最终生成 | DashScope `qwen-plus` 或可用的 Qwen chat 模型 | 你已有 DashScope API key，降低本地显存压力 | 回答必须引用检索到的页码和来源 |

不推荐第一版直接使用 ColQwen3.5 或大型多模态 reranker：它们更适合 GPU 资源充足、MVP 跑通后做质量对比实验。

## Commands

```bash
# 环境安装
pip install -r requirements.txt

# 文档预处理 & 索引构建
python -m src.cli index --config config.yaml

# 启动问答 UI
python -m src.cli serve --config config.yaml

# 运行评估实验
python -m src.cli evaluate --config config.yaml --output experiments/

# 运行单个查询（命令行调试）
python -m src.cli query "什么是脑出血的CT表现？" --config config.yaml

# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/ tests/
```

## Project Structure

```
e:\medical_rag\
├── data/                          # PDF 教材（已有）
│   ├── 医学影像学.pdf
│   ├── 医学影像学学习指导与习题集.pdf
│   ├── 医学影像诊断学.pdf
│   └── 医学影像诊断学习题集.pdf
├── src/                           # 源代码
│   ├── __init__.py
│   ├── cli.py                     # CLI 入口（index, serve, evaluate, query）
│   ├── config/                    # 配置管理
│   │   ├── __init__.py
│   │   └── settings.py            # YAML 加载、Config 数据类
│   ├── document/                  # 文档加载与分块
│   │   ├── __init__.py
│   │   ├── loader.py              # PDF → Markdown/Text（PyMuPDF）
│   │   └── chunker.py             # 多种分块策略（固定大小、语义、父子）
│   ├── embedding/                 # Embedding 提供者
│   │   ├── __init__.py
│   │   ├── base.py                # AbstractEmbeddingProvider
│   │   ├── qwen.py                # Qwen3-VL-Embedding
│   │   └── jina.py                # jina-embeddings-v4
│   ├── retrieval/                 # 检索策略
│   │   ├── __init__.py
│   │   ├── base.py                # AbstractRetrievalStrategy
│   │   ├── naive.py               # 朴素向量检索（单轮 top-k）
│   │   ├── hybrid.py              # 混合检索（BM25 + 向量）
│   │   └── parent_child.py        # 父子文档分块检索
│   ├── reranker/                  # Reranker 提供者
│   │   ├── __init__.py
│   │   ├── base.py                # AbstractReranker
│   │   ├── qwen.py                # Qwen3-VL-Reranker
│   │   └── colqwen.py             # ColQwen3.5
│   ├── generator/                 # LLM 生成
│   │   ├── __init__.py
│   │   └── generator.py           # OpenAI 兼容客户端 → 千问
│   ├── indexer/                   # 索引构建
│   │   ├── __init__.py
│   │   └── indexer.py             # 文档→分块→向量化→存储
│   ├── pipeline/                  # RAG 管线编排
│   │   ├── __init__.py
│   │   └── pipeline.py            # 串联检索→重排→生成
│   ├── evaluation/                # 评估框架
│   │   ├── __init__.py
│   │   ├── dataset.py             # 评测数据集管理（问答对 + ground truth）
│   │   └── evaluator.py           # RAGAS 评估（Recall, Precision, MRR 等）
│   └── ui/                        # Gradio 界面
│       ├── __init__.py
│       └── app.py                 # 聊天 UI
├── experiments/                   # 实验配置 & 结果记录
│   ├── README.md
│   └── results/                   # 各次实验的评估指标 CSV/JSON
├── tests/                         # 测试
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── config.yaml                    # 主配置文件
├── requirements.txt
└── README.md
```

## Code Style

遵循 PEP 8，类型注解、dataclass 配置。示例风格：

```python
"""文档加载模块 —— 从 PDF 提取文本内容."""
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Document:
    """解析后的文档单元."""
    content: str
    metadata: dict
    page_num: int | None = None


class PDFLoader:
    """PDF 加载器 —— 使用 PyMuPDF 提取文本."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        """加载 PDF 并返回 Document 列表."""
        docs: list[Document] = []
        with fitz.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        content=text.strip(),
                        metadata={"source": self.file_path.name, "page": page_num},
                        page_num=page_num,
                    ))
        return docs
```

**关键约定：**
- 所有公开函数/方法必须有类型注解
- 使用 `dataclass` 定义数据结构
- 抽象基类用 `ABC` + `@abstractmethod`
- 日志用 `logging` 模块，不直接用 `print`
- 中文注释优先，英文也可接受

## Testing Strategy

| 层级 | 框架 | 位置 | 覆盖目标 |
|---|---|---|---|
| 单元测试 | pytest | `tests/` | chunker、retrieval 逻辑、配置加载 |
| 集成测试 | pytest | `tests/` | 完整管线（文档→检索→生成，mock API） |
| 评估测试 | RAGAS | `experiments/` | 不同配置组合的 Recall/Precision 对比 |

- 核心逻辑（分块、检索、评估）必须有单元测试
- API 调用层用 mock 进行测试
- 不要求覆盖率阈值，但核心路径必须覆盖
- 每次新增检索策略必须同步新增对应的评估用例

## Data Flow

```
PDF 教材
  │
  ▼
[PDFLoader] ──→ Document 列表
  ├──→ 若 page_text_length < 50 ──→ 标记 needs_ocr
  ├──→ 文本提取 ──→ [Layout/Heading-aware Chunker] ──→ text chunks
  │
  └──→ 图片/页图提取 / OCR 候选页 ──→ [Local VLM Captioner] ──→ image captions / OCR text
                                      │
                                      ▼
                         multimodal evidence items
                                      │
                                      ▼
                         [EmbeddingProvider] ──→ 向量
  │
  ▼
[Indexer] ──→ FAISS 索引 + BM25 索引
  │
  ▼
[Query] ──→ [Retrieval] ──→ [Reranker] ──→ [Generator] ──→ Answer
              │                  │               │
              ▼                  ▼               ▼
           候选文档         重排后文档      最终回答 (+ 引用来源)
```

### Exercise Book Data Flow

习题集 PDF 的结构与教材正文不同，不能直接按教材正文 chunk。MVP 对习题集使用独立结构化流程：

```text
习题集 PDF
  |
  +--> 抽取题目区
  |
  +--> 抽取答案/解析区
  |
  +--> 识别章节 + 题号
  |
  +--> 建立 question_id
  |
  +--> 题目答案配对
  |
  +--> 一题一个 chunk
  |
  +--> 进入统一 evidence index
```

习题 chunk 必须包含：
- `question_id`
- `source_file`
- `chapter`
- `question_number`
- `question_type`
- `question_text`
- `options`
- `answer`
- `explanation`
- `page_question`
- `page_answer`
- `related_image_ids`
- `chunk_strategy: exercise_qa_pair`

如果题目或答案/解析无法配对，必须保留未配对记录并写入抽取报告，不能静默丢弃。

## Configuration Design (config.yaml)

```yaml
# 当前激活的实验配置
active:
  embedding: "bge_small_zh"   # bge_small_zh
  reranker: "bge_base"        # bge_base | none
  retrieval: "hybrid"         # naive | hybrid | parent_child
  chunking: "layout_heading"  # layout_heading | parent_child | semantic | fixed | exercise_qa_pair
  multimodal: true
  chunk_size: 512
  chunk_overlap: 50
  top_k_retrieval: 10
  top_k_rerank: 5

# LLM 配置
llm:
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问 OpenAI 兼容端点
  api_key: "${DASHSCOPE_API_KEY}"   # 环境变量引用
  model: "qwen-plus"
  temperature: 0.1
  max_tokens: 2048

# 本地图文 caption 模型配置
vision_captioner:
  provider: "local_qwen_vl"
  model: "Qwen/Qwen3-VL-2B-Instruct"
  quantization: "4bit"        # 4bit | 8bit | none
  max_images_per_page: 6
  ocr_text_min_chars: 50
  caption_prompt: "请用中文描述这张医学影像或教材图片，保留疾病、部位、序列、征象和图注信息。"
  ocr_prompt: "请对这一页医学教材内容进行中文OCR，尽量保留标题、题号、选项、图注、表格和页内结构。"

# Embedding 模型配置
embedding:
  bge_small_zh:
    provider: "sentence_transformers"
    model: "BAAI/bge-small-zh-v1.5"
    device: "auto"

# Reranker 模型配置
reranker:
  bge_base:
    provider: "sentence_transformers_cross_encoder"
    model: "BAAI/bge-reranker-base"
    device: "auto"

# 检索参数
retrieval:
  naive:
    top_k: 10
  hybrid:
    dense_top_k: 10
    bm25_top_k: 10
    fusion_method: "rrf"          # reciprocal rank fusion
    final_top_k: 10
  parent_child:
    child_top_k: 20
    parent_top_k: 5
    child_chunk_size: 256
    parent_chunk_size: 1024

# Chunk 策略配置
chunking:
  layout_heading:
    target_tokens: 500
    max_tokens: 900
    min_tokens: 120
    preserve_headings: true
    attach_nearby_captions: true
    split_on: ["chapter", "section", "subsection", "paragraph"]
  parent_child:
    parent_target_tokens: 1200
    child_target_tokens: 350
    child_overlap_tokens: 60
  semantic:
    target_tokens: 500
    similarity_threshold: 0.72
    fallback_to_layout: true
  fixed:
    chunk_size: 512
    chunk_overlap: 50
  exercise_qa_pair:
    enabled: true
    question_section_markers: ["习题", "选择题", "名词解释", "简答题", "病例分析"]
    answer_section_markers: ["参考答案", "答案", "解析", "参考答案与解析"]
    question_id_template: "{source_stem}:{chapter}:{question_number}"
    one_question_per_chunk: true

# 评估数据集配置
evaluation:
  test_dataset: "data/test_questions.json"  # 标注问答对
  metrics: ["recall", "precision", "mrr", "ndcg"]
  dataset_generation: "semi_auto"
```

## Experiment Matrix

每次实验记录以下变量的组合及其评估结果：

| 变量 | 候选值 |
|---|---|
| Embedding | bge-small-zh-v1.5, later: bge-m3 / jina-clip |
| Reranker | bge-reranker-base, none (baseline), later: bge-reranker-v2-m3 |
| 检索策略 | naive, hybrid, parent_child, multimodal_caption |
| Chunk 策略 | layout_heading, parent_child, semantic, exercise_qa_pair, fixed baseline |
| Chunk 大小 | 256, 512, 1024，仅用于 parent/child/fixed 等需要长度参数的策略 |

MVP 只跑关键组合，先确认端到端质量和速度；完整矩阵留到系统稳定后扩展。

## Chunking Strategy

MVP 不以固定长度切块作为默认方案。医学影像教材有清晰章节层级、诊断要点、图注、表格和页内图片，chunk 必须尽量保留这些语义边界。

| 策略 | 用途 | MVP 优先级 |
|---|---|---|
| `layout_heading` | 根据章节标题、段落、页码、图注和版面线索组织 chunk | 默认策略 |
| `parent_child` | 父 chunk 保留完整小节语境，子 chunk 用于精确召回 | 第一版支持 |
| `semantic` | 用句向量相似度在语义转折处切分 | 作为实验策略，MVP 可后置 |
| `exercise_qa_pair` | 习题集专用：题目区和答案/解析区分开解析后按章节 + 题号配对 | 习题集默认策略 |
| `fixed` | 固定 token/字符窗口 | 只作为 baseline，不作为默认 |

chunk 输出必须包含：
- `chunk_id`
- `source_file`
- `page_start`
- `page_end`
- `heading_path`
- `content`
- `nearby_image_ids`
- `nearby_caption_ids`
- `chunk_strategy`
- `needs_ocr`（页文本少于 50 字时为 true）

图片 caption 也视为 evidence，但需要绑定到页码、附近标题、原图路径和可能相关的文本 chunk。

PDF loader 必须记录每页原始抽取文本长度。若单页文本长度 `< 50` 个中文/英文字符，则该页标记为 `needs_ocr=true`，后续进入 OCR 或本地 VLM 页图理解流程。该规则用于召回扫描页、图片页、版式异常页和 PyMuPDF 抽取失败页。

习题集 evidence 必须一题一个 chunk，不与相邻题目混合；题目、选项、答案、解析和页码引用要作为结构化字段保留。

## Gradio UI Design

```
┌──────────────────────────────────────────────┐
│  🏥 医学影像学 RAG 问答系统                    │
│  ───────────────────────────────────────────  │
│  [配置面板]  (可折叠)                          │
│  ├─ Embedding: [qwen ▼]  Reranker: [qwen ▼]  │
│  ├─ 检索策略: [hybrid ▼]  检索数量: [5  - +]  │
│  └─ LLM: [qwen-plus ▼]                       │
├──────────────────────────────────────────────┤
│  ┌────────────────────────────────────────┐   │
│  │  👤 用户：什么是脑膜瘤的MRI表现？        │   │
│  │                                        │   │
│  │  🤖 助手：脑膜瘤在MRI上通常表现为：     │   │
│  │  1. T1WI呈等或稍低信号...               │   │
│  │                                        │   │
│  │  📚 参考来源：                          │   │
│  │  · 医学影像诊断学.pdf, 第152页          │   │
│  │  · 医学影像学.pdf, 第89页               │   │
│  └────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────┐   │
│  │  [输入你的问题...]          [发送]      │   │
│  └────────────────────────────────────────┘   │
├──────────────────────────────────────────────┤
│  [📊 实验模式] [🔄 清除对话] [⚙️ 设置]       │
└──────────────────────────────────────────────┘
```

**功能要点：**
- 左侧或顶部可折叠的配置面板，可切换实验中各参数（embedding、reranker、检索策略）
- 正文是对话区域，显示用户问题和带引用来源的回答
- 底部输入框 + 发送按钮
- "实验模式"：对同一问题用不同配置跑多次，并排展示结果和指标对比
- 支持 Markdown 格式的回答渲染

## Boundaries

- **Always do:**
  - 修改配置参数后重新索引再测试
  - 评估结果记录到 `experiments/results/` 中
  - API key 通过环境变量注入，不硬编码
  - 回答时必须附带来源引用（页码、文件名）

- **Ask first:**
  - 新增依赖库（pip install 新包）
  - 修改 config.yaml 的结构/字段
  - 新增检索策略
  - 修改评估指标定义

- **Never do:**
  - 将 API key 提交到版本控制
  - 修改 `data/` 下的原始 PDF
  - 删除实验记录
  - 使用 `print` 替代 `logging`（调试时可以临时用）

## Success Criteria

1. **问答功能可用：** 用户输入中文问题 → 返回带来源引用的中文回答
2. **可切换配置：** 通过 UI 或 config.yaml 自由切换 embedding / reranker / 检索策略
3. **评估可运行：** `python -m src.cli evaluate` 能跑通并输出 Recall、Precision 指标
4. **实验可复现：** 每次评估结果自动记录配置 + 指标，保存到 `experiments/results/`
5. **Gradio 可启动：** `python -m src.cli serve` 启动后浏览器可访问，完成一轮问答
6. **文档处理：** 4 本 PDF 全部可正确解析、按可配置策略分块、提取页图/图片并索引
7. **图文证据可用：** 至少能从教材图片 caption 中检索到相关页，并在回答中引用图片所在页
8. **Chunk 策略可比较：** 至少支持 `layout_heading` 和 `fixed` 两种策略，并能在评估结果中记录当前 chunk 策略
9. **习题集结构化可用：** 习题集 PDF 能按章节 + 题号建立 `question_id`，完成题目答案配对，并做到一题一个 chunk
10. **OCR 候选页识别：** 任一 PDF 页面提取文本少于 50 字时必须标记为 `needs_ocr`，并进入 OCR/视觉补救流程

## Open Questions

1. **显存大小：** 已确认有 RTX 4060 Laptop；通常按 8GB 显存保守设计。若实际显存不同，需要调整 VLM 量化和 batch size。
2. **DashScope 模型名：** 你已有 DashScope API key；实施时需要确认实际可调用的 chat/VL 模型名。
3. **半自动评测集规模：** MVP 建议先生成并人工抽查 30 条图文混合问题，再扩展到 100+。

---

👉 **请 review 以上 spec，确认或纠正后我们进入 Phase 2（Plan）。**
