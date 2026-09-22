# 英文公开资料索引与 DeepSeek 题集重建记录

执行日期：2026-09-19 至 2026-09-20。新语料独立于原来的四本中文教材；本轮不会更新或沿用旧的 5,410 条证据统计。

## 已完成

- 从三份官方 PDF 构建 2,809 条文本证据，逐页固定分块 1,200 字符、重叠 160 字符；保留来源和 PDF 页码。
- 使用本地 BAAI/bge-small-en-v1.5（FastEmbed / ONNX）生成 384 维语义向量，建立 Qdrant 本地索引和 BM25/RRF 混合检索。
- 本轮为文本索引，未执行图片描述或 OCR；不能把本轮成果写成新语料的多模态评测。
- 使用 DeepSeek 官方 deepseek-flash（官方文档对应 V4.1 Flash）生成 60 道英文候选题，每份资料 20 道。
- 分层抽样前固定来源页，开发集 30 道、评测集 30 道，来源页交集为 0。书籍和主题并未独立划分，因此不是跨领域泛化测试。
- 第二次模型调用核验题目和答案的证据支持性：58 道通过、2 道待重点复核。全部 reviewed=false，人审计数为 0。
- 增加 DeepSeek 回答入口、英文嵌入入口和英文 BM25 大小写归一化；测试 95 项通过，5 条第三方弃用警告。

## 文件位置（相对于项目根目录）

| 文件或目录 | 用途 |
|---|---|
| config.public-corpus.yaml | 新语料配置；无密钥 |
| artifacts/public-v1/index/ | 证据、BM25 与构建清单 |
| artifacts/public-v1/qdrant/ | 真实语义向量索引 |
| data/evaluation/public-v1/dev.json | 30 道开发候选题 |
| data/evaluation/public-v1/eval.json | 30 道评测候选题 |
| data/evaluation/public-v1/review_queue.json | 人工复核要求与优先处理题号 |
| experiments/results/public-v1/ | API 用量、题集版本与初步检索指标 |
| docs/evaluation/public_v1_run_summary.json | 可核查的汇总记录与代码摘要 |

## 模型调用与费用

共 24 次官方 API 请求（出题 12 次、核验 12 次），全部成功。输入 43,905 tokens，输出 6,706 tokens，供应商返回用量完整。
按核实的高峰价格保守计算约 0.1415 元，低于本轮 2 元预算；这是依据实际 token 用量计算的费用估算，不是账单金额。
发送内容仅为抽样公共教材文本与出题/核验指令；未上传整本 PDF、用户身份或私有文件。密钥只由隐藏输入读入进程内存并用于官方接口认证，任务结束后未持久化。日志只记录汇总量；生成的题目与参考答案是任务产物，保存在 Git 忽略的数据目录，不复制进汇总日志。

## 初步检索检查（全部候选题，含 2 道自动核验未通过题）

| 分组 | BM25 命中@5 | 向量命中@5 | 混合检索命中@5 |
|---|---:|---:|---:|
| 开发集 | 29/30 | 28/30 | 30/30 |
| 评测集 | 27/30 | 28/30 | 29/30 |

这是基于来源生成题和单个来源块标签的临时检查，存在题目与来源接近、相关证据标注不穷尽、模型自审偏差等限制。没有删除失败题以改善指标，也没有用评测集调参。上述命中数不是回答正确率、临床准确率或独立人工基准成绩。

## 运行与复现

安装环境依赖可使用 docs/evaluation/requirements-offline-test.txt。模型文件已下载到 models/fastembed，向量计算在本地完成。

本地查询（不调用付费回答模型）：

```bash
python -m src.cli query "What determines axial resolution in ultrasound?" --config config.public-corpus.yaml --embedding fastembed --reranker none --generator extractive
```

需要 DeepSeek 回答时，在进程环境设置 DEEPSEEK_API_KEY，再将 --generator 改为 deepseek。不要将密钥写进命令历史、配置或版本库。

重跑临时检索检查：

```bash
python scripts/evaluate_public_candidates.py --allow-unreviewed
```

正常 evaluate 命令仍拒绝把全部待审核题当成正式已审核题；--allow-unreviewed 仅存在于明确标为临时检查的专用脚本。

索引脚本 scripts/rebuild_public_corpus.py 核对 PDF 清单与 SHA-256，已有构建清单时拒绝覆盖；要建立新语料版本应使用新的路径。题集脚本 scripts/generate_public_questions.py 通过隐藏输入或环境变量读取密钥，保留每批结果以避免重复付费，单轮预算上限 2 元。若更换数据、提示词或生成策略，应建立新版本，不复用旧批次缓存。

LLM 输出不保证逐字复现：通过保留冻结题集、采样方案、脚本、模型标识、文件 SHA-256 和用量记录进行审计。embedding 模型文件摘要保存在 public_index_v1.json。

## 后续审核

优先复核 review_queue.json 中两道自动检查未通过题；逐题检查前提、参考答案、证据定位和漏标的相关块。审核/修改后保存新题集版本，才能作为正式人工评测集。跨语言问答与图像描述质量需要另外建立测试集。

## 官方依据

- [DeepSeek V4.1 Flash 发布说明](https://api-docs.deepseek.com/zh-cn/news/news260910/)
- [DeepSeek 官方模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)
- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
- [FastEmbed 本地英文嵌入说明](https://qdrant.github.io/fastembed/Getting%20Started/)
