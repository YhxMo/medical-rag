# Resume-v2 交付与复现

本版本面向医学影像教材学习。代码实现了文字／截图输入、图文证据检索、可选原图回答和引用检查。**在线多模态效果必须以实际实验报告为准，接口实现不等于已经通过多模态评测。** 当前状态见 [验收记录](status.json)。

## 安装与本地启动

没有教材和模型也能运行自写小语料演示：`python scripts/offline_resume_demo.py`；加 `--serve` 可打开 UI。此模式明确标为软件演示，不是医学检索基准。保存的实际回答可用 `python scripts/replay_resume_answers.py /path/to/record.json` 回放，始终显示历史标识。

从项目根目录执行，使用 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/resume-v2/requirements.lock.txt
python -m pytest -q -p no:cacheprovider --tb=short
python scripts/prepare_resume_v2.py
python scripts/build_resume_v2.py --mode text
python -m src.cli serve --config config.resume-v2.yaml
```

依赖锁文件来自本次 macOS ARM64 运行环境；CI 配置用于在 Linux 上额外验证，尚未实际运行的远程 CI 不计入通过记录。全量正文嵌入需要 CPU 时间；该时间不是查询延迟。

原始三份 PDF、模型权重和完整索引不随仓库分发。按 `docs/evaluation/public_corpus_v1.json` 的来源取得原文并核对 SHA-256；已有 `artifacts/public-v1/index/evidence.jsonl` 时可直接准备 v2。全新机器先按旧版公开语料说明构建文本基础：`python scripts/rebuild_public_corpus.py`。该基础与历史中文教材不同。

程序仅监听本机回环地址，不开放公网。同一个本地 Qdrant 目录不要被两个进程同时打开。新版使用 `artifacts/resume-v2/text/` 与 `artifacts/resume-v2/multimodal/` 独立索引，保留 v1。

## 模型服务及 30 元预算

- 文本：DeepSeek 官方兼容接口，`DEEPSEEK_API_KEY`。
- 视觉：百炼华北2北京兼容接口，`DASHSCOPE_API_KEY`；默认 `qwen3-vl-plus`。
- 输入截图会发送至配置的视觉模型服务；不将其加入知识库。
- 密钥不写入 YAML、报告或 Git。新终端的环境变量不会自动传入已运行的桌面进程，建议直接在配置密钥的终端运行脚本。

本机无回显输入入口：

```bash
python scripts/resume_v2_secure_run.py scripts/run_resume_v2.py
```

长实验可使用 `python scripts/resume_v2_secure_run.py --detach scripts/run_resume_v2.py`：先在本机无回显输入，再由独立本机进程执行；进度写入被 Git 忽略的 `artifacts/resume-v2/workflow.log`，密钥只通过进程内存中的环境传递。失败仍只等待 600 秒修复，不会无限重试。虚拟环境应放在持久目录，避免放在重启后可能清理的 `/tmp`；本次续跑环境为 `~/.local/share/medical-rag/resume-v2-venv`。

提示时分别输入密钥，输入内容不回显，只传给本次子进程。调用总上限 30 元，入库 10 元、题集／评测 15 元、重试／演示 5 元，全部使用一个跨进程加锁账本。调用前预留上界；缺少用量或连接结果不明时保留预留。额外启动其他不使用本账本的脚本不受此预算管理。

定价依据：2026-09-21 查阅 [百炼模型说明](https://help.aliyun.com/zh/model-studio/qwen3-vl-plus)，北京输入不超过 128K 时官方最高档为输入 1.5、输出 15 元／百万 token。本版本请求上界小于 100K，预算采用更保守的 2／16 元，不抵扣缓存优惠。DeepSeek 沿用已记录峰价上界 2／8 元；运行前核实 [官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。这里是费用估算，不是账单。

各阶段也可独立续跑：

```bash
python scripts/build_resume_v2.py --mode multimodal
python scripts/generate_resume_v2_dataset.py
python scripts/evaluate_resume_v2.py retrieval
python scripts/evaluate_resume_v2.py answers --split dev
python scripts/evaluate_resume_v2.py answers --split holdout
```

三页视觉 API 试跑通过后才继续 60 页描述。每项描述、评分与模型输出留存缓存；完成的索引不可用不同输入覆盖。题集生成、图片数量或提示词改变时建立新版本。预算耗尽时停止付费工作，不悄悄增加限额或删去失败项。

## 使用方式

```bash
# 只展示本地证据，不调用 API
python -m src.cli query 'What determines axial resolution in ultrasound?' --config config.resume-v2.yaml --json

# 在线自动选择文本／视觉回答
python -m src.cli query '超声的轴向分辨率由什么决定？' --config config.resume-v2.yaml --generator deepseek --json

# 教材截图，可省略文字问题
python -m src.cli query '解释图中的坐标与趋势' --image /absolute/path/textbook.png --config config.resume-v2.yaml --generator deepseek --json
```

UI 与 CLI 使用同一 QueryService。UI 默认只检索证据，取消该选项才调用模型。图片最大 10MB／2,000 万像素，发送前保留纵横比缩至长边最多 1,600 像素；细小公式可能仍不可辨认，应明确提示，不生成猜测数值。上传文件请求副本自动清理；Gradio 缓存按 60 秒周期清理过期文件。

流式输出先显示“引用待校验”，完成后展示编号检查结果。引用编号正确不等于语义正确。模型出错单列为 `service_error`，不会计为正确拒答。上下文不足返回 `insufficient_evidence`；部分可答返回 `needs_information`。

## 实验设计

新候选任务按来源组隔离：文本、图表、截图、证据不足／错误前提各 20 题，共 80 题，开发／留出各 40。页码相邻、重复图像与改写不能跨组。AI 出题及 AI 来源复核单独标记，人工审核数量不自动增加。存在来源审核疑点的样本保留并报告，不能作为可靠准确率证明。

检索开发实验固定 5 条／6,000 字符，比较基线、章节增强、章节增强加重排序；只使用开发集选择。若三种策略没有全部完成，不冻结默认策略。图像实验同一视觉模型比较描述与原图，保存完整检索上下文用于验证只有原图输入变化。文本基线模型不同，单列结果不混算收益。

旧 57 题可另作回归：

```bash
python scripts/evaluate_resume_v2.py retrieval --regression --dataset data/evaluation/medical-review-v3/regression_candidates.json
```

`--regression` 不写默认策略，不把旧题成绩当成新留出集结果。单次预热后的耗时仅是该设备运行观测；不能宣传为高并发生产性能。任何首次留出结果被用于修改系统后，该题集后续只能称回归。

## 验收与演示

五种场景：正常文本、图表解释、截图提问、证据不足、策略对照。每个在线演示均应保存实际回答、证据、配置与模型用量；没有凭据时只演示离线检索与历史回放，不伪造模型结果。

最终复核 `answers-*/review_queue.json` 中全部失败、策略分歧以及固定随机 10 题。该队列完成前在线实验只算模型代理评测。测试、实际 API 运行、助手来源核对与真实人工审核分别报告。
