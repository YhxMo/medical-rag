# 新下载医学影像资料 v1

下载日期：2026-09-19。所有原始 PDF 均保存在项目 `data/raw/`，由 Git 忽略规则排除。来源为 IAEA 和 MSF 官方域名；完整下载地址、SHA-256、文件大小和提取统计见同目录 `public_corpus_v1.json`。

| 文件 | 内容方向 | PDF 页数 | 至少 50 字符的文本页 |
|---|---|---:|---:|
| iaea_diagnostic_radiology_physics_2014.pdf | 诊断放射物理，2014 | 710 | 695 |
| iaea_radiology_quality_control_2023.pdf | 放射诊断设备质量控制，2023 | 222 | 216 |
| msf_ultrasound_manual_2018.pdf | 超声操作与影像教学，2018 | 345 | 327 |
| 合计 | 英文资料 | 1277 | 1238 |

PDF 页数包含封面和前言，可能不同于出版页面的正文页数。逐页解析未报错，并检查了封面与版权页；这不等于所有图表已被正确抽取。低文字量页面可能为图片、空白或封面，不能仅凭字符数判断是否需要 OCR。

## 来源

- [IAEA Diagnostic Radiology Physics](https://www.iaea.org/publications/8841/diagnostic-radiology-physics)
- [IAEA Handbook of Basic Quality Control Tests](https://www.iaea.org/publications/14890/handbook-of-basic-quality-control-tests-for-diagnostic-radiology)
- [MSF Ultrasound manual 官方 PDF](https://medicalguidelines.msf.org/sites/default/files/2023-07/MSF%20International%20Ultrasound%20manual%20Guideline%2020181004%20EN.pdf)

## 使用边界与后续处理

三份资料的版权页均保留版权，未识别到开放复用许可。官网下载不代表获准公开再分发全文、翻译或改编。此次保留官方原件、记录来源，不上传到模型服务、不发布全文或衍生数据集。

这是替代语料，不是遗失的四本中文教材。旧的 5410 条证据统计和旧评测标签不能沿用。现有 30 题仍保留原位，未自动改标或迁移到这些资料。

资料均为英文；后续应新建英文评测集，或明确标记中英跨语言测试，不应将英文语料直接套用旧中文检索质量结论。当前仅完成下载与解析验证，没有建立新索引、生成图像描述或声称检索效果。

## 本次附带修复

外置盘会生成 `._*.pdf` 元数据旁车文件。项目 PDF 扫描现已过滤隐藏文件与目录，避免把元数据当成 PDF 导致解析失败。新增该场景的回归测试；CLI 相关 10 项测试全部通过。

## 隐私记录

清单仅记录公开资料的标题、机构、版本、公开 URL 与完整性统计，不记录用户身份、密钥、私有配置、绝对路径或原始问答内容。PDF 中原有公开出版信息保持不变。
