# 路线图与验收

| 阶段 | 验收标准 | 当前状态 |
| --- | --- | --- |
| 基础框架 | 可安装；配置严格校验；CLI 帮助/版本/配置检查可用；预留命令失败；Ruff 与 pytest 通过 | 本地通过，见 [检查记录](local-checks.md) |
| 源码迁移 | 按迁移映射整理模块；确认归属和依赖；消除演示入口与导入副作用 | loss、optimizer、scheduler、采样、完整 TransformerLM 前向、checkpoint、BPE、Tokenizer、预处理与文件加载已迁移，其余待迁移 |
| 正确性验证 | tokenizer 往返和特殊边界；模型组件/loss/优化器数值及梯度对照；调度、采样与 checkpoint 测试 | loss、optimizer 的 CPU 检查见各迁移记录；[调度](scheduler.md)、[采样](data.md)、[基础层](layers.md)、[注意力](attention.md)、[完整模型](transformer.md)、[checkpoint](checkpoint.md)、[BPE](bpe.md) 与 [Tokenizer](tokenizer.md) 见对应记录，其余待验证 |
| 训练闭环 | 小数据可预处理、训练、验证、保存、恢复、续写；恢复状态与步数语义明确 | [数组训练/验证](training.md)、保存与精确续训已验证；[预处理/文件加载](preprocess.md) 已提供 API，[预处理 CLI](preprocess-cli.md) 已接通；训练/验证/生成 CLI 和续写待接入 |
| 基线实验 | 固定配置、种子、数据版本与硬件；报告实际 loss、token 数、耗时和样本；可复跑比较 | 待开始 |
| 首版发布 | 安装与文档从干净环境验收；许可审查完成；发布包和复现说明齐备 | 待开始 |

本次不下载语料/模型，不进行正式训练。CPU 配置检查通过不代表 GPU、
训练数值正确性或模型效果已通过验收。首版发布与当前框架版本号不是同一验收事件。
