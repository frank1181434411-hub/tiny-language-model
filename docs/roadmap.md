# 路线图与验收

| 阶段 | 验收标准 | 当前状态 |
| --- | --- | --- |
| 基础框架 | 可安装；配置严格校验；CLI 帮助/版本/配置检查及运行入口可用；Ruff 与 pytest 通过 | 初始框架见 [检查记录](local-checks.md)；全部原预留命令已接通 |
| 源码迁移 | 按迁移映射整理模块；确认归属和依赖；消除演示入口与导入副作用 | loss、optimizer、scheduler、采样、TransformerLM、checkpoint、BPE、Tokenizer、预处理、文件加载与单提示词生成已迁移 |
| 正确性验证 | tokenizer 往返和特殊边界；模型组件/loss/优化器数值及梯度对照；调度、采样与 checkpoint 测试 | loss、optimizer 的 CPU 检查见各迁移记录；[调度](scheduler.md)、[采样](data.md)、[基础层](layers.md)、[注意力](attention.md)、[完整模型](transformer.md)、[checkpoint](checkpoint.md)、[BPE](bpe.md) 与 [Tokenizer](tokenizer.md) 见对应记录，其余待验证 |
| 训练闭环 | 小数据可预处理、训练、验证、保存、恢复、续写；恢复状态与步数语义明确 | [预处理 CLI](preprocess-cli.md)、[训练/验证 CLI](training-cli.md) 与 [生成 CLI](generation.md) 已接通并测试；历史数据指纹绑定待实现 |
| 基线实验 | 固定配置、种子、数据版本与硬件；报告实际 loss、token 数、耗时和样本；可复跑比较 | 待开始 |
| 首版发布 | 安装与文档从干净环境验收；许可审查完成；发布包和复现说明齐备 | 待开始 |

本次不下载语料/模型，不进行正式训练。CPU 配置检查通过不代表 GPU、
训练数值正确性或模型效果已通过验收。首版发布与当前框架版本号不是同一验收事件。

## 接下来的工作

1. 为 checkpoint 绑定 tokenizer 和训练/验证数据的内容指纹，记录 eval_batches 等运行参数，
   防止同一路径内容被替换后误续训或使用同大小但不同的 tokenizer 生成。
2. 用固定划分的小型真实语料建立可复现基线，记录配置、数据版本、loss、token 数、耗时和生成样本。
3. 在目标 GPU 上验证精度、显存、吞吐量与续训，再根据测量决定 AMP、梯度累积和 KV cache 的优先级。
4. 根据实际需要补批量生成、保持边界语义的流式文本处理；完成干净环境安装和发布验收。

当前文本生成是单提示词、每步重算上下文，top-p 默认关闭；最新 CPU 全量验收为 1300 项测试通过，
详见 [生成说明与验收](generation.md)。这不代表正式语料效果或 GPU 已完成验收。
