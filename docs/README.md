# 使用与技术文档

[返回项目首页](../README.md) · [实验索引](../experiments/README.md)

## 使用入口

| 需要完成的工作 | 文档 |
| --- | --- |
| 创建环境、安装依赖、校验配置 | [项目首页](../README.md)、[依赖与历史环境](dependencies.md) |
| 训练 tokenizer、准备单文件或多文件数据 | [预处理命令](preprocess-cli.md) |
| 小型训练闭环、恢复训练、独立验证 | [训练与验证命令](training-cli.md) |
| 加载权重续写文本 | [文本生成](generation.md) |
| 理解恢复限制与历史数据校验 | [checkpoint 输入绑定](checkpoint-binding.md) |
| 准备或复现 TinyStories 数据 | [本地样本](tinystories-local.md)、[5 万篇数据](tinystories-expanded.md) |
| 查找训练结果、曲线、样本与运行配置 | [实验索引](../experiments/README.md) |

阅读历史命令时，将绝对 Python 路径替换为当前项目环境。拉取 Git 不会自动取得被忽略的
数据和模型权重；数据准备、已有产物与配置匹配要求见对应指南。

## 实现与接口

| 模块 | 文档 |
| --- | --- |
| 整体数据流 | [架构](architecture.md) |
| 文本与词表 | [BPE](bpe.md)、[Tokenizer](tokenizer.md) |
| 编码产物、文件加载与采样 | [预处理](preprocess.md)、[数据](data.md) |
| 模型 | [基础层](layers.md)、[注意力](attention.md)、[Transformer](transformer.md) |
| 数值计算与参数更新 | [损失函数](losses.md)、[优化器](optimizer.md)、[学习率调度](scheduler.md) |
| 训练 API | [训练与验证](training.md) |
| 状态保存 | [checkpoint 格式与恢复](checkpoint.md)、[输入绑定](checkpoint-binding.md) |

这些文档同时保留迁移时的接口约定、精度处理、边界条件及检查记录。
某一阶段通过的检查不代表后续所有版本或其他硬件已完成相同验收。

## 项目状态与历史记录

- [路线图](roadmap.md)：阶段验收、当前进度及待办。
- [迁移映射](migration.md)：旧源码对应关系与待复核事项。
- [来源与许可](provenance.md)：课程背景、代码归属及尚待核实的许可事项。
- [初始框架检查](local-checks.md)：框架建立阶段的历史记录。
- [输入绑定阶段验收](checkpoint-binding.md)：1349 项 CPU 测试的记录与范围。

当前已完成 CPU 训练闭环和数据规模对照；生成故事仍不稳定，GPU 与干净环境发布验收待完成。
各轮实验的原始结论和未来计划按当时状态保留，当前实验进度请从[实验索引](../experiments/README.md)进入。
