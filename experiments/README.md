# 实验记录约定

已完成 [TinyStories 50 步 CPU 试跑](tinystories-pilot.md)，包括中断续训对照和固定样本验证。
[1000 步 CPU 初始基线](tinystories-baseline.md)也已完成；消融实验尚未开始。每次运行至少记录：

后续已完成 [四 checkpoint 固定评估](tinystories-evaluation.md)，包含训练/验证 loss 曲线和 40 条完整生成样本。
2026-10-25 已完成 [同结构 5000 步实验](tinystories-5000.md)，包含五个 checkpoint 的评估及 50 条完整生成样本。
同日已完成 [同结构 15000 步实验](tinystories-15000.md)，训练 16 分 27 秒，包含六个 checkpoint 的评估及 60 条完整生成样本。
随后完成 [5 万篇、15000 步数据对照](tinystories-50k.md)，训练 24 分 51 秒，包含六个 checkpoint、60 条样本及新旧模型的独立留出评价。

| 字段 | 要求 |
| --- | --- |
| 标识 | 实验名称、时间、Git revision 与未提交改动说明 |
| 配置 | 完整配置快照、实际命令、输出目录 |
| 随机性 | 随机种子、随机数与可复现设置 |
| 环境 | OS、Python、依赖版本、CPU/GPU 型号、显存/内存、CUDA 版本（若使用） |
| 数据 | 来源、许可、版本/校验值、划分、tokenizer 版本、dtype |
| 规模 | 实际处理的训练 token 数、步数、batch、上下文长度 |
| 评价 | 验证 loss、评价步数、聚合方式与验证 token 数 |
| 成本 | 实际耗时、计时范围，可选吞吐量和峰值显存 |
| 生成样本 | 提示文本、seed、temperature/top-k、EOS/长度限制和实际输出 |
| 恢复与异常 | checkpoint、恢复位置、失败或偏离计划的情况 |

建议小型说明与可公开配置进入版本控制，运行产物放入 `experiments/runs/`
或项目 `outputs/`（已忽略）。不要提交大数据、权重、私密提示或虚构指标。
比较实验时说明变化因素，不把未运行的配置模板标注为已验证结果。
