# 实验索引与记录约定

[返回项目首页](../README.md) · [使用与技术文档](../docs/README.md)

当前已完成小语料训练闭环与 5 万篇数据规模对照，本阶段暂停追加训练，整理项目与复现资料。
最新结论见 [5 万篇、15000 步报告](tinystories-50k.md)：独立留出 loss 有改善，故事生成仍不稳定。
下表汇总历史报告，不代表本次重新运行了训练或评估。

## 实验总览

各轮模型均为 926,336 参数、2 层、128 维、4 头、上下文 128，沿用 2048 词表；训练使用 CPU FP32。
表中的 loss 是各轮终点结果，训练耗时不包含随后独立固定评估。

| 阶段与报告 | 训练故事数 | 更新步数 | 原验证 loss | 训练耗时 | 目的 |
| --- | ---: | ---: | ---: | --- | --- |
| [50 步试跑](tinystories-pilot.md) | 6133 | 50 | 5.6847 | 见报告 | 打通流程、单线程中断续训对照 |
| [1000 步初始基线](tinystories-baseline.md) | 6133 | 1000 | 3.2367 | 75.58 秒 | 初始训练结果与单提示词样本 |
| [四 checkpoint 固定评估](tinystories-evaluation.md) | 6133 | 1000（已有权重） | 3.2358 | 未重新训练 | 统一评估窗口，保留 40 条样本 |
| [5000 步实验](tinystories-5000.md) | 6133 | 5000 | 2.5907 | 约 343 秒 | 增加预算，评估 5 个保存点 |
| [15000 步实验](tinystories-15000.md) | 6133 | 15000 | 2.3680 | 16 分 27 秒 | 观察更长训练与训练/验证差距 |
| [5 万篇数据对照](tinystories-50k.md) | 50000 | 15000 | 2.3546 | 24 分 51 秒 | 固定模型与预算，扩充训练数据 |

50 步试跑与 1000 步初始基线采用单 seed、16 batches 的终点评估；后续固定评估采用
seed=1337/2026/42、每 seed 32 batches、batch=8、context=128。
1000 步的 3.2367 与 3.2358 来自同一权重、不同窗口集合，不代表又训练了一轮或模型变好。
5000 与 15000 步实验从头运行，学习率计划随总步数改变，不是单纯延长同一训练轨迹。
随机窗口有放回且可能重叠；评估 token 数不是独立样本数，也不代表遍历完整验证集。

## 最新数据对照

两轮 15000 步实验的模型、训练配置、词表与原验证文件保持一致，训练数据由 6133 篇扩大至 50000 篇。
各轮先按原验证 loss 选择 checkpoint，再对预留的 1000 篇故事进行一次固定窗口评价。

| 指标 | 原 6133 篇 | 扩充 50000 篇 |
| --- | ---: | ---: |
| 独立留出 loss | 2.7094 | 2.4873 |
| 独立留出 PPL | 15.02 | 12.03 |
| 最终十条生成遇到 EOS | 6/10 | 3/10 |
| 最终十条生成平均重复 token 4-gram | 0.73% | 2.41% |

留出集结果支持扩充数据改善这批未参与训练的故事上的预测表现；只有一个训练种子的结果，
不能据此声称统计显著或生成质量全面提升。EOS 次数与重复比例也不是故事质量评分。
训练集合不同，不能直接比较两轮训练 loss 的差值来推断同一集合上的改善。
完整方法与限制见[报告](tinystories-50k.md)，逐 seed 留出结果见[JSON](tinystories-50k-heldout.json)。

## 指标、样本与复现入口

固定提示词保存在 [tinystories-prompts.json](tinystories-prompts.json)，后续固定评估使用
temperature=0.8、top-k=20、top-p=1、最多新增 128 tokens；提示与生成种子保持一致。

| 实验 | 配置 | 指标与曲线 | 全部生成样本 |
| --- | --- | --- | --- |
| 50 步试跑 | [YAML](../configs/tinystories-pilot.yaml) | [试跑记录](tinystories-pilot.md) | 报告内样本 |
| 1000 步基线与固定评估 | [YAML](../configs/tinystories-cpu.yaml) | [CSV](tinystories-evaluation-metrics.csv) / [曲线](tinystories-evaluation-loss.png) | [40 条](tinystories-evaluation-samples.md) |
| 5000 步 | [YAML](../configs/tinystories-cpu-5000.yaml) | [CSV](tinystories-5000-metrics.csv) / [曲线](tinystories-5000-loss.png) | [50 条](tinystories-5000-samples.md) |
| 15000 步小语料 | [YAML](../configs/tinystories-cpu-15000.yaml) | [CSV](tinystories-15000-metrics.csv) / [曲线](tinystories-15000-loss.png) | [60 条](tinystories-15000-samples.md) |
| 5 万篇、15000 步 | [YAML](../configs/tinystories-cpu-50k.yaml) | [CSV](tinystories-50k-metrics.csv) / [曲线](tinystories-50k-loss.png) | [60 条](tinystories-50k-samples.md) |

- 数据来源、划分与准备步骤：[本地样本](../docs/tinystories-local.md)、[5 万篇扩充数据](../docs/tinystories-expanded.md)。
- 脚本：[小样本准备](../scripts/prepare_tinystories_sample.py)、[扩充数据](../scripts/expand_tinystories.py)、
  [50 步续训对照](../scripts/run_tinystories_pilot.py)、[固定评估](../scripts/evaluate_tinystories.py)、
  [独立留出评估](../scripts/evaluate_tinystories_heldout.py)、[绘图](../scripts/plot_tinystories_evaluation.ps1)。
- 精确命令、计时范围、运行时版本与检查范围以各轮报告为准。脚本有各自前置条件，不能仅凭拉取仓库就直接复现。

Git 保存报告、配置、小型指标与样本；语料、tokenizer、checkpoint 和运行输出被忽略，需另外准备或转移。
历史报告中的 Python 绝对路径应替换为当前设备的项目解释器。
新实验使用新输出目录，保留旧产物；不能直接修改旧配置的 max_steps 后恢复已经结束的实验。
早期报告的“下一步”保留当时判断，当前进度以本索引及[路线图](../docs/roadmap.md)为准。

## 新实验记录约定

每次运行至少记录以下内容；未测量、未执行的项目应明确注明：

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
| 生成样本 | 提示文本、seed、temperature/top-k/top-p、EOS/长度限制和实际输出 |
| 恢复与异常 | checkpoint、恢复位置、失败或偏离计划的情况 |

建议小型说明与可公开配置进入版本控制，运行产物放入 `experiments/runs/`
或项目 `outputs/`（已忽略）。不要提交大数据、权重、私密提示或虚构指标。
比较实验时说明变化因素，不把未运行的配置模板标注为已验证结果。
