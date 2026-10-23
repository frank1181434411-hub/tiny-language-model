# TinyStories 本地 50 步 CPU 试跑

2026-10-23 完成真实 CLI 训练、25 步中断续训、独立验证、生成和连续训练对照。
这是小规模流程与速度试跑，尚不是充分训练的模型基线。

## 环境与配置

- CPU：AMD Ryzen 9 7945HX with Radeon Graphics；系统报告 32 个逻辑处理器。
- Windows 11、Python 3.12.10、PyTorch 2.11.0+cpu；本环境 CUDA 不可用，没有验证 GPU。
- OMP_NUM_THREADS=1、MKL_NUM_THREADS=1；父进程另设 torch.set_num_threads(1)。FP32。
- 未测量峰值内存；系统 WMI 硬件查询受限，CPU 型号改从硬件注册表读取。
- 运行时 Git HEAD：`51fcb0c19ddbca4f38e93d201a8011791b3fb242`；存在未提交的数据准备/试跑脚本、配置和文档改动。
  本地 report.json 记录运行时 working tree 状态，不能将 HEAD 单独当作完整实验快照。
- 配置：[tinystories-pilot.yaml](../configs/tinystories-pilot.yaml)。词表 2048、上下文 128、
  2 层、d_model=128、4 个头、SwiGLU d_ff=352，共 **926,336 个参数**。
- batch=8、50 次更新、seed=42；最大学习率 0.001、最小 0.0001、预热 5 步、
  weight_decay=0.1、grad_clip=1.0；每 10 步验证 4 batches，每 25 步保存。

输入来自 [本地样本记录](../docs/tinystories-local.md)：训练 6133 篇 / 1,341,153 tokens，
验证 323 篇 / 69,576 tokens；使用仅在训练集上拟合的 2048 BPE。
源文件哈希、划分方式与上游版本未核实的限制见该记录，未重新下载或混入其他数据。
checkpoint v2 与 report.json 均记录两份实际 token 指纹及 tokenizer 指纹。

## 结果与计时口径

| 项目 | 实测值 |
| --- | ---: |
| 初始独立验证 loss | 7.67040947 |
| 50 步后独立验证 loss | 5.68465063 |
| 主训练处理 token 数 | 51,200 |
| 前 25 步 CLI 墙钟时间 | 7.45 秒 |
| 恢复到 50 步 CLI 墙钟时间 | 7.33 秒 |
| 两段训练 CLI 合计 | 14.77 秒 |
| 按上述合计时间计算的吞吐量 | 约 3,465 tokens/s |
| 连续 50 步 API 对照，禁用保存 | 3.94 秒 |

初始与最终验证都使用 seed=1337、batch=8、context=128、16 batches，
从同一份验证数据抽取相同窗口，总计 16,384 个目标 token；这是随机窗口估计，未遍历全验证集。
训练日志中的验证采样器持续推进，不能把其 loss 与上述固定窗口值直接当作同一指标。

训练 token 数为 `50*8*128`，不包括验证、生成或额外的连续训练对照；窗口采样有放回，
不表示看过 51,200 个互不重复的位置。步数计入第一个学习率为 0 的预热步骤。
CLI 时间包含两次 Python/PyTorch 启动、数据校验、初始化、训练内验证和 checkpoint IO；
不包含独立验证、生成、初始评估和连续训练对照。API 对照时间不含进程启动与权重保存，
不能将两种计时口径直接比较为性能优化效果。

## 续训核对与四线程观察

主流程先保存 step_00000025.pt，再在独立 CLI 进程恢复至 step_00000050.pt。
另以相同配置、种子和数据从头连续跑 50 步，对照结果：

- 全部模型权重逐位一致。
- optimizer 全部状态逐位一致。
- 每一步 train loss、validation loss、学习率等指标一致。
- 训练和验证两个独立采样器最终状态一致。

最初的四线程试跑保留在 `outputs/tinystories-pilot-v1/`。其训练/验证/生成均成功，
但连续训练对照未通过逐位一致检查：额外诊断发现第 5 步 loss 即有约 4.77e-7 差异，
发生在第 25 步恢复点之前；第 50 步权重最大绝对差约 4.77e-7，两个采样器状态一致。
因此不能将差异归因于 checkpoint 恢复；本次未进一步定位具体数值算子，
也未将多线程逐位复现标注为通过。固定单线程后的核对全部通过，正式记录位于 v2 目录。
这不保证其他硬件、依赖版本或线程设置下逐位一致。

## 生成样本

提示：`Once upon a time, there was a little girl`。
seed=42、temperature=0.8、top-k=20、top-p=1（不启用核采样过滤），最多新增 64 tokens。
实际新增 53 tokens 后遇到 EOS。完整输出为：

```text
Once upon a time, there was a little girl to. He. They was a the, they the.
.
. They said her a the was. They to her. He the
Once. They it the the hes. It.
 not the and. He, and as.<|endoftext|>
```

模型开始偏向常见词与标点，但仍有明显重复和语法错误；loss 下降不能解释为已经能写连贯故事。

## 产物与复现

`outputs/tinystories-pilot-v2/` 包含：

- 两份 checkpoint：step_00000025.pt、step_00000050.pt。
- 配置原文和解析后的路径快照。
- 两段训练的 JSON 摘要和逐步日志、独立验证结果、生成结果。
- report.json：环境、命令、耗时、数据绑定、初始/最终指标、逐步日志和续训核对结果。

所有本地数据、权重和运行日志均被 Git 忽略。配置、脚本及这份摘要进入版本控制。
复现脚本要求输出目录不存在；本机已经跑完，如需重跑，复制 YAML 并修改 output_dir 为新目录，
再执行（保持 YAML 在 configs/ 中，以保持相对数据路径的含义）：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe
"$PYTHON" scripts/run_tinystories_pilot.py --config configs/tinystories-pilot.yaml
```

脚本会运行两段 CLI 训练、固定样本评估与生成，以及额外一次完整连续训练对照。
它设置进程内线程环境，不修改系统环境。失败保留文件供检查，不覆盖旧实验。
本次新增配置/脚本经实际运行及 Ruff、格式、紧凑码风和空白检查；核心算法未改，未重跑全部 1349 项单元测试。

## 下一轮计划

更新：下列计划现已执行，实测结果见 [1000 步基线](tinystories-baseline.md)；以下保留当时计划与估计。

已准备 [tinystories-cpu.yaml](../configs/tinystories-cpu.yaml)，**本轮尚未执行**：
同样模型，1000 步，预热 50 步，每 100 步验证、250 步保存，共处理 1,024,000 个训练目标 token。
按本次不保存的 API 对照线性估计，计算部分约 79 秒；实际还含初始化和 IO，先预留约 1～3 分钟，
该估计不是已测得的 1000 步耗时，也不保证该步数足以收敛。

必须从头开始；50 步试跑的 max_steps 与学习率计划不同，不能修改配置后直接恢复其 checkpoint。
在 Git Bash 中运行下一轮：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p outputs/tinystories-cpu-baseline-v1
"$PYTHON" -m tiny_lm train --config configs/tinystories-cpu.yaml --dtype float32 --eval-batches 4 > outputs/tinystories-cpu-baseline-v1/train-summary.json 2> outputs/tinystories-cpu-baseline-v1/train-progress.jsonl
```

该命令适用于尚未开始的下一轮；重复执行会重训并覆盖同名日志/快照，应改用新配置和目录。
下一轮仍应沿用上述固定评估种子与生成提示记录结果，再判断是否增加步数或调整模型。
