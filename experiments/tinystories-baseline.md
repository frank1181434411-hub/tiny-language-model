# TinyStories 1000 步 CPU 初始基线

2026-10-23 使用 configs/tinystories-cpu.yaml 从头训练完成；没有恢复 50 步试跑权重。
环境与数据沿用 [试跑记录](tinystories-pilot.md) 和 [数据准备记录](../docs/tinystories-local.md)：
Ryzen 9 7945HX、Windows、Python 3.12.10、PyTorch 2.11.0+cpu、FP32，OMP/MKL 线程数均为 1。
运行时 HEAD 为 51fcb0c19ddbca4f38e93d201a8011791b3fb242，配置、脚本与文档存在未提交改动。
未测量峰值内存，没有 GPU 验证。

## 配置与结果

模型仍为 926,336 参数：2048 词表、2 层、128 维、4 个头、SwiGLU 352 维、上下文 128。
batch=8、seed=42、1000 步，预热 50 步，学习率从 0.001 余弦衰减至 0.0001，
weight_decay=0.1、grad_clip=1.0，每 100 步验证 4 batches、每 250 步保存。

| 固定窗口独立验证 | loss |
| --- | ---: |
| 初始模型（上次记录，同结构和初始化种子） | 7.67040947 |
| 50 步试跑 | 5.68465063 |
| 本次 1000 步 | 3.23673984 |

三者使用相同验证文件、seed=1337、16 batches、batch=8、context=128，
共 16,384 个目标 token；是随机窗口估计，不是遍历全验证集。
50 步与 1000 步具有不同预热长度和学习率计划，这不是只改变训练步数的严格消融。

训练命令实际耗时 **75.58 秒**，包含进程启动、数据校验、训练内验证和保存。
处理训练目标 token 数为 **1,024,000**，按命令墙钟时间计算约 **13,548 tokens/s**。
窗口有放回采样，不能将该数解释为独立 token 覆盖量或一次完整 epoch。
独立验证另用 4.80 秒，生成另用 3.46 秒；未计入上述训练耗时。

## 固定提示生成

seed=42、temperature=0.8、top-k=20、top-p=1，最多新增 64 tokens。
完整输出（包含提示词）：

```text
Once upon a time, there was a little girl named Lily. She wanted to try the water for the water.
The cat saw the cat and the bird. They played together and had a big smile. They liked the ball and play with. One day, the wind came to his mom. The big man was happy and wanted to play with it.
Tim
```

本次因达到 64 个新增 token 而停止，未遇到 EOS。较 50 步已有较完整的句子，
但存在重复、语法错误和角色/情节跳跃；单个样本不能证明稳定的故事生成质量。

## 产物与命令

目录：outputs/tinystories-cpu-baseline-v1/，由 Git 忽略，包含：
config.source.yaml、train.json、train.stderr.jsonl、evaluate.json、generate.json、
对应 stderr 文件、timings.json，以及第 250、500、750、1000 步 checkpoint。
timings.json 保存实际参数、退出码与计时；checkpoint 包含完整配置、运行绑定和训练状态。
本次训练、独立验证、生成均退出成功；未再次进行 1000 步连续/续训双跑对照，
此前单线程 50 步的逐位续训核对结果保留在试跑记录中。

Git Bash 中评估或生成现有最终模型：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PYTHON" -m tiny_lm evaluate --config configs/tinystories-cpu.yaml --checkpoint outputs/tinystories-cpu-baseline-v1/step_00001000.pt --dtype float32 --eval-batches 16 --seed 1337
"$PYTHON" -m tiny_lm generate --config configs/tinystories-cpu.yaml --checkpoint outputs/tinystories-cpu-baseline-v1/step_00001000.pt --prompt 'Once upon a time, there was a little girl' --max-new-tokens 64 --temperature 0.8 --top-k 20 --seed 42
```

重新训练应复制 YAML 并设置新的 output_dir，再执行 train，避免覆盖现有结果。
当前已达到 max_steps；继续训练需要另行设计学习率与恢复配置，不能直接改 max_steps 后强行加载。
下一步建议增加固定评估窗口与多条提示词，观察训练/验证差距，再决定是否扩充语料或训练预算。
