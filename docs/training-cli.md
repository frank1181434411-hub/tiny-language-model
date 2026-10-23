# 训练与验证命令

`train` 和 `evaluate` 已连接配置、tokenizer、只读 token 文件与现有数组接口。
模块入口 `python -m tiny_lm` 和控制台入口 `tiny-lm` 行为一致。
保持模型、loss、优化器、学习率和采样算法；checkpoint 已升级为 v2 输入绑定，兼容读取 v1。

## Git Bash 使用

在当前机器使用已有环境；其他机器替换 Python 路径：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe

"$PYTHON" -m tiny_lm train --config configs/smoke.yaml --stop-after 2
"$PYTHON" -m tiny_lm train --config configs/smoke.yaml --resume outputs/smoke/step_00000002.pt
"$PYTHON" -m tiny_lm evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/step_00000004.pt --eval-batches 4 --seed 42
```

上述命令需要配置中的 tokenizer、训练/验证数据及旁置 JSON 已存在。
`smoke.yaml` 采用 256 字节词表，不应直接使用先前 300 词表的示例 tokenizer。
可以先执行下面的小语料准备步骤，再执行上述训练命令：

```bash
mkdir -p artifacts/smoke data/smoke
printf 'A small validation text. The model predicts the next token.\n' > data/smoke/validation.txt

"$PYTHON" -m tiny_lm train-tokenizer --input examples/toy_corpus.txt --vocab-size 256 --output artifacts/smoke/tokenizer.json
"$PYTHON" -m tiny_lm prepare-data --input examples/toy_corpus.txt --tokenizer artifacts/smoke/tokenizer.json --output data/smoke/train.bin
"$PYTHON" -m tiny_lm prepare-data --input data/smoke/validation.txt --tokenizer artifacts/smoke/tokenizer.json --output data/smoke/val.bin
```

仅在对应演示文件尚不存在时运行准备步骤；已有 tokenizer 或数据不会被预处理命令覆盖。
训练会创建配置指定的输出目录；同名 checkpoint 沿用原有原子替换语义。
新的实验应使用新的 output_dir，续训则保留原配置并指定 `--resume`。
此例只有 4 次更新，验证文本是独立编写的流程测试文本，不代表有效训练或泛化评估。

预处理需要 `.[tokenizer,data]`；训练/验证需要 `.[compute]`。
训练/验证只加载 tokenizer 元信息，不编码原文，所以无需在运行时导入 regex。
帮助、版本和配置检查仍保持轻量，不要求 PyTorch 可用。

## 训练参数与输出

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--config` | 必填 | YAML 路径 |
| `--resume` | 无 | 从 checkpoint 恢复模型、优化器、已完成步数和训练/验证随机状态 |
| `--stop-after` | 配置的 max_steps | 本次最多完成的总更新次数，范围 0 到 max_steps，不是追加步数 |
| `--dtype` | `float32` | 模型参数精度：float16、bfloat16、float32、float64 |
| `--eval-batches` | 4 | 每次验证的随机窗口 batch 数，必须为正整数 |
| `--allow-unbound-checkpoint` | 关闭 | 显式允许旧 checkpoint 缺失的输入绑定；已知不匹配仍报错 |

训练按配置中的 batch、学习率、验证/保存间隔和设备运行；没有 AMP、GradScaler 或梯度累积。
每个步骤在更新、间隔验证及必要保存成功后，将一行 JSON 指标写到 stderr 并立即刷新。
字段为 step、learning_rate、train_loss 和 validation_loss；没有验证时后者为 null。
train_loss 来自更新前的训练 batch，validation_loss 来自更新后的模型。
可通过 Shell 重定向保留日志，例如：

```bash
"$PYTHON" -m tiny_lm train --config configs/smoke.yaml > training-summary.json 2> training-progress.jsonl
```

stderr 也包含错误诊断，因此失败时最后一行可能不是 JSON。重复使用重定向文件名会截断旧日志，
续训应使用新的日志文件名或显式追加。CLI 本身不管理日志文件或历史清理。
成功时 stdout 输出最终 JSON 摘要：已完成更新次数、本次保存路径、最后一步指标、dtype 和 eval_batches。
已达到终点的续训不更新、不另存文件，最后一步指标为 null，checkpoint 列表为空。

## 独立验证

`evaluate` 必填 `--config` 和 `--checkpoint`；支持相同的 `--dtype`、`--eval-batches`，
以及 `--seed`（默认 config.training.seed）和 `--allow-unbound-checkpoint`。batch_size 和 context_length 来自配置。

- 只读取 tokenizer 和验证数据；训练文件及配置中的 output_dir 不必存在。
- 复用现有 checkpoint 加载与校验接口，临时构造 AdamW 加载并验证其状态后释放，
  不进行 optimizer 更新，不恢复 checkpoint 中的训练 RNG，也不写 checkpoint。
  目前仍会读取完整 checkpoint，模型与优化器状态共同影响加载时的内存峰值。
- 严格检查模型构造配置、权重 dtype/shape 等；允许使用不同验证路径、输出目录和运行设备。
  训练续训仍要求完整 ProjectConfig 一致。
- 使用独立 CPU generator 从指定 seed 开始采样；相同模型、数据、采样设置和运行条件可复跑。
  其采样起点不同于训练期间不断推进的验证 generator，因此不承诺与训练日志中的 loss 相等。
- 返回 checkpoint 路径、已完成更新次数、loss、num_batches、num_tokens、seed 和 dtype。
  loss 是随机窗口的 token 加权平均交叉熵，不是完整语料遍历；重复抽取的 token 也计入 num_tokens。

## 输入检查与续训边界

YAML 中相对路径以 YAML 所在目录为准；命令行的 config、resume、checkpoint 相对路径以当前目录为准。
运行前核对实际 tokenizer 词表大小与 model.vocab_size；加载数据时强制核对 tokenizer 指纹、
完整数据哈希和全部 ID 范围，并要求至少 context_length+1 个 token。
只读 memmap 在成功或异常退出时都关闭，不遗留文件映射。

CPU/CUDA 为当前支持的设备；不可用 CUDA、MPS 或尚未实现的 GELU 在运行前明确报错。
普通输入、文件、配置、checkpoint 和运行数值错误返回退出码 2，在 stderr 输出诊断，
不输出成功摘要；缺少依赖会给出安装提示。原有配置加载器仍沿用中文配置错误信息，
本次新增参数和运行校验错误使用英文。

精确续训要求相同完整配置、实际数据内容、dtype、eval_batches 和软件/硬件条件。
checkpoint v2 记录并核对实际训练/验证 token 内容、tokenizer 指纹、dtype 和 eval_batches。
同一路径下同时替换数据与元信息也会被续训检查发现；同大小却内容不同的 tokenizer 同样拒绝。
独立 evaluate 只核对 tokenizer 绑定，允许新的验证集与不同 eval_batches、seed。
旧 v1 或未绑定的 v2 默认不能用于这些命令；确认旧输入后可显式使用 --allow-unbound-checkpoint，
该选项只允许缺失信息，不绕过已记录的不匹配，详见 [输入绑定与兼容](checkpoint-binding.md)。
数据集划分、训练/验证泄漏检查和大语料性能优化仍待实现；不承诺跨设备逐位一致或 GPU 已验证。

## 数组 API 的指标回调

`train(...,on_step=None)` 新增可选回调，默认行为不变。
回调接收不可变的 TrainingMetrics，在本步验证和必要保存完成后调用；CLI 用它即时输出指标。
回调不得改变训练对象或随机状态；回调失败会中止调用，不回滚已完成更新或已成功保存的文件。

## 本地验收

以下保留训练/验证 CLI 接入时的检查记录；最新全项目验收为 1349 项通过，见 [输入绑定](checkpoint-binding.md)。

2026-10-22，Windows、Python 3.12.10，使用 `D:/Python/.venvs/tiny-language-model`：

- 全项目 1180 项测试通过，119.28 秒；关闭 pytest 缓存，使用独立临时目录。
- 新增 45 项训练/验证 CLI 与回调测试，移除 train/evaluate 对应的 8 项占位测试。
- 模块和控制台入口均验证 FP32/FP64 训练与中断续训；训练 loss、最终全部权重、optimizer
  和两个采样器状态与连续数组训练精确一致。原有四种精度的数组续训测试继续通过。
- 独立验证 loss 与数组接口精确一致；重复执行相同采样设置得到相同结果；
  删除训练数据、指定不存在的输出目录仍可验证，已有 checkpoint 字节保持不变。
- 覆盖词表不符、同大小但不同 tokenizer、损坏数据哈希、缺少元信息、过短/空数据、
  非法参数、设备/模型不支持、损坏 checkpoint、恢复配置/dtype 不符及映射释放。
- 验证 on_step 接收的步数和已保存文件一致；Ruff、格式、紧凑码风及导入审计通过。
- 未下载语料、安装依赖或开展正式训练；GPU、远程 CI 和 Linux 未在本次验证。
