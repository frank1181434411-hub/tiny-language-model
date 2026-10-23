# tiny-language-model

一个从 Stanford CS336 Assignment 1 学习成果出发的个人语言模型项目。
已有自行实现的 BPE、Tokenizer、Transformer、损失函数、AdamW、调度、采样、
checkpoint 和生成源码，**目前已迁入 BPE、Tokenizer、文本预处理、数据加载、完整 TransformerLM、训练/验证、checkpoint 与文本生成；预处理、训练/验证和生成 CLI 已接通**。长期目标是提供可安装、可复现的
文本预处理、训练、验证、保存与恢复训练、文本续写流程。

当前版本 `0.1.0` 已完成工程框架：`src` 布局、严格的 YAML/dataclass 配置、
CLI、测试、Ruff、CPU CI 与迁移文档，并提供手写交叉熵 `tiny_lm.losses.cross_entropy`、
`tiny_lm.optim.AdamW`、`tiny_lm.optim.clip_gradient_norm` 和
`tiny_lm.schedule.get_lr_cosine_schedule`、`tiny_lm.data.get_batch`。
`tiny_lm.model.layers` 提供 Linear、Embedding、RMSNorm、silu_fn 和 SwiGLU。
`tiny_lm.model.attention` 提供 RotaryPositionalEmbedding、softmax、scaled_dot_product_attention
和 CausalSelfAttention。`tiny_lm.model.transformer` 提供 TransformerBlock 和 TransformerLM，
可通过 `TransformerLM.from_config(config.model,device=...,dtype=...)` 构造。
Python API 已支持文本编码为带元信息的 token 文件、只读加载、模型前向/反向、数组训练/验证和中断续训。
**CLI 已支持帮助、版本、配置校验、BPE 训练、文本预处理、模型训练/续训、独立验证和单提示词生成；没有正式训练结果或已验证基线。**
`smoke.yaml` 可在准备匹配数据后检查小型 CPU 流程，`baseline.example.yaml` 仅为待硬件和实验确定的模板。

## 本地安装（Windows PowerShell）

需要 Python 3.11 或以上。建议用标准 Windows CPython 创建项目独立环境：

```powershell
Set-Location D:\Python\tiny-language-model
python --version
python -m venv .venv
$ProjectPython = ".\.venv\Scripts\python.exe"
# MSYS2 Python 创建的环境使用 bin 而非 Scripts：
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = ".\.venv\bin\python.exe"
}
& $ProjectPython -m pip install -e ".[dev]"
```

无需激活环境即可执行上述命令，不修改全局 Python 包、CUDA 或驱动。
MSYS2 的 wheel 兼容性可能与标准 Windows CPython 不同；安装失败时保留真实错误，
不要将依赖安装失败当作检查通过。
后续 Linux GPU 环境使用 `.venv/bin/python`，仍通过项目环境安装依赖。

基础框架运行依赖仅有 PyYAML，开发依赖为 pytest 和 Ruff。
源码迁移需要的 PyTorch、NumPy、regex 及课程测试依赖按需单独安装，
见 [迁移依赖环境](docs/dependencies.md)。它们不会随框架默认安装；
未来 GPU 环境按实际硬件选择 PyTorch 构建，不预设 CUDA wheel 来源。
`compute` 可选依赖组声明计算组件所需的 PyTorch 和数据采样所需的 NumPy；
CPU 安装方法与数值验证说明见 [loss 迁移记录](docs/losses.md)
和 [optimizer 迁移记录](docs/optimizer.md)。
学习率调度只依赖标准库 `math`，接口与边界约定见 [scheduler 迁移记录](docs/scheduler.md)。
数据采样支持一维整数数组、memmap 和可选 CPU generator，见 [data 迁移记录](docs/data.md)。
导入 `tiny_lm.data` 不加载计算依赖；文件加载时使用 NumPy，采样时另使用 PyTorch。
基础层的初始化与精度约定见 [基础层迁移记录](docs/layers.md)。
注意力的精度、mask 与位置约定见 [注意力迁移记录](docs/attention.md)。
完整模型的结构、前馈选项、精度与配置构造见 [Transformer 迁移记录](docs/transformer.md)。
checkpoint 的格式、原子保存、随机状态与精确续训验证见 [迁移记录](docs/checkpoint.md)。
数组训练/验证接口、日志、保存间隔和使用示例见 [训练记录](docs/training.md)。
显式导入 `tiny_lm.optim`、`tiny_lm.model.layers` 或 `tiny_lm.model.attention` 会加载 torch；
基础包、模型包入口、配置与 CLI 保持轻量导入。

BPE 训练 `tiny_lm.tokenizer.bpe.train_bpe` 与版本化保存/加载已迁移，
可安装 `.[dev,tokenizer]` 使用；接口、边界和测试见 [BPE 迁移记录](docs/bpe.md)。
`tiny_lm.tokenizer.tokenizer.BPETokenizer` 已提供 encode/decode、独立记录迭代编码和
原生文件保存加载，见 [Tokenizer 迁移记录](docs/tokenizer.md)。
`tiny_lm.preprocess.process_corpus` 默认处理单个文件，支持显式多文件列表；
`tiny_lm.data.load_token_data` 校验元信息后只读加载，见 [预处理迁移记录](docs/preprocess.md)。
只使用 BPE、Tokenizer 和预处理时安装 `.[dev,tokenizer,data]`，无需 PyTorch。
`tokenizer` extra 提供 regex，`data` extra 提供 NumPy；基础安装仍仅需 PyYAML。

## 当前可运行的命令

以下命令继续使用上一步的 `$ProjectPython`：

```powershell
& $ProjectPython -m tiny_lm --help
& $ProjectPython -m tiny_lm --version
& $ProjectPython -m tiny_lm check-config --config .\configs\smoke.yaml
& $ProjectPython -m tiny_lm check-config --config .\configs\baseline.example.yaml
& $ProjectPython -m ruff check .
& $ProjectPython -m ruff format --check .
& $ProjectPython -m pytest

# 安装后也提供控制台入口，无需修改 PATH：
$TinyLm = Join-Path (Split-Path $ProjectPython) "tiny-lm.exe"
& $TinyLm --help
& $TinyLm --version
& $TinyLm check-config --config .\configs\smoke.yaml
```

若已激活项目环境，可直接使用 `tiny-lm --help`、`tiny-lm --version`、
`tiny-lm check-config --config configs/smoke.yaml` 和 `python -m tiny_lm`。
不带子命令会显示参数错误；查看用法请传 `--help`。

`train-tokenizer` 和 `prepare-data` 已可运行，参数及 Git Bash 示例见 [预处理命令](docs/preprocess-cli.md)。
`train` 和 `evaluate` 已接通文件加载、checkpoint 和独立验证采样，见 [训练/验证命令](docs/training-cli.md)。
`generate` 从 checkpoint 自动读取模型结构和精度，支持温度、可选 top-k/top-p、EOS 和独立采样种子，见 [生成命令](docs/generation.md)。
生成返回包含原提示的完整文本与 token IDs，不读取训练数据，也不写 checkpoint。
例如 `& $ProjectPython -m tiny_lm train --help` 可查看训练命令参数。

## 预处理示例（PowerShell，默认单文件）

从项目根目录执行；示例只使用已有的小型演示语料，不训练语言模型。
输出放入 Git 忽略的 `artifacts/preprocess-cli-demo/`（CLI）或 `artifacts/preprocess-demo/`（Python API）。同名输出已存在会明确报错，
重复执行时应改用新的输出目录。

```powershell
& $ProjectPython -m pip install -e ".[dev,tokenizer,data]"
New-Item -ItemType Directory -Force artifacts/preprocess-cli-demo | Out-Null
& $ProjectPython -m tiny_lm train-tokenizer --input examples/toy_corpus.txt --vocab-size 300 --special-token "<|endoftext|>" --output artifacts/preprocess-cli-demo/tokenizer.json
& $ProjectPython -m tiny_lm prepare-data --input examples/toy_corpus.txt --tokenizer artifacts/preprocess-cli-demo/tokenizer.json --output artifacts/preprocess-cli-demo/train.bin
```

也可以直接调用 Python API，下面使用不同的输出目录：

```powershell
@'
from pathlib import Path
from tiny_lm.tokenizer.bpe import train_bpe
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.preprocess import process_corpus
from tiny_lm.data import load_token_data

source=Path("examples/toy_corpus.txt")
output=Path("artifacts/preprocess-demo")
output.mkdir(parents=True,exist_ok=True)
special_tokens=["<|endoftext|>"]
vocab,merges=train_bpe(source,300,special_tokens)
BPETokenizer(vocab,merges,special_tokens).save(output/"tokenizer.json")
tokenizer=BPETokenizer.from_file(output/"tokenizer.json")
data_path=process_corpus(source,output/"train.bin",tokenizer)
data=load_token_data(data_path,tokenizer=tokenizer)
print("tokens:",len(data),"dtype:",data.dtype,"vocab:",len(tokenizer.vocab))
'@ | & $ProjectPython -
```

默认将单个文件作为完整文本，严格读取 UTF-8 并保留原始换行，不追加 EOS。
产生 `train.bin` 和 `train.bin.json`：前者为小端 uint16/uint32 token 数组，
后者记录 dtype、token 数、数据和 tokenizer 指纹及输入顺序。
加载默认校验内容哈希与所有 ID 范围；非空数据返回只读 NumPy memmap。

多文件接口已保留并实现。仅将 `process_corpus` 的第一个参数换为有序列表，例如
`[Path("corpus/a.txt"),Path("corpus/b.txt")]`，即可按列表顺序逐文件编码并写入同一份数据。
`train_bpe` 本身仍接收单个训练文本路径；这里的多文件接口针对预处理阶段。
各文件不会跨边界执行 BPE 合并；需要在每个文件末尾追加 EOS 时显式传
`eos_token="<|endoftext|>"`。现有训练采样仍可能跨文档取窗口。
对于每行本来就是独立记录的语料，可显式选择 `record_mode="line"`；
它不保证与整文件编码相同，不能把任意文本块当成独立记录。

默认模式仍需容纳单个完整文件及其编码结果，尚未实现任意分块的等价流式编码。
训练和验证语料应分别处理，BPE 只用训练文本拟合；模型词表大小使用实际 `len(tokenizer.vocab)`。
加载的数组可传入现有 [训练/验证 Python 接口](docs/training.md)，
也可通过训练 CLI 自动读取配置路径并校验文件；checkpoint v2 已绑定历史数据与 tokenizer 指纹，
并在续训时核对 eval_batches 和 dtype，见 [输入绑定与旧文件兼容](docs/checkpoint-binding.md)。

本地最新检查：Windows、Python 3.12.10，全量 **1349 项测试通过**；
本次新增 49 项输入绑定测试，覆盖内容替换、参数不符与旧 checkpoint 的显式兼容。
Ruff 静态检查、格式检查通过，详见 [输入绑定验收](docs/checkpoint-binding.md)。远程 CI、GPU、Linux 与大语料性能尚未在本次验证。

## 配置约定

- YAML 顶层必须包含 `model`、`data`、`training`、`output`。
  当前所有字段均为必填；未知字段、重复字段和非映射结构会报错。
- 配置内的**所有相对路径统一相对于配置文件所在目录**。
  例如 `configs/smoke.yaml` 中 `../data/smoke/train.bin` 指向项目的
  `data/smoke/train.bin`。绝对路径保持绝对路径语义，并经 `pathlib` 规范化。
  不展开环境变量和 `~`；跨平台共享配置建议使用 `/` 分隔的相对路径。
  `--config` 参数本身的相对路径按调用时工作目录定位。
- `check-config` 输出全部规范化参数与绝对路径，仅验证结构与参数；
  不检查数据/tokenizer 是否存在、不检查硬件、不创建 `output_dir`。
  实际文件存在性、词表匹配和硬件检查由训练/验证运行入口负责。
- 模型规模字段必须为正整数（拒绝布尔值）；`d_model` 能被 `num_heads` 整除，
  RoPE 的每个 head 维度为偶数，`rope_theta > 0` 且有限。
- 当前 `norm_mode` 取 `pre_norm` 或 `post_norm`，表示归一化位置；
  配置的 `ffn_type` 暂接受 `swiglu` 或 `gelu`，模型已支持 SwiGLU，GELU 尚未实现并会明确报错。
  旧 SiLU 分支目前只通过模型 Python 构造接口提供，配置枚举仍待对齐。
- batch、总步数、验证/保存间隔为正整数；seed 和 warmup 为非负整数，
  `0 <= warmup_steps < max_steps`。
  `0 <= min_learning_rate <= learning_rate` 且 `learning_rate > 0`；
  `weight_decay >= 0`、`grad_clip > 0`，所有数值必须有限。
- `device` 接受 `cpu`、`mps`、`cuda`、`cuda:N`（N 为非负整数），
  配置只检查拼写；训练 API 目前支持 CPU/CUDA，并在实际调用时检查 CUDA 可用性。

## 项目结构

```text
configs/                 smoke、基线模板与消融配置说明
src/tiny_lm/
  __init__.py            版本信息，无导入副作用
  __main__.py / cli.py   模块与控制台入口
  config.py              dataclass / YAML 加载和校验
  tokenizer/             BPE、共享预分词、编解码、独立记录迭代编码与版本化保存（已迁移）
  model/                 基础层、注意力、TransformerBlock 与 TransformerLM 前向（已迁移）
  preprocess.py         默认单文件/显式多文件编码、v1 元信息与不覆盖写入（已迁移）
  data.py               只读文件加载、完整性校验和随机窗口采样（已迁移）
  losses.py             平均交叉熵（已迁移）
  optim.py              AdamW 与全局梯度裁剪（已迁移）
  schedule.py           线性预热与余弦衰减（已迁移）
  checkpoint.py          v2 模型/优化器/配置/步数/RNG 保存与恢复，兼容读取 v1
  binding.py             tokenizer、实际 token 内容与运行参数的历史绑定
  train.py / evaluate.py  token 数组训练与验证（已实现）；CLI 已接入文件加载与续训
  generate.py            单提示词生成、top-k/top-p、模型专用加载（已迁移并接通 CLI）
tests/                   仅覆盖当前已实现能力
examples/toy_corpus.txt   自行编写的小型中英文流程示例
docs/                    架构、路线图、迁移映射、来源
experiments/              实验记录要求
.github/workflows/ci.yml  CPU 安装、Ruff、pytest
```

详细待复核事项见 [迁移文档](docs/migration.md)，阶段验收见
[路线图](docs/roadmap.md)，训练/验证命令说明见 [训练/验证 CLI](docs/training-cli.md)，续写见 [文本生成](docs/generation.md)；
[本地检查记录](docs/local-checks.md) 保留初始框架阶段的历史结果。
示例文本仅供流程测试，不是有效训练语料。
本仓库尚未授予开源许可证；课程来源与代码许可须按
[来源说明](docs/provenance.md) 审查后确定，不能因开源目标推定已有代码可公开发布。
