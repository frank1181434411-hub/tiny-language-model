# tiny-language-model

一个从 Stanford CS336 Assignment 1 学习成果出发的个人语言模型项目。
已有自行实现的 BPE、Tokenizer、Transformer、损失函数、AdamW、调度、采样、
checkpoint 和生成源码，**目前已迁入交叉熵损失、AdamW 和梯度裁剪，其他实现待迁移**。长期目标是提供可安装、可复现的
文本预处理、训练、验证、保存与恢复训练、文本续写流程。

当前版本 `0.1.0` 已完成工程框架：`src` 布局、严格的 YAML/dataclass 配置、
CLI、测试、Ruff、CPU CI 与迁移文档，并提供手写交叉熵 `tiny_lm.losses.cross_entropy`、
`tiny_lm.optim.AdamW` 和 `tiny_lm.optim.clip_gradient_norm`。
模型与数据处理模块仍只有职责说明和 TODO。
**当前不能训练、验证模型或生成文本；没有正式训练结果或已验证基线。**
`smoke.yaml` 只检查配置与框架，`baseline.example.yaml` 仅为待硬件和实验确定的模板。

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
`compute` 可选依赖组声明 loss 和 optimizer 所需的 PyTorch；
CPU 安装方法与数值验证说明见 [loss 迁移记录](docs/losses.md)
和 [optimizer 迁移记录](docs/optimizer.md)。
显式导入 `tiny_lm.optim` 会加载 torch；基础包、配置与 CLI 保持轻量导入。

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

`train-tokenizer`、`prepare-data`、`train`、`evaluate`、`generate` 提供帮助和基本参数；
传入所需参数后会提示“尚未实现，等待源码迁移”并返回非零退出码 `2`。
它们不会读取输入文件、加载模型、创建输出或返回模拟结果。
例如 `& $ProjectPython -m tiny_lm train --help` 可查看未来入口参数。

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
  实际文件存在性、词表匹配和硬件检查留给未来运行入口。
- 模型规模字段必须为正整数（拒绝布尔值）；`d_model` 能被 `num_heads` 整除，
  RoPE 的每个 head 维度为偶数，`rope_theta > 0` 且有限。
- 当前 `norm_mode` 取 `pre_norm` 或 `post_norm`，表示归一化位置；
  `ffn_type` 取 `swiglu` 或 `gelu`。这些是框架配置选项，
  不表示模型已经支持，迁移时还须与旧代码对齐。
- batch、总步数、验证/保存间隔为正整数；seed 和 warmup 为非负整数，
  `0 <= warmup_steps < max_steps`。
  `0 <= min_learning_rate <= learning_rate` 且 `learning_rate > 0`；
  `weight_decay >= 0`、`grad_clip > 0`，所有数值必须有限。
- `device` 接受 `cpu`、`mps`、`cuda`、`cuda:N`（N 为非负整数），
  仅检查拼写，不声称相应硬件可用。

## 项目结构

```text
configs/                 smoke、基线模板与消融配置说明
src/tiny_lm/
  __init__.py            版本信息，无导入副作用
  __main__.py / cli.py   模块与控制台入口
  config.py              dataclass / YAML 加载和校验
  tokenizer/             BPE、编解码、序列化（待迁移）
  model/                 基础层、注意力、Transformer（待迁移）
  data.py               采样（待迁移）
  losses.py             平均交叉熵（已迁移）
  optim.py              AdamW 与全局梯度裁剪（已迁移）
  schedule.py           调度（待迁移）
  checkpoint.py          保存与恢复（待迁移）
  train.py / evaluate.py / generate.py 运行流程（待迁移）
tests/                   仅覆盖当前已实现能力
examples/toy_corpus.txt   自行编写的小型中英文流程示例
docs/                    架构、路线图、迁移映射、来源
experiments/              实验记录要求
.github/workflows/ci.yml  CPU 安装、Ruff、pytest
```

## 后续迁移顺序

1. 先确认课程版本、代码归属、依赖和公开发布许可。
2. 迁移 BPE、Tokenizer 与统一序列化，验证编解码和特殊 token。
3. 迁移数据格式与采样、模型基础层、注意力、Transformer；loss 已先行迁移。
4. AdamW 和梯度裁剪已迁移；后续迁移调度和 checkpoint，接入训练状态恢复。
5. 接入训练、验证、生成 CLI，完成小数据训练闭环。
6. 明确硬件与实验配置，建立可复现基线，再验收首版发布。

详细待复核事项见 [迁移文档](docs/migration.md)，阶段验收见
[路线图](docs/roadmap.md)，本次实际执行结果见 [本地检查记录](docs/local-checks.md)。
示例文本仅供未来流程测试，不是有效训练语料。
本仓库尚未授予开源许可证；课程来源与代码许可须按
[来源说明](docs/provenance.md) 审查后确定，不能因开源目标推定已有代码可公开发布。
