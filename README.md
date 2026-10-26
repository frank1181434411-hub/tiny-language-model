# tiny-language-model

从 Stanford CS336 Assignment 1 学习成果出发的个人语言模型项目，提供自行实现的
BPE、Tokenizer、Transformer、交叉熵、AdamW 和学习率调度，以及文本预处理、训练、
验证、保存与恢复训练、文本续写的 Python API 和命令行入口。

**当前已完成核心源码迁移和训练闭环，并完成 TinyStories 5 万篇、15000 步 CPU 数据对照。**
生成仍有角色混乱、情节跳跃和重复，尚不能稳定讲述连贯故事。本阶段先整理项目与复现资料，暂停追加训练。

[使用与技术文档](docs/README.md) · [跨设备复现](docs/reproduction.md) · [实验索引](experiments/README.md) · [路线图](docs/roadmap.md) · [来源与许可](docs/provenance.md)

## 当前能力与实验结果

- 文本：BPE 训练、版本化 tokenizer 保存/加载、编码/解码、单文件及显式多文件预处理。
- 模型：完整 TransformerLM、RoPE、RMSNorm、SwiGLU/GELU 前馈、手写 loss 与 AdamW。
- 训练：配置校验、只读 token 数据加载、训练/验证、checkpoint 与随机状态恢复、历史输入绑定。
- 生成：单提示词续写、temperature、top-k/top-p、EOS 与独立采样种子。
- 工程：Python 3.11+、src 布局、可选计算依赖、pytest、Ruff 与 CPU CI 配置。

最近两轮保持同一模型、词表、15000 步预算和评估协议，比较训练数据规模：

| 指标 | 6133 篇训练故事 | 50000 篇训练故事 |
| --- | ---: | ---: |
| 原验证集 loss | 2.3680 | 2.3546 |
| 独立留出集 loss | 2.7094 | 2.4873 |
| 独立留出集 PPL | 15.02 | 12.03 |
| CPU 训练耗时 | 16 分 27 秒 | 24 分 51 秒 |

模型为 926,336 参数、2 层、128 维、上下文 128；训练使用 CPU FP32。
扩充数据改善了这份独立留出集上的预测表现，但不代表生成质量全面改善；
目前只有一个训练种子的对照，评估采用固定随机窗口，未遍历完整验证集。
详细方法、全部生成样本与局限见[最新实验报告](experiments/tinystories-50k.md)，
早期各轮结果见[实验索引](experiments/README.md)。

## 安装（Windows PowerShell）

在项目根目录使用 Python 3.11 或以上的标准 Windows CPython 创建独立环境。
已有项目环境时跳过创建步骤：

```powershell
Set-Location D:\Python\tiny-language-model
python --version
python -m venv .venv
$ProjectPython=".\.venv\Scripts\python.exe"
& $ProjectPython -m pip install -e ".[dev]"
```

基础安装仅需 PyYAML，dev 提供 pytest 和 Ruff；无需激活环境或修改全局 Python 包。
根据使用范围追加依赖：

| 用途 | 项目虚拟环境中的安装命令 |
| --- | --- |
| BPE、Tokenizer、预处理及数据加载 | `& $ProjectPython -m pip install -e ".[dev,tokenizer,data]"` |
| 包含训练、验证和生成的完整功能 | 先按平台安装合适的 PyTorch 构建，再执行 `& $ProjectPython -m pip install -e ".[dev,compute,tokenizer]"` |

compute 提供 PyTorch 与 NumPy，tokenizer 提供 regex，data 单独提供 NumPy。
CPU 安装和历史环境记录见[依赖说明](docs/dependencies.md)；GPU 环境按实际硬件选择 PyTorch 构建。
使用项目不要求安装全部旧课程测试依赖。
Linux 环境的解释器路径为 `.venv/bin/python`；MSYS2 环境可能使用 `.venv/bin/python.exe`，
其 wheel 兼容性与标准 Windows CPython 不同。

## 命令入口

以下命令继续使用上面的 `$ProjectPython`，无需训练数据或 PyTorch：

```powershell
& $ProjectPython -m tiny_lm --help
& $ProjectPython -m tiny_lm --version
& $ProjectPython -m tiny_lm check-config --config configs/smoke.yaml
```

安装后也提供 `tiny-lm`；激活虚拟环境后可直接执行 `tiny-lm --help`。
两种入口行为一致，各子命令使用 `--help` 查看完整参数。

首次使用可按[小型端到端示例](docs/end-to-end.md)依次完成数据准备、四步 CPU 训练、
中断恢复、验证与生成；示例使用独立目录，详细说明每一步的预期产物和检查范围。

| 命令 | 用途与说明 |
| --- | --- |
| `check-config` | 校验配置并显示解析后的路径；不加载模型、不创建输出 |
| `train-tokenizer` | [训练 BPE 并保存 tokenizer](docs/preprocess-cli.md) |
| `prepare-data` | [将文本编码为 token 文件及元信息](docs/preprocess-cli.md) |
| `train` | [训练及通过 --resume 恢复训练](docs/training-cli.md) |
| `evaluate` | [独立评估 checkpoint](docs/training-cli.md) |
| `generate` | [从 checkpoint 续写文本](docs/generation.md) |

### 小型预处理示例

先安装 tokenizer、data 可选依赖，再运行：

```powershell
New-Item -ItemType Directory -Force artifacts/preprocess-cli-demo | Out-Null
& $ProjectPython -m tiny_lm train-tokenizer --input examples/toy_corpus.txt --vocab-size 300 --special-token "<|endoftext|>" --output artifacts/preprocess-cli-demo/tokenizer.json
& $ProjectPython -m tiny_lm prepare-data --input examples/toy_corpus.txt --tokenizer artifacts/preprocess-cli-demo/tokenizer.json --output artifacts/preprocess-cli-demo/train.bin
```

输出为 tokenizer.json、train.bin 与 train.bin.json；已有同名文件会报错，重复演示时换用新目录。
示例仅验证预处理流程，不训练模型；自带文本不是有效训练语料。
此处的 300 词表与 `smoke.yaml` 的 256 词表不匹配，
运行小型训练闭环请使用[端到端示例中的配套准备步骤](docs/end-to-end.md)。

预处理默认严格读取 UTF-8、保留换行、不追加 EOS；支持有序多文件输入。
Python API、记录边界和内存约束见[预处理说明](docs/preprocess.md)。
训练与验证数据分别处理，BPE 只使用训练文本拟合。

## 配置与运行约定

- YAML 必须包含 `model`、`data`、`training`、`output`；拒绝未知字段、重复字段和非法参数。
- 配置内所有相对路径统一相对于**配置文件所在目录**，不展开环境变量或 `~`；
  `--config` 参数本身相对于命令当前目录。
- 模型规模为正整数，`d_model` 能被 `num_heads` 整除，RoPE 的 head dimension 为偶数；
  `rope_theta>0` 且有限。
- batch、总步数、验证/保存间隔为正整数；seed、warmup 非负，`warmup_steps<max_steps`。
  `0<=min_learning_rate<=learning_rate` 且 `learning_rate>0`；
  `weight_decay>=0`、`grad_clip>0`，数值必须有限。
- `norm_mode` 支持 pre_norm/post_norm，`ffn_type` 支持 swiglu/gelu。
  GELU 为两层无 bias 线性映射中的精确激活（非 tanh 近似）；旧 SiLU 分支仅通过模型 Python 构造接口提供。
  用法、精度和 checkpoint 约定见 [Transformer 文档](docs/transformer.md)。
- device 配置接受 cpu、mps、cuda、cuda:N；训练 API 当前支持 CPU/CUDA，
  实际执行时才检查设备可用性。GPU 验收尚未完成。
- `smoke.yaml` 用于准备匹配数据后的极小 CPU 流程；
  `baseline.example.yaml` 是模板，不代表已验证训练方案。
- 新实验使用独立 output_dir。恢复训练须满足配置、数据、tokenizer、dtype 等约束，
  不能直接修改 max_steps 后续训已结束的实验，详见[训练命令](docs/training-cli.md)与[输入绑定](docs/checkpoint-binding.md)。

**Git 拉取不会带回被忽略的语料、tokenizer、token 文件、checkpoint 或运行输出。**
复现实验需要按报告准备数据或另外转移对应产物；二进制 token 文件须带上旁置 JSON。
历史报告中的绝对 Python 路径属于当时设备，执行时替换为自己的项目解释器。
材料清单、传输校验、生成/评估命令与续训限制见[跨设备复现指南](docs/reproduction.md)，
已采集和待补齐的环境信息见[设备清单](docs/environments.md)。

## 开发检查与当前限制

在安装对应功能依赖后，使用项目虚拟环境执行：

```powershell
& $ProjectPython -m ruff check .
& $ProjectPython -m ruff format --check .
& $ProjectPython -m pytest
```

已有[输入绑定阶段的验收记录](docs/checkpoint-binding.md)为 Windows/Python 3.12.10 下
1349 项测试通过及 Ruff 检查通过。这是历史记录，不代表每次文档或实验更新都重新运行了全量检查。
后续数据准备与实验各自记录了实际检查范围；Linux、GPU 和干净环境发布验收仍待完成。

当前生成仅支持单提示词、每步重算上下文；尚无 KV cache、AMP 或梯度累积。
预处理尚不支持任意文本分块的等价流式编码，训练窗口可以跨故事边界，
EOS 不会自动隔离 attention。更多边界见[文档索引](docs/README.md)。

## 项目结构

```text
configs/                 smoke、实验配置、基线模板与消融说明
src/tiny_lm/
  cli.py / config.py     命令行入口与配置校验
  tokenizer/            BPE、编解码与版本化保存
  model/                基础层、注意力与 TransformerLM
  preprocess.py         单文件/多文件预处理
  data.py               token 文件加载与随机窗口采样
  losses.py / optim.py / schedule.py
  checkpoint.py / binding.py
  train.py / evaluate.py / generate.py
scripts/                 数据准备、实验评估与绘图
tests/                   当前实现的测试
examples/                小型演示语料
docs/                    使用指南、接口与迁移记录
experiments/             实验索引、报告、指标与生成样本
.github/workflows/ci.yml CPU CI
```

本仓库尚未授予开源许可证；课程来源、代码归属和许可待按[来源说明](docs/provenance.md)核实。
当前版本号不代表已完成首版发布验收。
