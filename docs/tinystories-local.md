# 本地 TinyStories 小样本准备

2026-10-23 已完成本地 5M 样本整理、2048 词表 BPE 训练和 token 文件生成。
本文记录数据准备阶段；随后已完成 [50 步 CPU 模型试跑](../experiments/tinystories-pilot.md)，
包括 loss、生成样本和中断续训核对；[1000 步初始基线](../experiments/tinystories-baseline.md)也已完成。

## 来源与划分

输入为 `D:/Python/CS336-Private/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt`，
共 5,279,869 字节，原文件保持不变；没有下载数据，也未使用另一个 5 篇故事的小测试文件。
源文件 SHA-256：

```text
09babf3bf1928e32185b133d8b85be949935b576a7c1b861f77107f3b3d82390
```

该文件首尾是截断故事，因此仅保留两个外侧片段之间的完整记录。
以 `<|endoftext|>` 分隔；CRLF/CR 统一为 LF，去掉故事外围空白，内部段落保留。
按规范化后的全文做精确去重，本次重复数为 0，最终保留 6456 篇故事。

seed=42；按 `SHA256(str(seed) + NUL + 故事正文)` 排序，前 `ceil(N/20)` 篇作为验证集，
其余作为训练集。相同源文件、种子和脚本产生相同划分；manifest 记录各故事的原始片段编号。
这是本地样本自行划分的验证集，不是官方 TinyStories validation，不应直接与官方基准分数比较。
只检查了完全相同的故事，未开展近似重复检测或核实该 fixture 对应的上游数据版本。

## 本次产物

所有产物位于项目根目录下的 `data/tinystories-local-v1/`，由现有 Git 忽略规则排除。

| 数据 | 故事数 | UTF-8 文本字节 | token 数 | EOS 数 |
| --- | ---: | ---: | ---: | ---: |
| train | 6133 | 4,975,369 | 1,341,153 | 6133 |
| val | 323 | 257,640 | 69,576 | 323 |

- `train.txt`、`val.txt`：每篇正文后接一个 `<|endoftext|>` 和换行，包含故事内部段落。
- `tokenizer.json`：仅使用 train.txt 训练；256 个字节 token、1 个特殊 token、1791 次合并，共 2048 词表。
- `train.bin`、`val.bin` 及各自 `.bin.json`：小端 uint16 数组及项目原生元信息。
- `manifest.json`：源哈希、规范化与划分规则、原始片段编号、文本哈希、tokenizer 指纹、
  token 内容指纹、token/EOS 数量和本机准备耗时。完成所有校验后才写入。

编码使用 file 模式并识别已有特殊 token，不额外追加 EOS，也不把每行误当成一篇故事。
旁置元信息的 num_records=1 表示一个输入文件；实际故事数量在 manifest 中。
当前训练采样仍可能跨故事边界取窗口；EOS 不等于 attention mask 隔离。

tokenizer 内容指纹：

```text
b3cb3f05297337ff79b14e7409a0f8862c839132c2d044c80cbe9586e0d1dbc0
```

## 复现命令

在 Git Bash 中执行；输出目录必须不存在，避免覆盖已有实验输入：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe
"$PYTHON" scripts/prepare_tinystories_sample.py \
  --source ../CS336-Private/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt \
  --output-dir data/tinystories-local-v1 \
  --seed 42
```

本机该目录已经准备完毕，无需再次执行；重复实验需改用新目录。
脚本只适用于上述哈希的已核对样本，哈希不匹配会在创建输出前报错，
不能用它盲目裁掉其他数据集的首尾故事。失败后保留本次部分输出供排查，不自动覆盖重跑。
复现需要同一源文件和项目环境，原始数据与生成文件不会随 Git 推送。

## 验证与后续

Windows、Python 3.12.10 本机准备耗时约 9.26 秒，其中 BPE 约 2.28 秒；不是模型训练耗时。
两份 bin 均经项目文件加载器验证哈希、词表绑定和 ID 范围；完整解码与整理后的文本逐字一致，
EOS 数量等于故事数，训练/验证无完全相同故事，源文件哈希未改变。
Ruff 静态检查、格式检查通过；本次未改变核心分词、预处理或训练算法。

现已提供 configs/tinystories-pilot.yaml 并完成 50 步试跑，接下来可从头运行
configs/tinystories-cpu.yaml 的 1000 步配置；该轮随后已完成，详见 [基线记录](../experiments/tinystories-baseline.md)，重跑应使用新输出目录。
现有 256 词表 smoke.yaml 不能直接用于这份数据。

后续扩充数据采用独立目录并保留本页全部原始输入，方案与核验结果见 [5 万篇扩充记录](tinystories-expanded.md)。
