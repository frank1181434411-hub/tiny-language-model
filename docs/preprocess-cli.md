# 预处理命令

`train-tokenizer` 与 `prepare-data` 已接通现有 Python API。
`python -m tiny_lm` 与安装后的 `tiny-lm` 控制台入口行为一致。
训练/验证与生成 CLI 已接通，见 [训练/验证命令](training-cli.md) 和 [文本生成](generation.md)。
预处理命令不改变 BPE、编码或预处理算法。

## Git Bash 示例

在当前机器的项目目录运行，使用已有项目环境：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe
mkdir -p artifacts/preprocess-cli-demo

"$PYTHON" -m tiny_lm train-tokenizer \
  --input examples/toy_corpus.txt \
  --vocab-size 300 \
  --special-token '<|endoftext|>' \
  --output artifacts/preprocess-cli-demo/tokenizer.json

"$PYTHON" -m tiny_lm prepare-data \
  --input examples/toy_corpus.txt \
  --tokenizer artifacts/preprocess-cli-demo/tokenizer.json \
  --output artifacts/preprocess-cli-demo/train.bin
```

其他机器将 PYTHON 改为所用环境的 Python 路径；安装依赖使用
`python -m pip install -e '.[tokenizer,data]'`，无需 PyTorch。
示例只做小文本预处理，不训练语言模型。重复执行须使用新的输出路径。
成功时 stdout 输出 JSON 摘要：tokenizer 命令包含绝对输出路径、实际词表大小、
合并规则数量和特殊 token 列表；预处理命令包含数据/元信息绝对路径与实际词表大小。

## 参数

`train-tokenizer` 必填 `--input`、`--vocab-size`、`--output`。

- 输入为单个 UTF-8 文本文件；只应使用训练文本拟合 BPE。
- `--special-token` 可重复指定，按参数顺序分配特殊 token ID；默认不注册任何特殊 token。
  重复、空字符串和词表容量不足会报错。Shell 中特殊 token 应加引号。
- 实际词表可能因没有剩余可合并字节对而小于请求值，以成功摘要中的 `vocab_size` 为准。
- 输出为原生 v1 JSON。训练前检查目标和父目录，保存时仍使用独占创建防止覆盖。

`prepare-data` 必填 `--input`、`--tokenizer`、`--output`。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--input` | 无，必填 | 可重复指定，按参数顺序处理文件；不自动排序、去重或扫描目录 |
| `--record-mode` | `file` | 每个文件一条记录；`line` 表示每行一条独立记录 |
| `--special-mode` | `recognize` | 识别已注册特殊 token；`ordinary` 将输入按普通文本编码 |
| `--eos-token` | 不追加 | 必须为已注册特殊 token；每条记录末尾无条件追加一次 |
| `--write-batch-size` | `1000000` | 每批写盘 token 数，须为正整数；不限制编码阶段内存 |

例如按顺序合并两个文档，并在各文档末尾追加 EOS：

```bash
"$PYTHON" -m tiny_lm prepare-data \
  --input corpus/a.txt --input corpus/b.txt \
  --tokenizer artifacts/preprocess-cli-demo/tokenizer.json \
  --output artifacts/preprocess-cli-demo/documents.bin \
  --record-mode file --special-mode ordinary \
  --eos-token '<|endoftext|>' --write-batch-size 65536
```

普通文本模式与显式追加 EOS 独立。line 模式只适合本来按行定义记录的数据，
不保证与整文件编码一致；记录边界也不会自动阻止训练采样跨文档。
默认整文件编码仍需容纳单个文件及其编码结果，不是任意块的流式编码。

## 路径、输出与错误

所有命令行相对路径均相对于运行时工作目录；这些命令不读取 YAML 配置。
输出父目录须由调用方提前创建。tokenizer 不覆盖已有文件；数据或旁置 JSON
任一已存在时，预处理拒绝输出。数据格式、哈希和写入中断保护沿用 [预处理接口](preprocess.md)。

成功返回退出码 0；参数、输入、格式或文件操作错误返回 2，错误信息写到 stderr，
常见错误不显示 Python traceback，也不打印成功摘要。缺少可选依赖时提示安装方式。
不自动删除旧输出、不自动切分训练/验证数据、不猜测旧 tokenizer 格式。
帮助、版本、配置检查和模块导入仍不加载 regex、NumPy 或 PyTorch。

训练和验证文本应先独立划分，使用同一训练所得 tokenizer 分别预处理。
训练入口核对模型词表与当前数据/tokenizer 指纹，checkpoint v2 还会绑定历史输入，见 [输入绑定](checkpoint-binding.md)；预处理命令不自动调用模型训练。

## 本地验收

2026-10-22，Windows、Python 3.12.10，使用 `D:/Python/.venvs/tiny-language-model`：

- 全项目 1143 项测试通过，36.73 秒；禁用 pytest 缓存，使用独立临时目录。
- 新增 55 项测试，覆盖模块/控制台入口、实际词表大小、文本往返、中文及含空格路径、
  多文件顺序、两种记录模式、两种特殊 token 模式、EOS 与写盘批大小参数。
- 验证非法参数、无效 UTF-8、缺失文件/父目录、损坏 tokenizer、已有数据或元信息保护，
  以及缺少 regex/NumPy 时的安装提示；当时尚未接通的训练/验证/生成命令明确失败。
  后续这些命令均已接通，当前完整验收见 [文本生成](generation.md)。
- Ruff 静态检查、格式检查、紧凑码风检查和导入审计通过；CLI 不引入计算依赖的顶层加载。
- 未下载语料、安装依赖或进行模型训练；远程 CI、Linux 和大语料性能未在本次验证。
