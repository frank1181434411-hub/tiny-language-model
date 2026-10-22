# BPE 迁移

来源：用户已完成的 `cs336_basics/train_bpe.py`。保留 `train_bpe` 名称、
主要变量、增量 pair 计数与倒排索引、从左到右的合并流程。旧源码未修改。
保存格式为新项目实现；测试独立编写，没有复制课程测试或 fixtures。

## 当前接口

```python
from pathlib import Path
from tiny_lm.tokenizer.bpe import train_bpe
from tiny_lm.tokenizer.serialization import load_tokenizer,save_tokenizer

special_tokens=["<|endoftext|>"]
vocab,merges=train_bpe(Path("examples/toy_corpus.txt"),300,special_tokens)
output_dir=Path("artifacts")
output_dir.mkdir(exist_ok=True)
save_tokenizer(output_dir/"toy-tokenizer.json",vocab,merges,special_tokens)
loaded_vocab,loaded_merges,loaded_specials=load_tokenizer(output_dir/"toy-tokenizer.json")
assert (loaded_vocab,loaded_merges,loaded_specials)==(vocab,merges,special_tokens)
```

从项目目录运行上述示例。Python API 的相对路径遵循调用时的工作目录，
YAML 内路径仍由配置系统相对于配置文件目录解析，两者不要混淆。
保存文件的父目录须已存在；同名文件会报 `FileExistsError`，不会覆盖。
保存先完整校验并生成 JSON，再独占创建文件；目前不保证写入中断时文件完整，
加载会拒绝不完整 JSON。示例由调用方显式创建 artifacts 目录，生成文件被 Git 忽略。

PowerShell 安装与定向检查：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,tokenizer]"
.\.venv\Scripts\python.exe -m pytest tests/test_bpe.py tests/test_tokenizer_serialization.py tests/test_imports.py
```

`tokenizer` 可选依赖组只增加 `regex`。模块导入不加载 regex、不读文件；
训练调用时才导入 regex，缺失时给出英文安装提示。BPE 不需要 PyTorch、NumPy 或 GPU。

## 训练约定与改动

- 输入为严格 UTF-8，保留原始 CR/LF，不采用文本模式的换行归一化；
  这是相对旧代码的有意改动，训练与未来直接编码原文应看到相同字节。
- 共享 `rules.py` 中的原预分词正则，不跨预分词或特殊 token 边界合并。
  特殊 token 按最左位置、最长匹配切分，按字面值处理正则元字符。
  例如同时指定 `<X>` 和 `<X>tail` 时，完整 `<X>tail` 不再泄漏到普通文本统计。
- `vocab_size` 必须为 Python 整数，拒绝 bool，且至少为 `256+len(special_tokens)`。
  特殊 token 必须是由非空、互异、可编码为 UTF-8 的字符串组成的 list。
  参数校验先于文件读取，所有新增可读错误信息为英文。
- 基础字节 ID 为 0–255，合并结果依次追加，特殊 token 最后按调用方列表顺序追加。
  匹配时的长度排序不会重排特殊 token 的 ID。
- 每步优先选择频率最高的字节对，同频时选择字节对字典序最大者。
  空文本、仅特殊 token 文本或没有剩余字节对时提前停止，不补虚构 token。
  模型和数据管线应使用实际 `len(vocab)`，不能直接假定等于请求的 `vocab_size`。
- 本批仍整文件读取、在内存统计，倒排索引允许保留已失效的候选位置。
  尚未进行大语料性能评估，也没有迁入分块、并行训练或断点续训。

## 原生文件 v1

单个 JSON 包含 `format=tiny-lm-bpe`、整数 `version=1`、
`pretokenizer=gpt2-regex-v1`、`vocab`、有序 `merges` 和 `special_tokens`。
词表数组的下标就是 ID；词表与合并字节均以规范 Base64 保存，
因此不要求每个 token 自身是有效 UTF-8。特殊 token 条目保存原文与独立 ID。

加载和保存检查连续 ID、完整的基础 256 字节、合并依赖顺序、重复合并规则、
合并结果与词表对应关系，以及特殊 token 的顺序、内容与 ID。
加载拒绝未知字段、重复 JSON 字段、未知版本/预分词规则和非法 Base64。
不自动猜测旧版 `vocab.json`/`merges.txt` 的编码，也不反序列化 pickle。

特殊 token 可以与普通 token 字节相同，例如特殊 token `a` 与基础字节 `b'a'`。
它们使用不同 ID，Tokenizer 已独立维护特殊 token 映射，不会把整个词表
简单反转成唯一的 bytes→ID 字典。格式也保留不同合并规则产生相同字节的独立 ID；
普通文本编码统一选取相同字节的最小普通 ID，详见 [Tokenizer 迁移记录](tokenizer.md)。

## 验证范围与后续

测试使用独立的全量重计数参考实现逐项比较完整词表与 merges，
覆盖固定用例、40 个固定随机种子、长度 1–8 的全部二元字母串、重复与重叠字节对、
同频排序、中英文、emoji、空白和特殊 token；另检查不同 Python hash seed 的一致性。
保存/加载测试覆盖全部 256 字节、实际训练结果往返、损坏元信息与拒绝覆盖。
导入审计继续验证无读写数据、目录创建、随机状态修改等副作用。

Tokenizer 的 encode/decode/encode_iterable 已迁移并复用共享预分词和原生文件格式。
迭代编码按独立文本记录定义，不把任意文本分块当作等价编码。
后续已提供 [数据预处理与文件加载](preprocess.md)；`train-tokenizer` CLI 已接通，见 [预处理命令](preprocess-cli.md)；旧格式导入仍待迁移。

本地检查记录（2026-10-22，Windows，项目 `.venv`，Python 3.12.10）：

- 全量 pytest：910 passed，60.98 秒；其中本批新增 119 项。
- Ruff 静态检查与格式检查通过；本批文件保留用户要求的紧凑码风。
- 原有导入审计、CLI 及数值组件测试一并通过。关闭缓存执行，未跳过测试。
- 离线构建并重新安装项目成功；首次替换 editable 文件遇到 WinError 5，
  随后在项目虚拟环境范围内获准重试并成功，未修改全局环境或下载依赖。
- CI 已加入 `tokenizer` extra；远程 CI、Linux 和大语料性能尚未在本次验证。
