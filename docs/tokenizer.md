# Tokenizer 迁移

来源：用户已完成的 `cs336_basics/tokenizer.py` 中的 `BPETokenizer`。
保留类名、主要变量及手写 BPE 编码流程：在每个预分词内选择最小 rank 的相邻字节对，
从左到右合并该规则的非重叠匹配，再重复查找。旧源码未修改；测试独立编写。

## 使用方式

安装与检查使用项目虚拟环境，不需要 PyTorch 或 GPU：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,tokenizer]"
.\.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py tests/test_bpe.py tests/test_tokenizer_serialization.py tests/test_imports.py
```

下面的 Python 示例从项目根目录运行，仅使用小型示例文本：

```python
from pathlib import Path
from tiny_lm.tokenizer.bpe import train_bpe
from tiny_lm.tokenizer.tokenizer import BPETokenizer

special_tokens=["<|endoftext|>"]
vocab,merges=train_bpe(Path("examples/toy_corpus.txt"),300,special_tokens)
tokenizer=BPETokenizer(vocab,merges,special_tokens)
text="你好，world!🙂<|endoftext|>"
ids=tokenizer.encode(text)
assert tokenizer.decode(ids,errors="strict")==text

ordinary_ids=tokenizer.encode(text,special_mode="ordinary")
record_ids=list(tokenizer.encode_iterable(["第一条文本","第二条文本"]))
output_dir=Path("artifacts")
output_dir.mkdir(exist_ok=True)
tokenizer.save(output_dir/"tokenizer.json")
restored=BPETokenizer.from_file(output_dir/"tokenizer.json")
assert restored.encode(text)==ids
```

`save` 和 `from_file` 复用已有原生 v1 JSON 校验。相对路径按调用时工作目录解释；
保存不自动创建父目录，不覆盖已有文件。旧版双文件格式没有自动导入支持。
类构造、解码及保存/加载只依赖标准库；处理普通文本时才加载可选依赖 regex。
模块导入不读取数据、不创建目录，也不加载 regex、NumPy 或 torch。

## 明确的行为约定

- 构造时校验完整基础字节、连续 ID、有序合并依赖和特殊 token 元信息，
  然后复制输入容器。调用方后续修改原词表、merges 或特殊 token 列表，不影响实例。
  不支持直接修改实例内的映射；需要新规则时重新构造。
- 普通预分词和特殊 token 切分复用 `rules.py`，与 BPE 训练一致。
  保留输入的空白、CR/LF 和 Unicode 字符，不做 Unicode 归一化。
  `None`、bytes、数字等不是合法文本；含孤立 surrogate 的字符串不能编码为 UTF-8。
- `encode` 默认 `special_mode="recognize"`，按最左位置、最长匹配识别已注册特殊 token。
  `special_mode="ordinary"` 将整个输入当作普通文本，不产生特殊 token ID。
  未注册的标记一直作为普通文本处理；两种模式均不会自动添加 BOS/EOS。
- 普通字节和特殊 token 使用独立映射。特殊 token `a` 可以拥有独立 ID，
  不覆盖基础字节 `b'a'` 的 ID 97。若多个普通合并 token 字节相同，编码统一选最小 ID，
  不依赖词表字典的插入顺序；解码仍接受所有合法 ID。
- `decode` 接受整数 ID 的可迭代对象，包含 Python 整数和 NumPy 整数；
  拒绝 bool、浮点数、字符串和未知 ID。Tensor 请先转换为整数列表。
  先拼接全部 token 字节再进行 UTF-8 解码，因此一个字符可以横跨多个 token。
- `decode` 默认 `errors="replace"`，保留原实现对无效或截断 UTF-8 的替换行为；
  `errors="strict"` 改为抛出 `UnicodeDecodeError`。不提供静默丢弃字节的模式。
- 对合法 Unicode 文本保证文本往返；不能据此假定任意 ID 序列的
  `encode(decode(ids))` 会还原原 ID，重复字节、特殊 token 和无效 UTF-8 都可能影响结果。

## 迭代编码的边界

`encode_iterable` 逐条读取独立文本记录，保持原实现逐条调用 encode 的语义。
它不会跨记录合并，不补分隔符或 EOS，只会把各条编码结果依次产出。
外层不一次读取全部记录，但每条记录仍先编码为列表；发生错误前可能已产出前面的 ID。
传入单个 str/bytes 会明确报错，避免意外按字符或字节逐项迭代。

因此存在 `ab` 合并规则时，`encode("ab")` 与 `encode_iterable(["a","b"])` 不同；
把 `<S>` 拆成两条记录也不会识别为一个特殊 token。
任意大小的文件读取块、文件行和逻辑文档不能自动视为同一种边界。
后续数据预处理必须先确定记录语义，或另外实现保持边界状态的流式编码器。

## 验证与范围

测试包括：精确 UTF-8 字节往返、中文/emoji/组合字符、空白与控制字符、合并优先级、
预分词边界、特殊 token 最长匹配与相邻匹配、两种特殊 token 模式、重复字节 ID、
输入复制、非法配置/输入/ID、严格 UTF-8 解码、惰性记录迭代及保存加载。
25 个固定随机种子使用独立的按规则顺序替换整数序列的参考编码器，
对照训练文本及未见文本的完整 ID 序列；另检查 BPE→保存→加载→编码→解码的小语料流程。

本批迁移提供 Python API；后续已提供 [数据预处理与文件加载](preprocess.md) 和 [预处理命令](preprocess-cli.md)。
模型训练和生成 CLI 已接通，见 [训练/验证命令](training-cli.md) 和 [文本生成](generation.md)；任意文本分块编码和旧格式导入仍待迁移，
没有进行正式训练或大语料性能评估。

本地检查记录（2026-10-22，Windows，项目 `.venv`，Python 3.12.10）：

- 本批新增 106 项测试；Tokenizer/BPE/保存格式/导入审计定向检查 227 项通过。
- 全量 pytest：1016 passed，64.20 秒，未跳过测试；执行时关闭 pytest 缓存。
- Ruff 静态检查、格式检查、`pip check` 与 `git diff --check` 通过。
- 未安装新依赖；现有 `tokenizer` extra 和 CPU CI 可运行新增测试。
  本次未在远程 CI、Linux 或大语料上验证性能。
