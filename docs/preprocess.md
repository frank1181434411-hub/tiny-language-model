# 预处理与 token 文件加载

来源：用户已完成的 `cs336_basics/preprocess.py`。保留文本编码和批量写入 token 的思路，
移除重复的旧 tokenizer 加载代码，改用已迁移的 `BPETokenizer`。
修正任意分块编码、固定 uint16 和先删除已有输出的行为。旧源码未修改，测试独立编写。

## Python 接口

```python
process_corpus(input_paths,output_bin,tokenizer,*,record_mode="file",special_mode="recognize",eos_token=None,write_batch_size=1_000_000)
load_token_data(path,*,tokenizer=None,verify_hash=True)
```

预处理函数位于 `tiny_lm.preprocess`，加载函数位于 `tiny_lm.data`。
`input_paths` 默认直接传一个 str 或 Path，也可显式传有序路径列表/元组。
不扫描目录、不展开通配符、不自动排序；重复路径按显式重复输入处理。
相对路径按调用时工作目录解释，配置内路径仍遵循配置文件目录规则。

安装使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,tokenizer,data]"
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py tests/test_tokenizer_serialization.py tests/test_imports.py
```

`tokenizer` extra 提供 regex；`data` extra 提供 NumPy，无需安装 PyTorch。
加载数据只需 NumPy；预处理普通文本另需 regex。模块导入仍不加载这些可选依赖。

## 默认单文件与记录边界

- 默认 `record_mode="file"`：每个文件是一条完整文本，各文件独立编码后按顺序写入。
  不跨文件执行 BPE 合并，也不插入隐式分隔符。空文件也是一条空记录。
- 显式 `record_mode="line"`：每行是一条独立记录，保留 CR、LF、CRLF 和空白，
  不跳过空行；空文件没有记录，末尾换行不会额外生成一条空记录。
  只有数据本来按行定义文档时才应选择此模式，它不保证与整文件编码等价。
- 两种模式均严格读取 UTF-8，保留 BOM 和控制字符，不归一化、不去重、不清洗。
  不接受无效 UTF-8，不把任意大小的读取块误当作独立记录。
- `special_mode` 沿用 Tokenizer 的 recognize/ordinary，默认为识别已注册特殊 token。
- `eos_token=None` 默认不追加 EOS。显式指定已注册的特殊 token 后，每条记录末尾
  无条件追加一次，包括空记录和本来已经以 EOS 结束的记录；不执行自动去重。
  此追加选项与 ordinary 文本模式独立，普通文本模式下也可以显式追加 EOS ID。
- 默认模式的内存开销随最大单文件增长；line 模式随最大单行增长。
  `write_batch_size` 只限制每次转换和写盘的 token 数，不代表编码时只占这些内存。

多个文档写入同一个连续 token 数组后，现有随机窗口采样仍可能跨文档取窗口。
追加 EOS 能标记边界，但不会自动改变 attention mask 或阻止跨文档采样。
需要严格隔离文档窗口时，应另行扩展采样器。

## 数据格式 v1

数据为连续的一维无符号整数数组，旁置元信息文件在原文件名后追加 `.json`：
例如 `train.bin` 与 `train.bin.json`。`process_corpus` 返回数据文件的绝对 Path。

- 实际词表大小不超过 65,536 时使用小端 uint16（`<u2`）；否则使用小端 uint32（`<u4`）。
  最大支持 2³² 个 ID。每批在转换前检查整数类型和词表范围，拒绝 bool、浮点和越界 ID。
- 元信息包含格式/版本、dtype、token 数、字节数、数据 SHA-256、实际词表大小、
  tokenizer SHA-256、记录/EOS/特殊 token 选项、总记录数及按处理顺序排列的源文件信息。
- 源文件信息记录绝对路径、实际读取字节的 SHA-256、字节数、记录数和 token 数。
  加载不依赖原始文件仍存在；数据文件和旁置 JSON 可以成对移动。
- tokenizer 指纹来自规范化的原生词表、merges、预分词规则与特殊 token 元信息，
  不受 JSON 缩进、换行或词表字典插入顺序影响。同样大小的词表不代表同一个 tokenizer。
- 数据哈希在写入过程中计算，不必再读一遍输出；源文件哈希对应保留换行后的原始 UTF-8 字节。
  处理期间不要修改输入文件或 tokenizer 实例。

## 输出保护与加载检查

父目录须由调用方创建；数据或元信息任一目标已存在时直接拒绝，不提供覆盖开关。
先在同目录写临时数据和元信息并 flush/fsync，再以不覆盖目标的硬链接逐个发布，
元信息最后发布作为完成标记。文件系统须支持硬链接；不支持时明确失败，不回退到覆盖写入。
本机 Windows 临时目录已验证此流程；其他文件系统仍需实际验证。

这不是跨文件原子事务。普通异常会清理本次临时文件，并回滚本次已发布的输出；
强制终止或断电可能留下临时文件或只有 `.bin` 的不完整结果，加载器会拒绝缺少元信息的文件。
不自动删除残留文件，应先确认再人工清理。不要由外部程序并发替换正在发布或使用的数据文件。

`load_token_data` 执行以下检查：

1. 元信息字段、版本、dtype、计数、EOS 结构和源文件汇总一致性。
2. 实际文件大小与 token 数、dtype 对应。
3. 如果传入 tokenizer，核对内容指纹以及 EOS 文本与 ID；训练使用时建议始终传入。
4. 默认完整扫描并核对数据 SHA-256，然后分块扫描全部 ID 是否小于实际词表大小。

`verify_hash=False` 仅跳过内容哈希，仍检查文件大小和全部 ID 范围，
因此仍有完整读取开销，也无法检测所有同尺寸且 ID 合法的内容损坏。
哈希只能校验与元信息的一致性，不是防篡改签名。

非空数据返回只读 `np.memmap`，空数据返回只读空数组，因为空文件不能建立 memmap。
现有采样/训练入口会拒绝不够 `context_length+1` 个 token 的数组。
使用期间保持文件不变；Windows 上移动或删除文件前须释放仍在使用的 memmap 及其视图。

## 与训练的关系

训练、验证文件分别调用预处理，不自动进行数据集切分；BPE 只应使用训练文本拟合。
返回的数组可以传给现有 `get_batch`、train 和 evaluate Python API。
模型 `vocab_size` 应使用实际 `len(tokenizer.vocab)`。
checkpoint 尚未自动记录或核对本格式的数据指纹；续训时仍须由调用方确认数据一致。
新增元信息记录不等于已经完成训练配置/CLI 的集成。

预处理 API 迁移时未接通 CLI；现已接通 `prepare-data` 和 `train-tokenizer`，见 [预处理命令](preprocess-cli.md)。
没有进行正式训练、下载语料或评估大语料性能。
任意文本块的等价流式编码、跨文档采样隔离和旧裸 bin 的自动猜测加载不在本批范围。

## 本地验证

2026-10-22，Windows、项目 `.venv`、Python 3.12.10：

- 新增 80 项测试；预处理、tokenizer 序列化和导入审计定向检查 127 项通过。
- 全量 pytest：1096 passed，59.40 秒，未跳过测试；禁用缓存执行。
- Ruff 静态检查、格式检查、依赖检查通过。
- 实测单/多文件、记录与 EOS 边界、原始换行、真实 uint32 大 ID、语义指纹、
  写入异常回滚、竞争文件保护、损坏元信息/数据拒绝加载和只读数据接入采样器。
- CI 加入轻量的 data extra；远程 CI、Linux 和其他文件系统未在本次执行。
- 项目虚拟环境的离线 editable 安装已刷新，data extra 元信息和 `pip check` 通过。
  首次替换旧安装文件遇到 WinError 5，获准在项目虚拟环境范围内重试后成功；未下载依赖。
- README 的单文件示例已在隔离临时目录实际执行：词表大小 300、273 个 uint16 token，
  加载后严格解码与原始示例文本完全一致。示例文件没有写入项目运行目录。
