# 数据采样迁移记录

来源：`D:/Python/CS336-Private/assignment1-basics/cs336_basics/data.py`。
迁入 `tiny_lm.data.get_batch`，保留原四个位置参数、主要变量和采样方式，
增加仅限关键字传入的可选 `generator`。未迁移文件加载、文本预处理或训练入口。

## 输入与采样约定

- `dataset` 为一维整数 NumPy ndarray 或 memmap；不接收列表、张量、掩码数组，
  也不接收浮点、复数、bool、object 或字符串 dtype。
- `batch_size`、`context_length` 为正 Python 整数，不接受 bool；
  至少有 `context_length+1` 个 token。不足时英文报错，不自动填充。
- 保留均匀、有放回的窗口采样，允许重复起点。
  数据长度为 n、窗口长度为 L 时，合法起点为 `[0,n-L-1]`，包含最后一个合法位置。
- x 为起点后的 L 个 token，y 向后错一位；两者形状都是 `(batch_size,context_length)`。
  不处理文档边界，不自动插入 EOS，也不划分训练/验证数据。
- 返回指定 `device` 上的 `torch.long` 张量；先在 CPU 转成原生字节序的 int64，
  再移动设备，避免设备上的 unsigned dtype 支持影响输入转换。
  支持只读 memmap、非连续数组和非原生字节序数组；输出不与原数据共享存储。

## token 检查范围

仅扫描本次抽中的 x/y，拒绝负值和超过 `2**63-1` 的 token，转换前完成检查。
uint64 的极大值不会通过强制转换绕回负数；浮点 token 不会被截断。
没有 `vocab_size` 参数，不验证 token 是否小于模型或 tokenizer 词表大小。
未被抽中的非法值不会导致本次采样失败；现有文件加载和训练/验证入口负责整体数据与词表一致性检查。
这是整数的范围与精确转换检查，不涉及浮点精度提升。

## 随机状态

索引始终在 CPU 生成；`generator` 仅接受 CPU `torch.Generator` 或 None。
传入 generator 时消耗其状态，不改变全局 CPU 随机状态；不传时延续旧版全局状态行为。
函数不调用 `manual_seed`，调用者负责初始化 seed。
未来训练与验证应使用各自的 generator，并将各自 `get_state()` 的结果纳入 checkpoint。
恢复 generator 状态可以续接抽样；当前未接入 checkpoint，也未验证完整训练恢复。
如果抽样后的 token 检查或设备转换失败，已经消耗的随机状态不会回滚。

## 依赖与格式

NumPy 已加入 `compute` 可选依赖，与 PyTorch 一起按需安装。
`tiny_lm.data` 导入时不加载二者，现有轻量导入审计继续适用。
源码与新测试采用紧凑空格、单行括号和起始三引号独占一行的文档字符串；
Ruff 格式化排除这两个文件，静态检查保留并对长行放宽。
迁移源码中的连续 import 之间不留空行，包括 future import 与其他导入；
data 与 losses、optim 一样对 Ruff 的导入排序规则作文件级例外。

## 验证范围

新测试覆盖首尾起点、重复采样、目标错位、最短输入、八种整数 dtype、
int64 上界、uint64 越界、只读 memmap、非连续与非原生字节序数组，
以及输入错误、只检查抽中 token、全局/独立随机状态和恢复后采样一致性。
使用 meta 设备检查输出设备转移路径；这不等同于 GPU 执行验证。
当前环境是 CPU 构建，未验证 CUDA generator 的运行时拒绝分支或实际 GPU 数据传输。
后续新增的 v1 数据格式、元信息、文档边界、输出保护和 `load_token_data` 见
[预处理迁移记录](preprocess.md)。任意文本分块的等价编码仍未实现。

2026-10-21 本地验收使用项目 `.venv/Scripts/python.exe`（Python 3.12.10）：

- 新增采样测试 **60 passed**；全项目 **293 passed**，耗时 25.01 秒。
- Ruff 静态检查、格式检查、单行括号与文档字符串样式检查通过。
- 通过离线可编辑安装刷新了项目元数据，`pip check` 通过。
  首次刷新因沙箱无法替换既有 `.pth` 文件失败，获准在沙箱外仅刷新项目虚拟环境后成功；
  没有安装新计算依赖或修改全局环境。
- 测试使用 D 盘独立临时目录并禁用 pytest 缓存，未更改系统 TEMP/TMP 设置。
- 未运行远程 CI、GPU 测试或真实训练。

从项目根目录执行的主要检查命令（PowerShell）：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --no-cache --check .
.\.venv\Scripts\python.exe -m pip check
```
