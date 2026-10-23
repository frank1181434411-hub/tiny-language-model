# Checkpoint v2 输入绑定

新保存的 checkpoint 使用 format_version=2，增加 run_state。
训练会自动绑定实际训练/验证 token 内容、模型参数精度和 eval_batches；CLI 还传入 tokenizer，
记录其内容指纹。原模型、优化器、学习率、采样算法和 RNG 恢复语义保持不变。

## 保存内容

| run_state 字段 | 内容 |
| --- | --- |
| data_fingerprint_format | 固定为 token-ids-u64le-v1 |
| train_data | sha256 和 num_tokens |
| validation_data | sha256 和 num_tokens |
| tokenizer_sha256 | 原生 tokenizer 的规范化 SHA-256；未提供 tokenizer 的数组 API 为 null |
| eval_batches | 训练期间每次验证的 batch 数 |
| dtype | float16、bfloat16、float32 或 float64；须与模型参数 dtype 一致 |

其余参数（batch_size、seed、验证间隔、学习率计划等）继续保存在 ProjectConfig 或 optimizer 状态中。
不保存语料本身。无运行输入的通用 save_checkpoint 默认保存 run_state=null，不能据此声称有完整绑定。

数据指纹来自实际加载的 token 数组，不从旁置 JSON 复制声明值：
先写入 ASCII 前缀 `token-ids-u64le-v1` 和一个零字节，再按顺序将 token ID 转为小端 uint64 字节并计算 SHA-256。
每次最多处理 1,000,000 个 token，不复制整份语料；空数组也有确定指纹，但训练仍拒绝过短数据。
相同 token 序列的 uint16/uint32/int64、大端/小端及非连续数组具有相同指纹。
这是逻辑 token 内容哈希，与预处理元信息中原始 bin 文件字节的 SHA-256 不同。
tokenizer 指纹复用现有规范化词表、merges、特殊 token 和预分词规则，不受 JSON 缩进或字段顺序影响。

## 各入口的检查

- 续训：比较训练数据、验证数据、tokenizer（已记录时）、eval_batches 和 dtype，
  同时保留完整配置、模型和优化器检查。相同路径下同时替换 bin 与 JSON 也不能绕过历史内容核对。
- 独立验证：核对 tokenizer；允许使用新的验证集、独立采样种子和不同 eval_batches。
  独立评估不改变原训练输入，不将它与精确续训混为一谈。
- 文本生成：核对 tokenizer 指纹，拒绝同样词表大小但内容不同的 tokenizer。
  不读取训练/验证数据，不恢复 optimizer 或训练 RNG。
- 单纯读取 checkpoint 或不传输入绑定的低层权重加载：保留兼容接口，仅校验格式和所请求的状态，
  不声称核对了外部 tokenizer 或语料。

已记录的内容或参数不匹配始终以英文报错，且发生在应用保存权重、optimizer 和 RNG 前。
train 本身仍会初始化新模型和随机种子，再调用恢复接口；失败不产生新 checkpoint。
模型参数 dtype 与 run_state.dtype 也会核对，防止元信息与权重互相矛盾。

## v1 与未绑定文件

| 输入 | 默认行为 | 显式允许后 |
| --- | --- | --- |
| 正常绑定的 v2 | 核对通过后继续 | 已记录的不匹配仍拒绝 |
| v1，缺少 run_state | 可读取；续训及文本命令拒绝缺失绑定 | 允许历史未知信息，不能追溯核实旧输入 |
| 通用 API 保存的 v2，run_state=null | 与缺失绑定的 v1 相同 | 同上 |
| 数组训练 v2，已绑定数据但 tokenizer=null | 相同数组可续训；CLI 要求核对 tokenizer 时拒绝 | 只放行缺失 tokenizer，仍核对已记录数据与参数 |
| 无版本旧作业文件或未知版本 | 拒绝 | 不提供绕过 |

使用旧文件且已经自行确认原 tokenizer 和数据时，可在对应命令后追加：

```bash
--allow-unbound-checkpoint
```

例如 Git Bash 中：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe
"$PYTHON" -m tiny_lm train --config configs/smoke.yaml --resume outputs/smoke/old-v1.pt --allow-unbound-checkpoint
```

CLI 会在 stderr 明确提示历史缺失信息不能追溯验证。该开关只允许缺失信息，不能忽略不匹配、损坏格式、
不兼容模型或不同完整训练配置。旧文件不会原地升级；续训完成实际更新后保存的新文件为 v2，
绑定本次传入的数据和 tokenizer。它只能证明本次开始采用的输入，不能证明早期训练用了这些输入。
若 checkpoint 已达到终点，则不产生新文件；单独验证和生成也不自动重写旧文件。

## Python 接口

- `train(...,tokenizer=None,allow_unbound_checkpoint=False)` 自动计算 run_state，保存和续训均使用它。
  可传 BPETokenizer 绑定词表；无 tokenizer 时仍绑定两份数组和运行参数。
- `save_checkpoint(...,run_state=None)` 写 v2；传入的字典须满足上表结构，保存时复制。
- `load_checkpoint(...,run_state=None,tokenizer_sha256=None,allow_unbound_checkpoint=False)`
  可核对完整运行信息或仅 tokenizer；未请求绑定检查的低层调用仍可读取 v1。
- `load_generation_model(...,tokenizer=None,allow_unbound_checkpoint=False)`
  传入 BPETokenizer 后核对其指纹；CLI 始终提供 tokenizer。
- `tiny_lm.binding` 提供 build_run_state、token_data_fingerprint、get_tokenizer_fingerprint，
  模块导入不加载 NumPy、PyTorch 或 regex，也不访问数据。

## 范围与限制

数据加载本身仍核对原始 bin 文件哈希；绑定另有一次 O(n) 的逻辑序列哈希扫描，之后每次保存复用结果。
运行期间不得修改数组、映射文件或 tokenizer；不提供并发写入防护或逐步重新哈希。
完整 ProjectConfig 比较仍包含绝对路径；内容指纹相同不意味着可以直接修改路径后精确续训。
输入指纹不保证训练/验证划分独立，也不代替数据泄漏检查、硬件/软件一致性或模型效果评估。

## 本地验收

2026-10-23，Windows、Python 3.12.10、PyTorch 2.11.0+cpu：

- 新增 49 项绑定测试；全项目 1349 项通过，155.65 秒，关闭 pytest 缓存并使用独立临时目录。
- 覆盖相同路径替换数据和元信息、相同词表大小但不同 tokenizer、eval_batches/dtype 不符、
  缺失/损坏字段、未知版本，以及 v1/未绑定 v2 在三个 CLI 命令下的默认拒绝与显式兼容。
- 独立构造参考哈希，核对不同整数 dtype、端序、步幅、分块边界和 tokenizer JSON 重排。
- 验证绑定不符不会应用保存权重、optimizer 或 RNG；已有训练、精确续训与生成回归全部通过。
- Ruff 静态检查、格式检查、紧凑码风和 Git diff 空白检查通过。
- 未开展正式语料训练；GPU、Linux、远程 CI 和大语料性能未在本次验证。
