# 源码迁移计划

用户已指定旧源码目录 `D:/Python/CS336-Private/assignment1-basics`。
目前已迁移其中的 `cs336_basics/losses.py`、`cs336_basics/optimizer.py`
以及 `cs336_basics/scheduler.py`、`checkpointing.py`、`data.py` 中的采样函数、`nn.py` 中的基础层、注意力、TransformerBlock 与 TransformerLM 前向；
另已迁移 BPE、Tokenizer 编解码和原生保存格式，其余状态见下表。
optimizer 的原算法、精度与测试见 [迁移记录](optimizer.md)。
loss 的精度修改与测试范围见 [迁移记录](losses.md)。
调度的参数范围与步数约定见 [迁移记录](scheduler.md)。
采样的数组约定、token 检查与随机状态见 [迁移记录](data.md)。
基础层的初始化、精度与梯度检查见 [迁移记录](layers.md)。
注意力的精度、mask、位置与缓存策略见 [迁移记录](attention.md)。
Block 与完整 LM 的结构、配置构造和验证见 [迁移记录](transformer.md)。
checkpoint 的版本、步数、配置与随机状态恢复见 [迁移记录](checkpoint.md)。
数值组件已连接为 token 数组训练/验证接口，步数、日志与验证约定见 [训练记录](training.md)。

| 旧模块/功能 | 目标 | 迁移重点 |
| --- | --- | --- |
| train_bpe | `src/tiny_lm/tokenizer/bpe.py` | 已迁移；增量统计、确定性合并、特殊 token 与参数校验 |
| tokenizer | `src/tiny_lm/tokenizer/tokenizer.py` | 已迁移；编解码、独立记录迭代、输入校验与原生文件接口 |
| 重复的 tokenizer 保存加载 | `src/tiny_lm/tokenizer/serialization.py` | 原生 v1 JSON 已实现；旧格式导入待迁移 |
| nn.py | `src/tiny_lm/model/{layers,attention,transformer}.py` | 基础层、注意力、Block 与 LM 前向已迁移；生成采样待迁移 |
| data sampling / 数据预处理 | `src/tiny_lm/data.py`、`preprocess.py` | 采样、文本预处理、v1 元信息与只读加载已迁移；预处理 CLI 已接入 |
| loss / cross_entropy | `src/tiny_lm/losses.py` | 已迁移；平均 loss、精度处理与 CPU 梯度验证 |
| optimizer | `src/tiny_lm/optim.py` | 已迁移；AdamW、全局裁剪、精度与状态恢复 |
| scheduler | `src/tiny_lm/schedule.py` | 已迁移；线性预热、余弦衰减与边界校验 |
| checkpointing | `src/tiny_lm/checkpoint.py` | 已迁移；v1 格式、原子保存、状态/配置/RNG 恢复与校验 |
| 训练/验证流程 | `src/tiny_lm/train.py`、`evaluate.py` | 已接通数组接口与文件 CLI；采样、更新、验证、保存、续训和即时指标输出 |
| inference | `src/tiny_lm/generate.py` | 自回归循环、EOS 与采样 |

按用户要求已迁移独立计算组件并组装 TransformerLM 前向，提供 ModelConfig 构造入口。
数组训练循环已支持精确 CPU 续训；预处理、只读文件加载及训练/验证 CLI 已接通；后续接入文本生成和 checkpoint 历史数据指纹绑定。
每批迁移确认接口与正确性后再接通 CLI。
PyTorch、NumPy、regex 已在本地迁移环境按需安装；包依赖随实际实现逐步声明。

BPE 与版本化保存已经迁移，具体改动和验证范围见 [BPE 迁移记录](bpe.md)。
共享 `rules.py` 已统一训练和编码的预分词与最长特殊 token 匹配。
Tokenizer 的模式、ID 校验和记录边界见 [Tokenizer 迁移记录](tokenizer.md)。
默认单文件、显式多文件、数据格式与完整性检查见 [预处理迁移记录](preprocess.md)。

## 待复核事项（尚未修复或验证）

- 任意文本分块的 UTF-8 字节、词边界与流式状态仍待设计；当前迭代编码明确按独立记录处理。
- 特殊 token 最长匹配、相邻边界、识别/普通文本模式已验证；跨块识别与禁止特定 token 策略尚未实现。
- 配置选项与实现是否对应；当前 `norm_mode` 指归一化位置，
  不能直接推定旧实现支持所有枚举；运行 CLI 已检查模型实际词表与 tokenizer 是否匹配，并拒绝未实现的 GELU。
- v1 数据已记录 dtype、token 数、字节序、tokenizer 指纹与内容哈希；
  训练/验证切分、防止数据泄漏及 checkpoint 自动绑定数据指纹仍待运行流程集成。
- checkpoint 的 GPU/分布式状态、GradScaler 与数据工作进程恢复仍待扩展和验证；
  v1 的 CPU 精确恢复、配置校验与写入失败保护已验证。
- 生成 EOS 是否正确终止；温度、top-k、最大长度、空提示与上下文截断的边界。
- 是否存在模块顶层数据读取、训练、目录创建、演示、随机种子修改或设备初始化。

请同时提供相关测试、作业 adapters、依赖文件、旧配置和调用示例，
以及已知失败用例或边界问题。无需提供大语料或模型权重。
