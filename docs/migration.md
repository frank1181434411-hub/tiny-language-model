# 源码迁移计划

用户已指定旧源码目录 `D:/Python/CS336-Private/assignment1-basics`。
目前已迁移其中的 `cs336_basics/losses.py`、`cs336_basics/optimizer.py`
以及 `cs336_basics/scheduler.py`、`checkpointing.py`、`data.py` 中的采样函数、`nn.py` 中的基础层、注意力、TransformerBlock 与 TransformerLM 前向；
其他映射仍为计划。optimizer 的原算法、精度与测试见 [迁移记录](optimizer.md)。
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
| train_bpe | `src/tiny_lm/tokenizer/bpe.py` | BPE 统计、合并、特殊 token |
| tokenizer | `src/tiny_lm/tokenizer/tokenizer.py` | 编码、解码、迭代处理 |
| 重复的 tokenizer 保存加载 | `src/tiny_lm/tokenizer/serialization.py` | 统一格式与版本 |
| nn.py | `src/tiny_lm/model/{layers,attention,transformer}.py` | 基础层、注意力、Block 与 LM 前向已迁移；生成采样待迁移 |
| data sampling / 数据预处理 | `src/tiny_lm/data.py`、未来预处理入口 | 采样已迁移；数据格式、文件加载、预处理待迁移 |
| loss / cross_entropy | `src/tiny_lm/losses.py` | 已迁移；平均 loss、精度处理与 CPU 梯度验证 |
| optimizer | `src/tiny_lm/optim.py` | 已迁移；AdamW、全局裁剪、精度与状态恢复 |
| scheduler | `src/tiny_lm/schedule.py` | 已迁移；线性预热、余弦衰减与边界校验 |
| checkpointing | `src/tiny_lm/checkpoint.py` | 已迁移；v1 格式、原子保存、状态/配置/RNG 恢复与校验 |
| 训练/验证流程 | `src/tiny_lm/train.py`、`evaluate.py` | 已接通数组接口；采样、更新、验证、保存与续训；文件/CLI 待接入 |
| inference | `src/tiny_lm/generate.py` | 自回归循环、EOS 与采样 |

按用户要求已迁移独立计算组件并组装 TransformerLM 前向，提供 ModelConfig 构造入口。
数组训练循环已支持精确 CPU 续训；后续迁移 tokenizer、数据格式与文件加载，接入 CLI 与文本生成。
每批迁移确认接口与正确性后再接通 CLI。
PyTorch、NumPy、regex 已在本地迁移环境按需安装；包依赖随实际实现逐步声明。

## 待复核事项（尚未修复或验证）

- 任意文本分块编码是否改变预分词/BPE 结果；UTF-8 字节、词边界与流式状态如何处理。
- 特殊 token 的匹配优先级、相邻与跨块边界、允许/禁止策略是否一致。
- 配置选项与实现是否对应；当前 `norm_mode` 指归一化位置，
  不能直接推定旧实现支持所有枚举；模型词表与 tokenizer 是否匹配。
- 数据 dtype 的范围是否覆盖 token ID；是否记录 shape、token 数、字节序、
  tokenizer/数据版本与训练/验证划分，是否防止数据泄漏。
- checkpoint 的 GPU/分布式状态、GradScaler 与数据工作进程恢复仍待扩展和验证；
  v1 的 CPU 精确恢复、配置校验与写入失败保护已验证。
- 生成 EOS 是否正确终止；温度、top-k、最大长度、空提示与上下文截断的边界。
- 是否存在模块顶层数据读取、训练、目录创建、演示、随机种子修改或设备初始化。

请同时提供相关测试、作业 adapters、依赖文件、旧配置和调用示例，
以及已知失败用例或边界问题。无需提供大语料或模型权重。
