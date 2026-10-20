# 源码迁移计划

用户已指定旧源码目录 `D:/Python/CS336-Private/assignment1-basics`。
目前已迁移其中的 `cs336_basics/losses.py` 和 `cs336_basics/optimizer.py`；
其他映射仍为计划。optimizer 的原算法、精度与测试见 [迁移记录](optimizer.md)。
loss 的精度修改与测试范围见 [迁移记录](losses.md)。

| 旧模块/功能 | 目标 | 迁移重点 |
| --- | --- | --- |
| train_bpe | `src/tiny_lm/tokenizer/bpe.py` | BPE 统计、合并、特殊 token |
| tokenizer | `src/tiny_lm/tokenizer/tokenizer.py` | 编码、解码、迭代处理 |
| 重复的 tokenizer 保存加载 | `src/tiny_lm/tokenizer/serialization.py` | 统一格式与版本 |
| nn.py | `src/tiny_lm/model/{layers,attention,transformer}.py` | 分离基础层、RoPE/注意力和模型组装 |
| data sampling / 数据预处理 | `src/tiny_lm/data.py`、未来预处理入口 | 格式元信息、输入/目标错位和随机状态 |
| loss / cross_entropy | `src/tiny_lm/losses.py` | 已迁移；平均 loss、精度处理与 CPU 梯度验证 |
| optimizer | `src/tiny_lm/optim.py` | 已迁移；AdamW、全局裁剪、精度与状态恢复 |
| scheduler | `src/tiny_lm/schedule.py` | warmup、衰减与边界 |
| checkpointing | `src/tiny_lm/checkpoint.py` | 保存、恢复、配置和格式版本 |
| 训练/验证脚本 | `src/tiny_lm/train.py`、`evaluate.py` | 整理运行入口与统计语义 |
| inference | `src/tiny_lm/generate.py` | 自回归循环、EOS 与采样 |

按用户要求先后迁移独立的 loss 和 optimizer。后续迁移 tokenizer、数据/模型、
调度/checkpoint，最后接入训练、验证与生成。
每批迁移确认接口与正确性后再接通 CLI。
PyTorch、NumPy、regex 已在本地迁移环境按需安装；包依赖随实际实现逐步声明。

## 待复核事项（尚未修复或验证）

- 任意文本分块编码是否改变预分词/BPE 结果；UTF-8 字节、词边界与流式状态如何处理。
- 特殊 token 的匹配优先级、相邻与跨块边界、允许/禁止策略是否一致。
- 配置选项与实现是否对应；当前 `norm_mode` 指归一化位置，
  不能直接推定旧实现支持所有枚举；模型词表与 tokenizer 是否匹配。
- 数据 dtype 的范围是否覆盖 token ID；是否记录 shape、token 数、字节序、
  tokenizer/数据版本与训练/验证划分，是否防止数据泄漏。
- checkpoint 是否保存配置、模型、优化器、步数与随机数状态；
  恢复时是否检查兼容性；中断写入和不可信文件应如何处理。
- 生成 EOS 是否正确终止；温度、top-k、最大长度、空提示与上下文截断的边界。
- 是否存在模块顶层数据读取、训练、目录创建、演示、随机种子修改或设备初始化。

请同时提供相关测试、作业 adapters、依赖文件、旧配置和调用示例，
以及已知失败用例或边界问题。无需提供大语料或模型权重。
