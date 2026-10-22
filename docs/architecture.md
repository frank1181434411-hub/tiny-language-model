# 架构与数据流

当前已实现：配置加载与校验、CLI 帮助/版本与各功能入口、平均交叉熵、
AdamW、全局梯度裁剪、学习率调度、数据采样、模型基础层、注意力、TransformerBlock、TransformerLM 前向和 checkpoint。下列计算流程中未标注实现的部分仍为规划，
已有接收 token 数组的 Python 训练/验证循环；BPE、Tokenizer、预处理和文件加载 API 已迁移；预处理、训练/续训、验证和单提示词生成 CLI 已连接。

```text
原始 UTF-8 文本
  → tokenizer/bpe.py：训练词表与 merges（已实现）
  → tokenizer/serialization.py：持久化 tokenizer 与元信息（原生 v1 已实现）
  → tokenizer/tokenizer.py：文本编码为 token IDs（已实现，迭代编码按独立记录处理）
  → preprocess.py：单/多文件编码，写入 token 数据与版本化元信息（已实现）
  → data.py：只读加载、校验和采样输入 x 与下一 token 目标 y（已实现）
  → model/layers.py：Linear、Embedding、RMSNorm、SiLU、SwiGLU（已实现）
  → model/attention.py：RoPE、softmax、缩放点积与因果多头注意力（已实现）
  → model/transformer.py：TransformerBlock、TransformerLM 与 logits 输出（已实现）
  → losses.py：计算平均交叉熵（已实现并有 CPU 数值/梯度测试）
  → train.py：数组训练循环、反向传播与梯度裁剪（已实现）
  → schedule.py + optim.py：学习率与参数更新（已接通数组训练）
  → evaluate.py：随机验证窗口的平均 loss（已实现）
  → checkpoint.py：保存与恢复模型、optimizer、配置、步数和 RNG（已实现）

提示文本 → tokenizer 编码 → checkpoint 重建模型 → generate.py 采样与 EOS 判定（已实现）
         → 逐步追加 token → tokenizer 解码 → 续写文本
```

`config.py` 提供结构化参数，TransformerLM.from_config 接受 ModelConfig 构造模型，
train 的 Python 接口从 ProjectConfig 构造模型与 optimizer；CLI 加载对应数据文件后调用该接口。
模块导入不能启动任务或访问训练数据；预处理命令在调用时检查输入和输出路径，训练设备检查由训练入口负责。
预处理 CLI 的参数、记录语义与输出保护见 [预处理命令](preprocess-cli.md)。
训练/验证入口校验模型词表和当前数据指纹，关闭文件映射，并支持即时训练指标输出；见 [训练/验证命令](training-cli.md)。
生成入口从 checkpoint 重建模型并推断 dtype，默认关闭 top-p 过滤，不构造 optimizer 或恢复训练 RNG；见 [文本生成](generation.md)。
checkpoint v1 包含配置、已完成更新次数、模型、优化器与随机数状态，
检查模型构造配置及参数顺序，并提供同目录临时文件替换。实验记录仍应关联 tokenizer 和数据版本。
采样支持独立的 CPU generator，可通过具名字典分别保存与恢复训练和验证采样器。
`norm_mode` 的 pre/post 位置由 TransformerBlock 决定，RMSNorm 本身不处理位置。
Python 构造接口支持 SwiGLU 与旧 SiLU 分支，配置中的 GELU 仍待实现并在模型构造时明确报错。
