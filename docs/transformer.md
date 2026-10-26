# TransformerBlock 与 TransformerLM 迁移记录

来源：Assignment 1 的 `cs336_basics/nn.py` 中的 TransformerBlock 和 TransformerLM。
迁移目标为 `tiny_lm.model.transformer`，复用已迁移的
CausalSelfAttention、RMSNorm、Linear 和 SwiGLU。
已完成 Block 与完整模型前向组装，训练/验证 CLI 已接入；生成逻辑已迁移至独立模块，见 [文本生成](generation.md)，未修改课程目录。

## 结构和接口

- 保留参数创建顺序、初始化方式和名称：attn、ln1、ln2、ffn。
- Pre-Norm：先归一化再进入子层；每个子层输出分别加回残差。
- Post-Norm：先做子层与残差相加，再归一化；注意力和前馈各一次。
- `use_rms_norm=False` 用 Identity 替换两个归一化层，保持旧消融选项。
- 输入和输出均为 `(batch,seq_len,d_model)`，支持非连续输入与空 batch，
  不接受空序列。前向不会原地修改输入。
- max_seq_len 与 theta 同时提供时启用 RoPE，同时为 None 时禁用；
  token_positions 原样传递给注意力，沿用其形状、设备和范围检查。
- 默认名称为 `norm_mode="pre_norm"`、`ffn_type="swiglu"`。
  兼容旧的 pre/post、SwiGLU/SiLu；内部规范为 pre_norm/post_norm 和 swiglu/silu/gelu。
- 保留旧 SiLU 分支的 Linear→SiLU→Linear，隐藏宽度使用传入的 d_ff，
  不根据旧注释擅自强制为 4*d_model。
- GELU 分支使用 Linear→GELU→Linear，两个 Linear 均无 bias，隐藏宽度为 d_ff。
  激活使用 `torch.nn.GELU(approximate="none")`，对应 `0.5*x*(1+erf(x/sqrt(2)))`，不是 tanh 近似或门控 GEGLU。
- 非法选项和输入使用英文错误消息。非法 norm_mode 不再静默返回输入。

## 配置边界

现有 ModelConfig 的 pre_norm/post_norm 和 swiglu/gelu 可直接对应 Block。
模型通过 `TransformerLM.from_config(config.model,device=...,dtype=...)` 构造。
工厂接受 ModelConfig；context_length 对应 Block 的 max_seq_len，
rope_theta 对应 theta。设备与参数 dtype 由调用方指定，不读取配置文件或权重，
也不重设随机种子。CLI 的 check-config 仍只校验配置，不创建模型。

配置中预留的 gelu 已接通 Block、LM 和训练/验证/生成入口；默认仍为 swiglu。
旧实验分支 silu 和 use_rms_norm 可通过 Block/LM 的直接构造接口使用，
未扩展 YAML schema，也没有将 check-config 变为模型可运行性检查。

在新的实验配置副本中将 `model.ffn_type` 设置为 `gelu`，使用新的 output_dir 并从头训练。
已有 SwiGLU checkpoint 不能通过修改配置直接变成 GELU 模型。
GELU 和 SiLU 虽然都使用 `ffn.0.weight`、`ffn.2.weight`，参数形状相同，
checkpoint 仍依据 model_config.ffn_type 区分并拒绝混用；v2 格式不变，旧分支的名称、参数顺序与结构保持不变。
同一个 d_ff 下 GELU 使用两组投影权重，SwiGLU 使用三组；切换分支不等于参数量匹配的消融实验。

## 完整模型

- 保留 token_embeddings、layers.N、ln_final、lm_head 的参数名称与创建顺序。
- 每层 Block 独立创建，不共享参数；输入 Embedding 和输出投影不绑定权重。
- Embedding→多层 Block→最终 RMSNorm→词表投影；关闭 RMSNorm 时，
  Block 内与最终归一化都替换为 Identity。Post-Norm 也保留旧模型的最终 RMSNorm。
- 输入是形状 `(batch,seq_len)` 的稠密 int32/int64 token IDs，范围
  `[0,vocab_size)`，须与模型在同一设备。序列长度须在 `[1,context_length]`。
- 位置沿用 `0..seq_len-1` 并扩展到各 batch。支持非连续输入与空 batch；
  空 batch 可前向，但不能用作要求非空目标的交叉熵输入。
- 返回 `(batch,seq_len,vocab_size)` 的原始 logits，不加 softmax、
  不自动计算 loss，不截断过长序列，不引入 KV cache 或 padding mask。
- 最初模型迁移没有包含 generate/top-p 或训练/验证 CLI；这些入口已在后续工作中接通。
  CPU 小张量集成测试连接了 get_batch、模型、loss、裁剪、调度与 AdamW，
  验证两步反向和参数更新；这不是正式训练或效果验收。

## 精度

支持 FP16、BF16、FP32、FP64，沿用基础层与注意力已有约定：
RMSNorm、RoPE 和注意力核心对低精度输入使用 FP32 中间计算，FP64 不降精度。
Linear 投影、SiLU/SwiGLU/GELU 和残差加法沿用张量 dtype/autocast 行为；
不宣称整个 Block 都在 FP32 计算，也不增加 FP32 主参数。
普通同 dtype 前向返回相同 dtype，低精度输出会正常舍入。
有限输入和权重是计算前提，极端值仍可能在投影、前馈或残差阶段溢出。
本次未验证 GPU 或完整 AMP 训练。

GELU 使用 PyTorch 内置激活，不在 Python 层强制提升 FP16/BF16 或降低 FP64。
内部计算由对应 PyTorch kernel 决定；普通同 dtype 前向的输出与参数梯度保持各自 dtype。
数值对照使用独立 FP64 erf 公式，包括负值、零点、正值与饱和区间。

## 验证和码风

独立 FP64 参考使用 PyTorch functional 线性层、SiLU、缩放点积注意力，
以及独立的 GELU erf 公式和 RMSNorm/RoPE 计算；参考输入和权重来自相同的已量化数据。
覆盖前向及全部输入/参数梯度、FP64 有限差分、两种归一化位置、
三种前馈结构、关闭 RMSNorm、未来 token 不影响历史输出或梯度、
默认和自定义位置、参数名称、序列化恢复及非法参数。

显式导入 transformer 会加载 torch，已纳入计算模块导入审计。
基础包、配置和 CLI 仍保持轻量；导入模块不创建模型、写文件或改变 CPU 随机状态。
实现与测试遵循连续 import、紧凑运算符/逗号、单行括号调用，
文档字符串三引号单独一行、正文不留空行。Ruff 已配置相应例外。

### GELU 接入验收

2026-10-26，Windows、项目 Python 3.12.10、PyTorch 2.11.0+cpu，设置 OMP/MKL 单线程：

- 相关测试 **575 passed**，耗时 338.94 秒；未重跑全部测试。
- GELU 的 FP16/BF16/FP32/FP64 前向、输入/参数梯度对照独立 FP64 erf 公式，
  Block 与 LM 的 FP64 有限差分通过；沿用已有数值容差，没有为 GELU 放宽。
- 四种 dtype 的 checkpoint 中断恢复与连续训练逐张量一致；同形状 GELU/SiLU 混用被拒绝，加载失败不修改目标权重。
- YAML→模型、两种命令入口的 FP32/FP64 训练/恢复/验证，以及真实训练权重的文本生成通过。
- 相关配置、导入、输入绑定、生成和训练回归通过；Ruff 静态/格式检查、文档链接及紧凑格式检查通过。
- 未修改现有实验配置、训练数据、词表或历史 checkpoint，未运行正式语料训练或 GPU 检查。

项目虚拟环境中的测试命令：

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m pytest tests/test_transformer.py tests/test_training_cli.py tests/test_generate_cli.py tests/test_checkpoint.py tests/test_config.py tests/test_cli.py tests/test_imports.py tests/test_generate.py tests/test_binding.py tests/test_training.py -q -p no:cacheprovider
```

实际运行还将 TEMP/TMP 与 pytest basetemp 设置为 D 盘独立临时目录，未修改全局环境。

### 初始迁移验收

2026-10-21 首批 Block 的本机 CPU 验收使用 Python 3.12.10 和 PyTorch 2.11.0+cpu：

- Block 专项 81 passed，全项目 638 passed（13.99 秒）。
- Ruff 静态检查、格式检查和 Git diff 空白检查通过。
- 实现与测试的单行函数调用、文档字符串、等号/比较符及逗号空格检查通过。
- 导入审计通过；测试使用 D 盘独立临时目录并禁用 pytest 缓存。
- 未验证 GPU、远程 CI、完整模型和训练流程。

## TransformerLM 数值验收方法

在相同的已量化输入权重上使用独立 FP64 参考，覆盖两层模型的 logits 和全部参数梯度，
以及前置/后置归一化、SwiGLU/SiLU/GELU、开关 RMSNorm、四种 dtype。
额外检查 FP64 参数有限差分、重复 token 梯度累加、因果性、独立层与未绑定权重、
四种精度的序列化恢复、上下文边界、非法 token 和 smoke 配置构造。

单层测试精度阈值保持不变。完整模型的 FP16/BF16 梯度逐张量检查相对 L2 与
最大绝对误差，以 `8*finfo(dtype).eps` 乘以对应参考范数/最大值为容差，
另计一个最小非正规数量化步长（L2 项乘以元素数平方根）。
这样既检查每个参数张量，也考虑多层累加抵消与微小梯度的有限表示精度。
FP32/FP64 仍使用严格逐元素对照。

一组 Post-Norm/SwiGLU 参考输入的完整梯度相对 L2 误差为：
FP16 约 0.076%、BF16 约 0.475%、FP32 约 2.61e-7、FP64 约 2.77e-16。
这些是小规模确定性测试数据，不代表所有模型规模或训练数值的误差上界。

2026-10-21 完成 TransformerLM 后，同一 CPU 环境验收：

- 新增 LM 65 项测试；Block/LM 合计 146 passed，全项目 703 passed（14.02 秒）。
- Ruff 静态检查、格式检查、Git diff 空白检查均通过。
- AST/token 检查确认实现与测试的连续 import、单行调用、文档字符串与紧凑空格符合约定。
- 导入审计、配置构造、四种精度恢复和两步数值组件集成检查通过。
- 未运行正式训练、生成、GPU 或远程 CI。
