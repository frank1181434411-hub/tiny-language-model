# 文本生成

已迁移用户作业 `cs336_basics/nn.py` 中的 generate、top-p 过滤，以及 inference.py 的模型加载/编解码流程。
新实现位于 `tiny_lm.generate`，通过 `generate` CLI 调用；原作业目录未修改。
首版处理一条非空提示词，不提供批量生成、交互对话或 KV cache。

## Git Bash 命令

先按 [训练/验证命令](training-cli.md) 准备数据并产生 checkpoint，再运行：

```bash
cd /d/Python/tiny-language-model
PYTHON=/d/Python/.venvs/tiny-language-model/Scripts/python.exe

"$PYTHON" -m tiny_lm generate \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke/step_00000004.pt \
  --prompt 'Hello' \
  --max-new-tokens 32 \
  --temperature 0.8 \
  --top-k 20 \
  --seed 42
```

默认 top-p=1，不启用核采样过滤；可显式追加 `--top-p 0.9`。
生成需要 `.[compute,tokenizer]`；其他机器替换 PYTHON 路径。
smoke 只验证运行流程，4 次更新不能代表有意义的语言生成能力。

## 参数与输出

| 参数 | 默认值 | 约定 |
| --- | --- | --- |
| `--config` | 必填 | 提供 tokenizer 路径、设备、默认种子；模型字段须匹配 checkpoint |
| `--checkpoint` | 必填 | 项目 v2 checkpoint；从中重建模型并自动推断参数 dtype；旧 v1 须显式允许缺失绑定 |
| `--allow-unbound-checkpoint` | 关闭 | 允许缺失 tokenizer 绑定的旧文件；已有指纹不匹配仍拒绝 |
| `--prompt` | 必填 | 必须编码为至少一个 token；不自动补 BOS，空白文本可以输入 |
| `--max-new-tokens` | 64 | 非负整数，只限制新增 token 数；0 返回原提示 |
| `--temperature` | 1.0 | 有限正数；0 不表示贪心，会明确报错 |
| `--top-k` | 不过滤 | 整数，范围 1 到实际词表大小 |
| `--top-p` | 1.0 | 有限数值，范围 (0,1]；1 不进行核采样过滤 |
| `--seed` | training.seed | 独立 CPU 采样器种子，范围 0 到 2⁶⁴−1 |
| `--eos-token` | 已注册的 `<\|endoftext\|>`，不存在则无 | 可显式指定另一个已注册特殊 token；未注册时报错 |
| `--special-mode` | recognize | 提示词识别已注册特殊 token；ordinary 按普通文本编码 |

成功时 stdout 输出 JSON，包含 text（完整提示加续写）、token_ids（完整序列）、
num_prompt_tokens、num_generated_tokens、finish_reason（eos/length）、seed 和自动推断的 dtype。
不另行输出“仅续写”的解码字符串，避免把一个跨 token 的 UTF-8 字符拆开解码。
全部字节拼接后再解码，沿用 tokenizer 的 errors=replace；无效生成字节会显示替换字符。
EOS 保留在 token_ids 和 text 内，不隐式删除特殊 token。

命令只读取 YAML、tokenizer 和 checkpoint；训练/验证文件及 output_dir 均可不存在。
不创建输出文件或更新 checkpoint。参数、文件、权重和采样错误返回退出码 2，并在 stderr 给出诊断；
成功返回 0。命令行相对路径以当前目录为准，YAML 内相对路径以 YAML 目录为准。

## 采样算法与精度

每步取完整已生成序列的最后 context_length 个 token，重算前向，取最后位置 logits，
再依次执行温度缩放、可选 top-k、可选 top-p、手写 softmax 和 multinomial 采样。
每次截断后的窗口位置从 0 开始，与原作业一致；返回值始终保留完整原始提示和新增 token。

- 温度为 1 时不缩放，否则保留原算法除以 temperature+1e-8 的行为。
- top-k 使用第 k 大分数为阈值，同分候选全部保留，因此实际候选数可能超过 k。
  top-k=1 在最大值唯一时选取该 token；并列最大值仍进行采样。
- top-p 保留原算法的降序累计概率和右移掩码，保留首个使累计概率严格超过 p 的候选，至少保留一个。
  累计值刚好等于 p 时下一项也被保留；同分排序沿用 PyTorch sort，不额外更改顺序。
- 同时指定时先 top-k，再在剩余候选归一化后的分布上应用 top-p。
- FP16/BF16 logits 提升 FP32 后缩放、过滤与归一化；FP32/FP64 保持原精度。
  模型投影和残差仍沿用模型参数精度，不宣称整个前向都使用 FP32。
- 模型输出和温度缩放结果须有限；NaN、无穷或缩放溢出明确报错，不自动改分布继续采样。
- 为使用独立 CPU generator，每步把概率复制到 CPU 后采样，再把 token 放回模型设备。
  相同条件和种子可复跑，但不承诺不同硬件/软件下或与旧全局 RNG 流逐位一致。

只检查新生成 token 是否为 EOS；提示本身以 EOS 结尾仍会开始续写。
达到新增长度上限或采样到 EOS 后结束，没有批量提示的结束掩码/补齐规则。

## Python 接口与状态保护

`load_generation_model(src,*,device="cpu",config=None,tokenizer=None,allow_unbound_checkpoint=False)` 从 checkpoint.model_config 构造 TransformerLM，
从 embedding 权重推断 dtype；可传 ModelConfig 核对配置。严格核对类、构造参数、权重键、
shape、dtype、layout 和数值有限性，最终返回 eval 模式的模型。
与训练恢复共享模型状态校验，但不构造 optimizer、不恢复训练 RNG。
构造时保护 CPU RNG，再移动模型到目标设备；读取仍会在 CPU 加载完整 checkpoint，
所以文件中的 optimizer tensors 仍影响读取内存峰值。只支持 CPU/CUDA，GPU 尚未实测。

`generate(model,prompt_ids,max_new_tokens,*,eos_token_id=None,temperature=1.0,top_p=1.0,top_k=None,generator=...)`
要求 prompt_ids 为同设备的 int32/int64 张量，形状 (1,非零长度)，全部 ID 在词表范围内。
generator 必须是独立 CPU torch.Generator，不能使用全局默认 generator。
返回新的同设备 int64 张量；即使 max_new_tokens=0，也返回副本。

生成使用 no_grad、eval，并在成功或异常时逐模块恢复原有模式，包括混合 train/eval 模式。
不修改提示、参数或已有梯度；不会消耗全局采样 RNG。显式 generator 会推进，
生成中途失败时不回滚已消耗的采样状态。RoPE 等非持久化缓存可按现有模型逻辑建立。

## 当前边界

接受项目 v1/v2 checkpoint 和原生 tokenizer JSON，不猜测无版本旧权重、旧双文件 tokenizer，
不自动去除旧 `_orig_mod.` 前缀。CLI 核对 tokenizer 历史指纹，同大小却内容不同也会拒绝。
v1 或未绑定 tokenizer 的 v2 默认拒绝文本生成，确认旧 tokenizer 后可显式追加 --allow-unbound-checkpoint；
已有指纹不匹配不能绕过。低层 load_generation_model 不传 tokenizer 时仅加载模型，不验证外部词表。
详细规则见 [输入绑定](checkpoint-binding.md)；KV cache、批量生成与大模型性能优化仍待实现。

## 本地验收

以下保留生成接入时的记录；最新全量 1349 项通过，见 [输入绑定验收](checkpoint-binding.md)。

2026-10-22，Windows、Python 3.12.10，使用 `D:/Python/.venvs/tiny-language-model`：

- 新增 100 项生成 API/CLI 测试通过；全项目 1300 项通过，144.55 秒，关闭 pytest 缓存。
- 四种精度对照独立 softmax/候选枚举参考，覆盖温度、top-k/top-p、组合过滤、并列最大值与越界参数。
- 验证固定种子复现、滑动上下文、EOS、零步生成、完整提示保留、异常时模式恢复，
  以及参数/梯度和全局 RNG 保持不变。
- 四种权重精度的模型专用加载与原模型生成一致；拒绝结构、权重 shape/dtype、缺失键和非有限权重。
- 模块和控制台入口验证中文/emoji、自动推断精度、固定种子、独立 EOS 与特殊 token 模式。
  用真实文件训练生成的 checkpoint 已跑通文本续写，测试中不依赖任何下载的语料或权重。
- 全部命令的帮助和必填参数检查、Ruff、格式、紧凑码风、checkpoint 回归及导入审计通过。
- 原作业目录未修改；未进行正式训练，GPU、Linux 和远程 CI 未在本次验证。
