# 模型基础层迁移记录

来源：`D:/Python/CS336-Private/assignment1-basics/cs336_basics/nn.py`。
本次仅迁移 Linear、Embedding、RMSNorm、silu_fn 和 SwiGLU 到 `tiny_lm.model.layers`。
RoPE、softmax、注意力、TransformerBlock 与 TransformerLM 尚未迁移。

## 保留的行为

- Linear 无 bias，weight 形状为 `(out_features,in_features)`，保留 einsum 计算。
  初始化标准差为 `sqrt(2/(in_features+out_features))`，截断在正负三倍标准差内。
- Embedding weight 形状为 `(num_embeddings,embedding_dim)`，保留直接查表；
  初始化标准差为 1，截断范围为 `[-3,3]`。重复 token 的梯度累加到同一行。
- RMSNorm 在最后一维计算 `sqrt(mean(x**2)+eps)`，增益 weight 初始全 1，默认 eps 为 1e-5。
  它不减均值，也不保证输出方差为 1；不是 LayerNorm。
- SiLU 保留手写的 `x*sigmoid(x)`。SwiGLU 保留 `w2(silu_fn(w1(x))*w3(x))`，
  以及 w1、w3、w2 的创建顺序、参数名和无 bias 结构。
- 保留 device/dtype 构造参数，省略 dtype 时遵循 PyTorch 默认浮点 dtype。
  原 weight 与 `w1.weight`/`w3.weight`/`w2.weight` 状态字典名称保持不变。

## 修改与检查

- 构造维度必须为正 Python 整数，拒绝 bool；参数支持 FP16、BF16、FP32、FP64。
  RMSNorm 的 eps 必须为有限正数，避免零输入出现零分母。
- 浮点输入必须是稠密张量；Linear、RMSNorm、SwiGLU 检查最后一维。
  不在每次前向扫描浮点输入的 NaN/Inf；输入与权重仍应由调用者保证有限。
- Embedding 仅接受 int32/int64 的稠密 token 张量，并要求与权重处于相同设备。
  负 ID 和超出词表的 ID 显式英文报错，避免原来的负索引绕到词表末尾。
  当前范围检查使用 `.item()`，在 GPU 上可能产生同步开销，尚未优化。
- RMSNorm 仅将 FP16/BF16 输入提升到 FP32 计算，保留 FP32/FP64，输出转回输入 dtype。
  原来的无条件 `.to(torch.float32)` 会丢失 FP64 精度，甚至将有限大数转成无穷大。
  FP16/BF16 的参数梯度仍按参数 dtype 保存，没有增加 FP32 主参数。
- Linear、Embedding、SiLU 和 SwiGLU 未统一提升到 FP32；它们保持原来的 dtype 计算语义。
  极端输入仍可能在当前计算 dtype 下溢出，当前不承诺任意有限输入都能稳定计算。
- 非连续输入和空 batch 可用；Embedding 还支持标量和空 token 张量。
  不新增 padding、稀疏梯度、权重绑定、自动设备搬移或完整模型功能。

## 配置、依赖与导入

未来由模型构造入口将 d_model、d_ff、vocab_size 映射到对应层；本次没有接通配置构造。
`norm_mode` 表示 pre/post 归一化位置，由未来 TransformerBlock 决定。
只实现 SwiGLU；配置接受 `ffn_type: gelu` 不代表 GELU 已有实现。

显式导入 `tiny_lm.model.layers` 会加载 PyTorch，因为类继承 nn.Module。
基础包、`tiny_lm.model`、配置与 CLI 仍保持轻量。现有 `compute` 依赖已包含 PyTorch，
本次不增加依赖。导入审计将 layers 纳入计算模块，并检查不改变 CPU 随机状态。
参数分配与随机初始化仅发生在显式实例化时。

源码与新测试保持紧凑空格、单行括号、连续 import 无空行；
文档字符串起始三引号独占一行、内容中无空行。Ruff 对这些文件保留相应格式例外。

## 测试范围

`tests/test_layers.py` 使用自行编写的小张量，不复制课程测试或 fixtures：

- 四种 dtype 的前向数值与输入/参数梯度，对照 PyTorch 函数的 FP64 计算。
  对照使用同一份已量化的输入和权重，按各 dtype 设置误差容限。
- Linear、RMSNorm、SwiGLU 的 FP64 输入和全部参数有限差分 gradcheck，以及 SiLU gradcheck。
- RMSNorm 的 FP64 大值、低精度大值和零输入回归；Embedding 重复索引的梯度累加。
- 初始化形状、截断界限、可训练参数、状态字典名称与 SwiGLU 状态加载一致性。
- 非连续/空输入、非法尺寸/dtype/eps/索引和 Embedding 设备不一致。

CPU 测试不代表 GPU、autocast、完整训练或完整 checkpoint 恢复已验收。

2026-10-21 本地验收使用项目 `.venv/Scripts/python.exe`（Python 3.12.10）：

- 新增基础层测试 **138 passed**；全项目 **431 passed**，耗时 23.42 秒。
- Ruff 静态检查与格式检查通过；连续 import、单行括号和 8 处新增/更新文档字符串检查通过。
- 导入审计通过：轻量模块不加载计算依赖，计算模块不写文件或改变 CPU 随机状态。
- 检查使用 D 盘独立临时目录并禁用 pytest 缓存；没有修改全局 Python 环境。
- 未运行远程 CI、GPU 测试、正式训练，也未修改旧作业源码。

从项目根目录执行的主要检查命令（PowerShell）：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --no-cache --check .
```
