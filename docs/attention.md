# 注意力迁移记录

来源：`D:/Python/CS336-Private/assignment1-basics/cs336_basics/nn.py`。
本次迁移 RotaryPositionalEmbedding、softmax、scaled_dot_product_attention 和
CausalSelfAttention 到 `tiny_lm.model.attention`。保留手写运算，没有替换成融合注意力内核。
原课程目录未修改；本次测试独立编写。

## 精度约定

- 支持 FP16、BF16、FP32、FP64。RoPE、softmax 和缩放点积注意力的低精度输入提升到
  FP32 计算，FP32/FP64 保持计算精度，最终返回输入 dtype，与 RMSNorm 的输出约定一致。
- 注意力核心的 QK 打分、softmax 和对 V 的加权求和都在计算 dtype 中完成。
  明确禁用这些局部运算的 autocast，避免已经提升的矩阵乘法再被自动降精度。
- CausalSelfAttention 的四个 Linear 投影保持基础层既有的 dtype/autocast 行为。
  注意力核心不能挽救投影阶段已经产生的无穷大或精度损失；没有增加 FP32 主参数。
- 没有对 Q/K/V 的每个输入元素做有限性扫描，调用者仍须保证输入有限。
  softmax 的行最大值若非有限则报错，可拒绝 NaN、正无穷和整行负无穷；
  部分位置为负无穷的合法 mask 可用。极端值仍可能超出计算 dtype 的表示范围。
- 低精度输出会发生正常舍入，例如 10 万项均匀 FP16 概率之和可能略偏离 1，
  但不会再因为 FP16 指数和溢出而整体变成 0。

## RoPE 位置与缓存

- 保留相邻奇偶维配对，旋转角为 `position*theta**(-2*i/d_k)`。
  频率计算采用数学等价的 `exp(-log(theta)*(2*i/d_k))`，避免先形成过大的 theta 幂。
- theta 必须为有限正数，d_k 为正偶数，max_len 为正整数；非有限缓存角度明确报错。
- 输入为 `(...,seq_len,d_k)`，要求非空序列。位置为 int32/int64，且在 `[0,max_len)` 内。
  输入、位置和缓存必须在同一设备，不自动移动位置。
- 位置支持共享 `(seq_len,)`、与输入去掉最后特征维后的完整形状，
  以及省略 heads 维的 batch 位置。例如输入 `(B,H,S,D)` 时支持 `(S,)`、`(B,S)`、`(B,H,S)`；
  输入 `(B1,B2,H,S,D)` 时还支持 `(B1,B2,S)`。其他形状报错，不猜测广播方式。
- 保留非持久化 cos_cached/sin_cached buffer。构造时创建 FP32 缓存，前向需要 FP64 时
  根据原 theta 和整数配置重建。模块的 dtype/device 转换会使缓存失效，下次前向重新生成；
  `.half().double()` 不会直接沿用已量化的缓存。
- max_len 固定，不自动扩容，不引入 KV cache。缓存不进入 state_dict，未来 checkpoint
  必须保存匹配的 theta、head dimension、max_len 等构造配置。

## 缩放点积注意力与 mask

- Q/K 最后一维相等，K/V 序列长度相等；V 特征宽度可以不同于 Q/K。
  Q/K/V 必须同 dtype、同 device，其前导 batch 维必须可广播。
- mask 仅接受同设备的稠密 bool 张量，`True` 表示允许关注。
  必须能广播到最终打分形状，不能隐式引入新的 batch 维。
  例如 `(B,H,Q_len,K_len)` 的打分可配 `(Q_len,K_len)` 或 `(B,1,Q_len,K_len)` 的 mask。
- 每个 query 至少允许一个 key，否则报错；不把全遮罩行替换成虚构的均匀概率或零输出。
- 显式检查空序列、空特征维、非法 dtype、维度、广播和设备，并使用英文错误消息。
  允许空 batch，支持非连续张量。当前 `.item()` 校验在 GPU 上可能带来同步开销。

## 因果多头注意力

- 输入保持 `(batch,seq_len,d_model)`，检查正维度、整除关系和输入形状。
  仅启用 RoPE 时要求 head dimension 为偶数。
- q_proj、k_proj、v_proj、output_proj 的参数名、创建顺序和初始化沿用旧实现。
  bias=False 保持默认；bias=True 明确报错，避免旧版静默忽略参数。
- theta 和 max_seq_len 同时提供或同时省略。没有启用 RoPE 却提供位置时明确报错。
  未提供位置时使用 `0..seq_len-1`；启用 RoPE 时序列长度不得超过 max_seq_len。
- 因果 mask 保留对角线，按输入序列顺序决定可见性。位置 ID 影响旋转角，
  不改变因果 mask 的顺序。本次不增加 padding、dropout、KV cache 或完整 Transformer。

## 格式、导入与验证

沿用紧凑空格、单行括号、连续 import 无空行；注释公式的运算符也不留两侧空格。
文档字符串的起始三引号单独一行，内容中无空行。Ruff 保留对应文件的格式例外。
显式导入 attention 会加载 PyTorch；基础包、model 包入口、配置与 CLI 仍轻量。
导入不创建参数/缓存，不执行演示，不修改 CPU 随机状态；现有计算模块导入审计已扩展。

测试覆盖四种 dtype 的前向与梯度、FP64 有限差分、因果性、mask 与 batch 广播、
不同位置形状、只保留允许位置的梯度、空 batch、非连续输入、非法参数，
以及 FP16 指数和/打分溢出、RoPE 缓存精度往返等回归。还验证了 CPU autocast 下
注意力核心不会被降精度。没有验证 GPU、完整 AMP 训练、远程 CI 或完整模型。

2026-10-21 本地验收使用项目 `.venv/Scripts/python.exe`（Python 3.12.10）：

- 新增注意力测试 **126 passed**；全项目 **557 passed**，耗时 30.66 秒。
- Ruff 静态检查、格式检查、连续 import、单行括号与 7 处文档字符串样式检查通过。
- 导入审计通过，计算模块导入不产生文件写入或 CPU 随机状态变化。
- 测试使用 D 盘独立临时目录并关闭 pytest 缓存，没有修改全局环境或安装新依赖。
- 没有下载数据/模型、运行正式训练、修改旧作业源码、提交或推送 Git。

从项目根目录执行的主要检查命令（PowerShell）：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --no-cache --check .
```
