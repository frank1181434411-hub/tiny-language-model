# optimizer 迁移与精度检查

`tiny_lm.optim.AdamW` 和 `clip_gradient_norm` 来自 Assignment 1 的
`cs336_basics/optimizer.py`，保留原算法、默认超参数和主要变量名。
本次仅迁移数值组件，不启用训练 CLI，也不迁移 scheduler 或 checkpoint 文件接口。

## 原算法约定

- AdamW 先更新一阶矩 m、二阶矩 v 和每个参数的 step。
- 更新步长为 `lr * sqrt(1-beta2**step) / (1-beta1**step)`，
  分母为 `sqrt(v) + eps`。
- 先做 Adam 更新，再对更新后的参数乘以 `1-lr*weight_decay`。
  不调整为 PyTorch AdamW 的衰减顺序或 eps 位置，不能要求与其结果完全一致。
- `grad=None` 的参数不更新、不衰减、不增加 step；零梯度仍参与更新。
- 保留参数组、closure、step 返回 closure loss，以及基类的 zero_grad 接口。
- 梯度裁剪仍按所有梯度的全局 L2 范数判断，只在超过 max_norm 时乘以
  `max_norm/(total_norm+1e-6)`，返回 None。

## 参数检查

所有新增错误消息使用英文。lr、weight_decay、max_norm 必须是有限非负数；
eps 必须是有限正数，避免零梯度和零分母产生 NaN。
betas 必须包含两个有限数值，分别在 `[0,1)` 内；不接受 bool 或张量超参数。
默认值、参数组覆盖、后续新增参数组与 step 时修改的参数组均会检查。
参数和梯度只支持稠密的 FP16、BF16、FP32、FP64 张量。

AdamW 与 losses 一样约定输入张量有限，不在每一步扫描全部 NaN/Inf。
梯度裁剪本来就计算总范数，因此总范数非有限时会在修改梯度前报错。
超出浮点表示范围的输入/中间量不保证可计算。

## 精度与恢复

- FP16/BF16 梯度、矩状态和参数更新计算使用 FP32；参数和梯度写回原 dtype。
- FP32/FP64 保持原计算精度。低精度参数没有额外 FP32 主参数副本，
  因此小更新仍可能在写回时被舍入，不能宣称等价于完整 FP32 训练。
- 梯度裁剪在提升精度后求范数和缩放，保留原来的 Python 标量全局累加方式。
  GPU 上逐个参数取 `.item()` 的同步开销尚未优化。
- 保存仍使用 `state_dict()`。恢复会从传入状态重新取矩张量并转换为所需
  device/dtype，防止基类按低精度参数转换状态后造成不可逆精度损失。
  验收覆盖本实现保存的状态；不承诺任意第三方 optimizer checkpoint 兼容。

## 导入与依赖

显式导入 `tiny_lm.optim` 会加载 PyTorch，因为 AdamW 需要继承
`torch.optim.Optimizer`。基础包、配置、CLI 和 loss 模块仍不在导入时加载 torch。
继续通过 `compute` 可选依赖安装，不向基础框架强加 PyTorch。
导入测试分别检查轻量模块和计算模块；后者在 PyTorch 自身初始化后
检查项目模块没有读数据、写文件或启动外部进程等副作用。

## 验证

`tests/test_optim.py` 包括：

- FP64 标量原公式对照，覆盖四种 dtype、多步矩状态与参数更新。
- 刻意使用较大的 eps 和 weight_decay，区分不同的公式与衰减顺序。
- FP16/BF16 大、小梯度的矩状态精度检查。
- 参数组、缺失梯度、零梯度、零学习率和 closure。
- 四种 dtype 的序列化恢复，与不中断运行逐步完全一致；
  恢复后训练不会修改传入 checkpoint 的矩张量。
- 超参数错误、稀疏梯度、复数参数，以及非法后续参数组。
- 全局裁剪、零阈值、无梯度、非连续梯度、FP16 大范数和非有限值。

本机使用 `D:\Python\.venvs\tiny-language-model\Scripts\python.exe`：

```powershell
& D:\Python\.venvs\tiny-language-model\Scripts\python.exe -m pytest tests/test_optim.py
& D:\Python\.venvs\tiny-language-model\Scripts\python.exe -m pytest
```

CPU CI 已扩展为运行全项目测试；未验证 GPU 行为。
数值实现和 optimizer 测试遵循紧凑空格风格，Ruff 格式化对此显式排除，
静态检查仍保留。

2026-10-20 本机 CPU 验收：Python 3.12.10、PyTorch 2.11.0+cpu。
全项目 164 项测试通过（其中 optimizer 50 项），Ruff 静态检查、
格式检查及 `pip check` 均通过。测试临时目录改用 D 盘，
绕开既有系统临时目录的访问权限问题；未修改原作业目录。
