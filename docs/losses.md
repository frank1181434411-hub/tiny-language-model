# loss 迁移与精度检查

`tiny_lm.losses.cross_entropy` 从用户的 Assignment 1 实现迁移，
保留手写交叉熵流程和 `m`、`target_logits`、`log_sum_exp`、`loss` 等变量名。
删除未使用的 `vocab_size`，精简中文注释，新增可读错误信息使用英文。

## 接口与精度约定

- logits 形状为 `(..., vocab_size)`，targets 为对应前导形状的 int64 token ID。
- 返回所有 token 的平均 loss 标量，不支持 padding、ignore_index 或概率标签。
- FP16/BF16 输入提升至 FP32 计算并返回 loss；FP32/FP64 保留原精度。
- 两个张量需位于同一设备，输入非空。形状、dtype 和设备错误明确报错；
  越界 ID 由 PyTorch 的 gather 拒绝，不额外进行需要设备同步的范围扫描。
- 输入分数需为有限数值；函数不扫描 NaN/Inf。仍受所用浮点类型的表示范围限制。
- 调用函数时才加载 PyTorch；导入包、导入 loss 模块和配置 CLI 不加载计算依赖。

## 如何检查精度

1. **解析值对照**：所有类别分数相同时，loss 应为词表大小的自然对数。
2. **公共偏移检查**：给每个 token 的所有类别分数加同一个常数，
   loss 和梯度应基本不变。测试选择偏移后仍能准确表示类别差值的输入。
3. **高精度对照**：将实际收到的输入提升为 FP64，和 PyTorch 的交叉熵比较
   loss 及梯度；对低精度输入的梯度按其 dtype 的舍入精度设置容差。
4. **梯度检查**：对 FP64 输入运行有限差分 `gradcheck`，独立核对自动求导。
5. **压力场景**：大公共偏移、大词表、极端分数差、FP16/BF16 输入。

高精度参考必须来自同一份已量化输入。如果类别差值在转换成低精度输入时就已丢失，
之后提升到 FP32/FP64 不能恢复这些信息，不能把它全部归因于 loss 实现。

本次在 CPU 上对旧源码的实测：

| 输入 | 旧 loss | FP64 参考 |
| --- | --- | --- |
| FP32，4 类均为 0 | 1.386294365 | 1.386294361 |
| FP32，4 类均为 1e8 | 0 | 1.386294361 |
| FP16，100000 类均为 0 | inf | 11.512925465 |

第二个例子中旧实现的梯度仍正确，第三个例子的最大梯度绝对误差约为 0.99999；
因此必须分别检查 loss 和梯度。
迁移实现保持在减去最大值后的分数上完成计算，避免加回大数再相减；
低精度输入的指数、求和等计算使用 FP32。

## 安装与运行测试

```powershell
Set-Location D:\Python\tiny-language-model
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,compute]"
.\.venv\Scripts\python.exe -m pytest tests/test_losses.py
```

仅安装框架开发依赖时，缺少 PyTorch 会明确跳过 loss 测试。
CI 保留轻量框架检查，另加 CPU loss 测试任务；没有数据下载或训练步骤。
本次迁移不启用训练 CLI，不迁移 scheduler，也不改动旧作业目录。

## 本地验收

2026-10-20，Windows、Python 3.12.10、PyTorch 2.11.0+cpu：

- loss 专项：31 passed。
- 全项目：113 passed（17.59 秒），含 CLI、配置与无导入副作用检查。
- Ruff 静态检查、格式检查和 `pip check` 均通过。
- 当前项目的可编辑安装已离线刷新，`compute` 依赖组已生效。

上述为 CPU 验收；新增 GitHub Actions 任务和 GPU 行为尚未实际验证。
