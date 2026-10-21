# scheduler 迁移记录

来源：`D:/Python/CS336-Private/assignment1-basics/cs336_basics/scheduler.py`。
迁入 `tiny_lm.schedule.get_lr_cosine_schedule`，保留函数名、参数名、主要变量和
线性预热加余弦衰减公式。只依赖标准库 `math`，没有内部状态或导入副作用。

## 参数与端点

- `it`、`warmup_iters` 为非负 Python 整数；拒绝 bool。
- `cosine_cycle_iters` 是衰减结束的绝对步数，必须大于 `warmup_iters`。
  例如预热 100、结束 1000，衰减阶段长度是 900。
- 学习率接受 Python int/float，必须有限；最大值为正，最小值在 `[0,最大值]` 内。
  参数校验在阶段分支之前执行，错误信息为英文。
- 有预热时 `it=0` 返回 0，预热中允许低于 `min_learning_rate`。
  预热先算步数比例再乘学习率，避免大但有限的学习率乘步数时溢出。
- `it=warmup_iters` 直接返回最大值，也涵盖零预热的起点。
  原公式在最大值 0.01、最小值 0.001 时可能得到 0.010000000000000002；
  端点直接返回可避免此处的加减舍入误差，内部余弦计算仍使用 Python 浮点数。
- `it>=cosine_cycle_iters` 直接返回最小值；所有合法调用返回 float。
- 最小值等于最大值时，预热仍执行，随后保持恒定。

## 步数与未来接入

`it` 约定为本次更新前已完成的更新次数。从 0 开始，有预热时第一次返回 0。
未来运行入口可将 `learning_rate`、`min_learning_rate`、`warmup_steps`、`max_steps`
分别映射到最大值、最小值、预热步数和结束步数。
若执行 `range(max_steps)`，最后一次更新使用 `max_steps-1`，结束点不会用于该次训练的更新；
不在函数内部暗自加减一步。训练入口尚未接入。

调度函数无需保存单独的内部状态；未来 checkpoint 应保存配置及已完成的更新次数，
恢复后据此查询下一次学习率。当前只验证相同步数查询一致性，尚未验证真实训练恢复。

## 验证范围

`tests/test_schedule.py` 独立编写，覆盖阶段端点、零预热、单步衰减、相等的学习率、
曲线范围与单调性、乱序查询和续接步数、极大学习率、返回类型及非法参数。
现有导入审计覆盖该模块，不需要 PyTorch；现有 CPU CI 会自动收集新测试。

2026-10-21 本地验收：使用项目 `.venv/Scripts/python.exe`（Python 3.12.10），
全项目 **233 passed**（其中 scheduler 69 项），耗时 59.39 秒。
Ruff 静态检查、格式检查及新增文档字符串样式检查均通过。
初轮边界测试发现上述预热终点舍入问题，修正后全量通过。
测试禁用缓存，并将 TEMP/TMP 指向 D 盘新建的独立临时目录，避免既有目录权限问题。
这些环境变量仅用于测试进程，不修改系统设置。远程 CI 和真实训练恢复未执行。

从项目根目录运行的检查命令（PowerShell）：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --no-cache --check .
```
