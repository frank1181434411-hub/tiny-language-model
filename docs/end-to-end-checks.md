# 小型端到端示例实跑验收

2026-10-26 按 [端到端文档](end-to-end.md)顺序执行全部十个 PowerShell 命令块。
使用仓库自带文本和极小 CPU 模型，没有下载依赖、语料或模型，没有使用或修改 TinyStories 产物。
执行前工作区干净，Git HEAD 为 `82f5d29bc14c13c00c490e3f400869b5b71c0f5f`。

## 环境与命令适配

- 原训练设备，Windows 11 10.0.26300、CPython 3.12.10、PyTorch 2.11.0+cpu。
- 解释器为 `D:/Python/.venvs/tiny-language-model/Scripts/python.exe`；本机没有文档默认的项目内 `.venv`。
- NumPy 2.5.4、regex 2026.9.29、PyYAML 6.0.3、pytest 9.1.1、Ruff 0.17.1。
- CPU FP32，OMP_NUM_THREADS=1、MKL_NUM_THREADS=1；依赖已齐全，按文档允许跳过安装，pip check 返回 No broken requirements found。
- 从文档提取命令块顺序执行，只替换解释器、跳过安装并增加 UTF-8 输出和日志记录。
  PowerShell 使用进程级 ExecutionPolicy Bypass，没有改变持久系统设置。

本次完整 PowerShell 进程退出码 **0**，耗时 **24.12 秒**，包括多个 Python 进程启动、依赖检查、预处理、训练、验证和生成。
这不是四步纯训练的性能指标。未测量峰值内存、温度或功耗。

## 实际结果

| 检查项 | 实测结果 |
| --- | --- |
| 文档命令块 | 10 个全部成功 |
| tokenizer | 256 字节词表，0 个合并，无特殊 token |
| 训练数据 | 387 tokens；元信息、ID 范围与完整解码核对通过 |
| 验证数据 | 60 tokens；元信息、ID 范围与完整解码核对通过 |
| 第一段训练 | completed_updates=2，生成 step_00000002.pt |
| checkpoint 恢复 | completed_updates=4，生成 step_00000004.pt |
| 训练日志 | 步数依次为 1、2、3、4，无重复或缺失；loss 有限 |
| 独立验证 | loss=5.641384243965149；4 batches、128 个目标 tokens、seed=42 |
| 文本生成 | 提示 Hello 为 5 tokens；新增 32 tokens；finish_reason=length |

训练日志如下，首步学习率为零符合 smoke 配置的一步预热语义：

| 步数 | 学习率 | 训练 loss | 训练中验证 loss |
| --- | ---: | ---: | ---: |
| 1 | 0 | 5.756402 | — |
| 2 | 0.001 | 5.744615 | 5.707888 |
| 3 | 0.000775 | 5.715435 | — |
| 4 | 0.000325 | 5.823607 | 5.737265 |

生成文本包含控制字符和 UTF-8 替换字符，符合未充分训练的字节模型预期；不能视为有用语言输出。
为了避免在 Markdown 中嵌入控制字符，完整文本与 token IDs 以 JSON 转义形式保存在本地 verification.json。

## 日志与问题处理

- 演示产物：`artifacts/e2e-demo-v1/`，包含配置、文本、tokenizer、两份 token 数据和两个 checkpoint。
- 验收记录：`outputs/e2e-acceptance-20261026-v1/`。
  - end-to-end.source.md：执行前文档快照。
  - run-documented-example.ps1：实际执行的命令脚本。
  - stdout.txt、stderr.txt：UTF-8 原始输出和四步训练指标。
  - run.json：实际命令、环境适配、退出码、耗时及文档 SHA-256。
  - verification.json：断言核对结果、依赖版本、验证与生成结果、全部演示产物 SHA-256。
- 上述目录均由 Git 忽略；本记录进入版本控制，未将模型权重或二进制数据加入 Git。

PowerShell 流程已经成功结束后，外层记录工具尝试通过默认 GBK 标准输出回显生成文本，
遇到替换字符导致 UnicodeEncodeError。原始 UTF-8 文件完整，随后读取并解析所有日志、核对全部结果成功；
没有重跑训练、覆盖产物或忽略 CLI 错误。本次将 UTF-8 设置补入教程，后续读取日志也应明确使用 UTF-8。

本次未修改模型或 CLI 实现。没有运行全量单元测试；验收证据来自真实文档命令及输出/文件核验，
另外检查文档链接与 Git diff 空白。
这只证明本机已有环境下示例可运行，不证明干净环境安装、跨设备逐位一致、连续/恢复轨迹逐张量一致，
也不覆盖突然断电、GPU、发布包或正式语料效果。
