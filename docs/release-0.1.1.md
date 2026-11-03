# v0.1.1 正式版发布草稿

2026-11-03：发布配置、最终候选完整回归和最终 wheel 独立安装验收已完成，**尚未发布**。详见[最终验收记录](final-0.1.1-checks.md)。
源码/Python 包版本为 `0.1.1`，计划 Git 标签为 `v0.1.1`；已有 `0.1.0` 历史快照和 alpha.2 附件保持原样。

## 安装与交付约定

- 当前已发布的用户安装包仍为 alpha.2，安装命令见 [README](../README.md)；本地 0.1.1 附件已验收待发布。
- 当前 main 的 editable 安装版本为 `0.1.1`；源码支持的依赖 extras 和命令入口不变。
- 正式附件为 `dist/tiny_language_model-0.1.1-py3-none-any.whl` 及 `.whl.sha256`，已通过最终验收，可用于发布。
- 安装新附件时沿用 README 的新环境步骤，将 wheel 路径换为最终 `0.1.1` 文件，版本输出应为 `0.1.1`。
- wheel 不含 `configs/` 和 `examples/`，须配合同一个 `v0.1.1` 标签的 Source code ZIP；不得拿其他版本配置替代而宣称同版本验收。
- 不上传语料、派生数据、训练词表、权重或个人复现包。

## 版本与打包配置核查

- `setuptools>=77.0.3`：按 [Python Packaging 指南](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) 设置支持新版许可字段的最低构建后端版本。
- 将旧 TOML 许可表和 `tool.setuptools.license-files` 迁移到 `project.license` 与 `project.license-files`，见 [setuptools 迁移说明](https://setuptools.pypa.io/en/stable/userguide/license_migration.html)。
- 原创和 Stanford 部分保留 MIT；OpenAI 文件为 Modified MIT，保留其全部原文并以 `LicenseRef-OpenAI-GPT2` 明确映射，详见[第三方声明](../THIRD_PARTY_NOTICES.md)。组合表达式 `MIT AND LicenseRef-OpenAI-GPT2` 不表示可任选一种许可。
- 运行依赖、Python 最低版本、算法、持久化格式和恢复约束不变。

本轮使用 CPython 3.12.10、setuptools 78.1.0 执行配置检查构建：

```powershell
& D:/Python/.venvs/tiny-language-model/Scripts/python.exe -X utf8 -m pip -v wheel . --no-deps --no-build-isolation --no-index --no-cache-dir --wheel-dir outputs/release-config-20261103
```

构建退出码 0，详细日志没有 warning/deprecation；检查版本、许可表达式、四份许可/声明逐字节内容、23 个源码文件和 `tiny-lm` 入口通过。
运行依赖元数据与原配置一致。检查用 wheel 仅在忽略目录 `outputs/release-config-20261103/`，不是供发布的最终附件。
构建日志及元数据检查 JSON 同目录保存；本轮没有执行该 wheel 的独立安装验收。

版本/导入专项命令：

```powershell
& D:/Python/.venvs/tiny-language-model/Scripts/python.exe -X utf8 -m pytest tests/test_cli.py tests/test_imports.py -q -ra -p no:cacheprovider --basetemp=D:/Python/tiny-language-model/outputs/release-config-20261103/pytest-temp
```

结果 **34 passed in 12.01s**，退出码 0。Ruff check 和 format --check 均通过（使用 `--no-cache`），退出码 0。
版本调整前的 1414 项回归仍作为[历史记录](regression-20261103.md)，不能替代本版本最终完整回归。

## 可用于发布的说明

标题：`v0.1.1 — 首个 Windows CPU 正式版`

首个正式版面向个人学习和小型语言模型实验，支持范围为 Windows 11 x64、标准 CPython 3.12、CPU FP32。
提供 BPE/Tokenizer、预处理、TransformerLM、训练与验证、checkpoint 保存恢复及单提示词续写。
本次整理配置英文报错、安装说明与许可元数据，未增加训练效果承诺或改变算法。

已知限制：已有 TinyStories 实验使用 SwiGLU；GELU 未做正式语料效果对照。
故事生成仍可能重复和不连贯；尚无 KV cache、AMP、梯度累积或批量生成。
Linux/GPU 及其他 Python 版本不在首版完整流程承诺内。

最终候选 1414 项测试通过，无失败或跳过；Ruff、依赖检查及新独立环境 wheel 安装和四步闭环全部通过。
附件为 `tiny_language_model-0.1.1-py3-none-any.whl`（60735 字节）及对应校验文件。
SHA-256：`58974deebcb3d5e77c37cf028198d9861bf155874d7bc25cd16ec78b5a7ff4d8`。
验收范围与环境见 `docs/final-0.1.1-checks.md`。

发布前核对最终提交与标签；Release 不选 Pre-release。发布后重新下载核对附件并登记真实发布日期。
