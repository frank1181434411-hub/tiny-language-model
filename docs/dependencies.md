# 源码迁移依赖环境

使用项目 `.venv`（Windows CPython 3.12.10）准备 Assignment 1 源码迁移和测试。
运行实现需要 PyTorch、NumPy、regex；课程测试另外需要 einops、jaxtyping、
psutil、tiktoken、pytest 和 pytest-timeout。

这是一组按需安装的迁移依赖，清单在 `requirements/migration.txt`。
项目默认安装仍只包含 PyYAML；框架 CI 另安装 dev、tokenizer 与 data 可选依赖。
已迁移的数据采样需要 NumPy，现与 PyTorch 一起列入 `compute` 可选依赖组。
按实际平台准备好 PyTorch 后，可用 `python -m pip install -e ".[dev,compute]"`
安装当前计算组件及测试依赖。基础安装仍不强制安装 NumPy/PyTorch。
`data` extra 单独提供 NumPy，`tokenizer` extra 提供 regex；
`python -m pip install -e ".[dev,tokenizer,data]"` 可测试预处理与文件加载，无需 PyTorch。
未被当前实现或测试使用的 W&B、einx、ty 等课程环境依赖不在该清单中。

## PowerShell 安装命令

```powershell
Set-Location D:\Python\tiny-language-model
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements/migration.txt
.\.venv\Scripts\python.exe -m pip check
```

PyTorch 2.11.0 对齐旧项目的版本范围和历史测试环境。
CPU wheel 来源遵循 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v2110)，
不安装 torchvision、torchaudio 或 CUDA 运行库。
未来 GPU 环境应按实际平台重新选择 PyTorch 构建；本清单没有固定 CUDA 下载源。

## 平台限制

课程原始测试使用 Python 3.12 泛型语法，并在 `test_tokenizer.py` 中直接导入
Unix 的 `resource` 模块。安装依赖不会消除 Windows 平台限制。
原始 tokenizer 测试需在 Linux 运行，或迁移时处理平台兼容性；
不通过安装同名第三方包替代标准库模块。

安装依赖不会自动迁入旧源码，也不会启用尚未实现的训练、验证或生成命令。

## 本次安装与检查结果

2026-10-20，实际安装在 `D:\Python\tiny-language-model\.venv`：

| 依赖 | 版本 |
| --- | --- |
| PyTorch | 2.11.0+cpu |
| NumPy | 2.5.4 |
| regex | 2026.9.29 |
| einops | 0.8.2 |
| jaxtyping | 0.3.11 |
| psutil | 7.2.2 |
| tiktoken | 0.14.0 |
| pytest | 9.1.1 |
| pytest-timeout | 2.4.0 |

实际检查结果：

- 上述模块均可导入；解释器确认位于项目虚拟环境。
- PyTorch 确认为 CPU 构建，CPU 张量矩阵运算、NumPy 数据转换和反向传播通过。
- `python -m pip check` 通过，没有依赖冲突。
- 框架测试 `python -m pytest -p no:cacheprovider`：82 passed（8.02 秒）。
- Ruff 静态检查和格式检查通过。

本次没有运行旧课程测试，没有下载语料、模型或 tokenizer 数据，没有启动训练。
上表记录实际安装版本；迁移依赖清单中的下限不是完整环境锁定文件。
