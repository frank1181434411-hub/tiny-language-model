# 本地检查记录

日期：2026-10-20（Asia/Shanghai）。

本记录对应初始框架验收；后续按用户要求准备的计算与课程测试依赖见
[迁移依赖环境](dependencies.md)。

## 实际环境

- Windows，CPython 3.12.10。
- 项目解释器：`D:\Python\tiny-language-model\.venv\Scripts\python.exe`。
- 使用 `python -m pip install -e ".[dev]"` 安装项目。
- PyYAML 6.0.3、pytest 9.1.1、Ruff 0.17.1。
- 这些是本次检查版本，不是已验证的所有平台依赖锁定清单。

## 实际执行结果

以下命令从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --no-cache --check .
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m tiny_lm check-config --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m pip check
```

- Ruff：静态检查与格式检查均通过。
- pytest：**82 passed**，本次报告耗时 9.63 秒。
- smoke 配置检查成功，输出路径解析到项目的 `data/`、`artifacts/`、`outputs/`，
  没有创建这些目录或检查其中数据存在性。
- `pip check`：没有依赖冲突。
- 测试实际运行了模块入口与安装生成的 `tiny-lm` 控制台入口；
  帮助、版本、配置检查通过，五个未实现子命令均明确失败并返回 `2`。
- 配置测试覆盖两份示例、维度/枚举/数值/未知字段等错误与路径解析。
- 独立进程导入全部包模块通过；审计检查未发现数据读取、文件写入、
  目录创建、子进程启动或网络连接，无标准输出/错误输出，未导入大型计算依赖。
- 已初始化本地 Git，无提交、无远程地址。虚拟环境、egg-info 和运行数据被忽略，
  演示语料与实验说明保留在待跟踪文件中。

## 遇到的问题与实际覆盖范围

首次依赖安装被沙箱网络限制阻止，随后获准联网安装。
初始 MSYS2 Python 3.14.6 的平台标识为 `mingw_x86_64_ucrt_gnu`，
Ruff 的源码构建依赖不支持该平台，安装失败；没有将该次尝试记为通过。
已保留失败环境目录 `.venv-msys2/`（不用于运行），另用本机现有标准
CPython 3.12.10 创建 `.venv`，随后安装和检查成功。

沙箱对 Ruff 缓存目录写入有限制，本次关闭 Ruff 与 pytest 缓存后执行检查；
这不跳过 lint、格式或测试内容。常规本地环境与 CI 仍可使用 README 的默认命令。

GitHub Actions 配置已创建，但尚未在远程执行。Python 3.11/3.14 的 CI、
Linux 环境、真实训练、数据处理、恢复训练、生成和 GPU 行为尚未实际验证。
本次没有下载语料或模型，也没有进行训练。
