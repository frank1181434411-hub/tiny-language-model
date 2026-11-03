# 2026-11-03 完整代码回归

正式版清单 B 阶段已通过；本记录不代表最终版本、wheel 或正式 Release 已验收。

## 源码与环境

- 提交：`e305e475348fa9bb28c71288aada4748ca91002e`。执行前工作区干净；执行后仅补本文及进度文档，没有修改源码、测试或打包配置。
- 系统：Windows 11 x64，系统版本 `10.0.26300`。
- CPU：AMD Ryzen 9 7945HX，32 个逻辑处理器。
- Python：标准 CPython 3.12.10 x64；解释器 `D:/Python/.venvs/tiny-language-model/Scripts/python.exe`。
- 依赖：torch 2.11.0+cpu、numpy 2.5.4、regex 2026.9.29、PyYAML 6.0.3、pytest 9.1.1、ruff 0.17.1。
- `tiny_lm` 实际来自本仓库 `src/tiny_lm/__init__.py`，计算及文本依赖实际导入通过，CPU 张量运算成功。
- 设置进程环境 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONIOENCODING=utf-8`；所有命令串行执行，没有 pytest 并行 worker。

## 命令与结果

以下命令均在仓库根目录使用上述解释器执行，并加 `-X utf8`。机器可读命令与环境记录见 [JSON](regression-20261103.json)。

| 检查 | 结果 | 退出码 | 进程耗时 |
| --- | --- | --- | --- |
| `python -m pip check` | 无依赖冲突 | 0 | 1.672 秒 |
| 计算/文本依赖及项目导入，CPU 张量运算 | 通过 | 0 | 4.250 秒 |
| `python -m ruff check --no-cache .` | 通过 | 0 | 0.641 秒 |
| `python -m ruff format --no-cache --check .` | 56 files already formatted | 0 | 0.203 秒 |
| 全量 pytest，命令见下 | **1414 passed，0 failed，0 errors，0 skipped** | 0 | 234.297 秒 |

```powershell
& D:/Python/.venvs/tiny-language-model/Scripts/python.exe -X utf8 -m pytest -q -ra -p no:cacheprovider --basetemp=D:/Python/tiny-language-model/outputs/regression-20261103/pytest-temp --junitxml=D:/Python/tiny-language-model/outputs/regression-20261103/pytest.xml
```

pytest 自身报告 232.83 秒；表中为包含进程启动和退出的耗时。已等待实际退出，并核对 JUnit XML 的 1414 条测试计数。
没有因缺少依赖跳过测试，没有失败需要修复；未增加新测试或修改现有断言。
Ruff 按仓库既有格式排除配置执行，保留数值实现的紧凑码风，没有批量重排代码。

## 日志与范围

完整 stdout/stderr、JUnit XML 和运行脚本保存在本机 `outputs/regression-20261103/`，不随 Git 提交。
此次新建专用 pytest 临时目录，并关闭 pytest/Ruff 缓存，避免旧 `.pytest_cache` 权限问题；没有删除旧缓存。
CIM 硬件查询被当前权限拒绝，改为注册表读取 CPU 型号、Python 平台信息读取系统版本，没有改变系统权限。

本轮覆盖当前测试集的 Windows CPU 执行，不等同于 GPU、Linux 或其他 Python 版本验收。
未运行长时间语料训练，未构建新 wheel，未改版本或发布标签；已有训练权重不受影响。
下一步为清单 C：整理正式版本号、许可打包配置及安装说明；变更后需对最终候选再做回归和独立 wheel 验收。
