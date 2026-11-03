# 0.1.1 最终候选与 wheel 验收

2026-11-03 完成。**验收通过，尚未发布。** 原始命令、环境、文件清单及结果见 [JSON](final-0.1.1-checks.json)。

## 源码身份与完整回归

基于提交 `84b814852b612b553133975b06ac6e44373b3b3a` 加本地 0.1.1 发布配置变更进行检查。
不是声称该提交本身已包含最终版本；提交前差异保存于本地日志目录。
JSON 记录最终源码和构建输入散列，用于核对后续发布提交。测试期间仅调整 README 的进度措辞，构建前已完成；源码及 pyproject.toml 未变。

环境：Windows 11 x64（10.0.26300），AMD Ryzen 9 7945HX，CPython 3.12.10，
torch 2.11.0+cpu、numpy 2.5.4、regex 2026.9.29、PyYAML 6.0.3、pytest 9.1.1、ruff 0.17.1。
设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONIOENCODING=utf-8`，串行执行。

- `pip check`、依赖实际导入和 CPU 张量运算通过，退出码 0。
- `ruff check --no-cache .` 与 `ruff format --no-cache --check .` 通过，退出码 0；保留既有格式排除配置。
- **1414 passed，0 failed，0 errors，0 skipped**；pytest 报告 218.68 秒，进程耗时 220.094 秒，退出码 0，JUnit 计数一致。
- 完整测试包含错误配置、未知 checkpoint 版本、已记录指纹不匹配不可绕过、未绑定恢复需显式选择等拒绝行为。

```powershell
& D:/Python/.venvs/tiny-language-model/Scripts/python.exe -X utf8 -m pytest -q -ra -p no:cacheprovider --basetemp=D:/Python/tiny-language-model/outputs/final-011-20261103/pytest-temp --junitxml=D:/Python/tiny-language-model/outputs/final-011-20261103/pytest.xml
```

## 最终附件

- [tiny_language_model-0.1.1-py3-none-any.whl](../dist/tiny_language_model-0.1.1-py3-none-any.whl)：**60735 字节**。
- [校验文件](../dist/tiny_language_model-0.1.1-py3-none-any.whl.sha256)。
- SHA-256：`58974deebcb3d5e77c37cf028198d9861bf155874d7bc25cd16ec78b5a7ff4d8`。

使用 setuptools 78.1.0、pip 25.0.1，从当前源码执行：

```powershell
& D:/Python/.venvs/tiny-language-model/Scripts/python.exe -X utf8 -m pip -v wheel . --no-deps --no-build-isolation --no-index --no-cache-dir --wheel-dir dist
```

构建退出码 0，详细日志没有 warning/deprecation。该构建使用已有后端环境，不声称完成联网隔离构建验收。
核验 ZIP CRC、RECORD 散列和大小、23 个源码文件逐字节一致、版本 0.1.1、入口、Python 声明和依赖元数据。
四份许可/声明与仓库逐字节一致，表达式为 `MIT AND LicenseRef-OpenAI-GPT2`。
内容仅有 tiny_lm 与 dist-info，没有语料、权重或缓存。旧版本 wheel 未覆盖。

## 仓库外独立安装

新环境位于 `D:/Python/wheel-011-acceptance-20261103/.venv`，`include-system-site-packages=false`。
从既有 wheel 缓存离线正常解析并安装依赖，没有复制旧 site-packages；随后安装上述最终 wheel 的 compute/tokenizer extras。
运行依赖版本同上，新环境 setuptools 为 81.0.0。安装来源为该 wheel，非 editable，direct_url 的 SHA-256 与最终附件一致。

12 个实际命令全部成功：pip check、模块版本、控制台版本、帮助、配置检查、训练 tokenizer、训练/验证数据预处理、训练两步、恢复到四步、独立验证、生成。
七个流程命令由审计包装器禁止读取原仓库，未触发违规读取；导入来源全部位于新环境 site-packages。

- 使用自带 toy_corpus、smoke 配置和 256 词表（零次 merge），临时配置中的路径指向验收目录。
- 验证文本用明确的 LF 字节写入；输入散列保存在 JSON。
- 训练到第 2 步，恢复到第 4 步；独立验证 seed=42、4 批、128 tokens，loss=`5.641384243965149`。
- 从 `Hello` 生成 32 个新 token，因长度结束；不以四步模型的生成质量作为门槛。
- GELU 前向检查输出有限；未改变已有 SwiGLU 权重。

## 日志与下一步

仓库本地日志：`outputs/final-011-20261103/`；独立安装日志：`D:/Python/wheel-011-acceptance-20261103/logs/`。
环境、缓存、测试 checkpoint 不随 Git 提交；只有最终 wheel/校验文件和验收文档进入版本控制。
本轮未运行长时间语料训练，不扩展 Linux、GPU 或其他精度承诺。

下一步提交并推送最终文件，核对提交包含 JSON 中对应源码后创建 `v0.1.1` 非预发布 Release，
上传这份 wheel 与校验文件。发布后重新下载并核对 SHA-256，登记真实发布日期；不移动历史标签。
