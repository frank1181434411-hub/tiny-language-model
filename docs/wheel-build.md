# wheel 构建记录（2026-10-28）

[文档索引](README.md) · [来源与许可](provenance.md)

本次完成正式 wheel 文件构建及静态内容检查，未执行 wheel 独立安装或端到端运行验收。
`py3-none-any` 是包的构建标签，不代表所有 Python 版本、操作系统或 GPU 环境已经测试。

## 构建环境与命令

- 源码基点：`cdcd8552a5d23c16656ed676972dbf560853f17e`。
- Python 3.12.10，pip 25.0.1，setuptools 78.1.0，wheel 0.48.0。
- 使用已有跨设备验收虚拟环境中的构建工具，未更改全局环境，未联网下载依赖。
- `--no-build-isolation` 使用该环境中已有的构建后端；这不是干净构建环境验收。
- 初次执行因 Windows 写入权限失败；获准在项目目录写入后重试成功，未修改目录 ACL。

实际构建命令（PowerShell）：

```powershell
Set-Location D:/Python/tiny-language-model
$BuildPython = 'D:/Python/reproduction-acceptance-20261028/.venv/Scripts/python.exe'
New-Item -ItemType Directory -Path artifacts/wheel-build-20261028 -Force | Out-Null
& $BuildPython -I -X utf8 -m pip --log artifacts/wheel-build-20261028/pip.log wheel . --no-deps --no-build-isolation --no-index --no-cache-dir --wheel-dir dist
```

构建退出码为 0，完整 pip 日志位于 `artifacts/wheel-build-20261028/pip.log`。
日志包含现有 `project.license` 表和 `tool.setuptools.license-files` 的弃用提示；
本次 setuptools 仍成功处理它们，四份许可文件均已实际进入 wheel。
后续可单独迁移至新版许可元数据写法，并同步调整最低 setuptools 版本。

## 产物与检查结果

- 文件：`dist/tiny_language_model-0.1.0-py3-none-any.whl`。
- 大小：59728 字节。
- SHA-256：`51116e78da2f740f86bac1dba6cd9db704639ed776884cf8c23ac6aed42b9551`。
- 压缩包共 32 个文件，其中 23 个 Python 源文件与当前 `src/tiny_lm/` 逐字节一致。
- ZIP 完整性检查通过，RECORD 中记录的文件大小与 SHA-256 均匹配。
- 包名 `tiny-language-model`，版本 `0.1.0`，要求 Python `>=3.11`。
- 入口信息为 `tiny-lm = tiny_lm.cli:main`；这里只核对入口声明，尚未执行安装后的命令。
- 基础依赖 PyYAML，dev/compute/tokenizer/data 可选依赖均与构建配置一致。
- LICENSE、THIRD_PARTY_NOTICES.md、Stanford 和 OpenAI 许可文本均随包收录，并与本地文件逐字节一致。
- 所有文件位于 `tiny_lm/` 或包的 `.dist-info/` 下；没有打入语料、权重、材料 ZIP、虚拟环境、测试或缓存。

配置模板及示例文本保留在 Git 仓库，本 wheel 不包含它们，运行时需另行提供。
wheel 和构建日志已被 `.gitignore` 忽略，本次 Git 提交仅记录文档，不强制添加二进制产物。

下一步是创建独立环境安装这个 wheel，在源码目录外验证导入来源、CLI 和四步端到端流程。
本次没有运行训练、模型推理或全量测试，也没有上传发行附件或发布到 PyPI。
