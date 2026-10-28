# 本机 wheel 独立安装验收（2026-10-28）

[文档索引](README.md) · [构建记录](wheel-build.md) · [机器可读结果](wheel-install-checks.json)

已在本机 Windows CPU 上完成 wheel 的独立环境安装、命令入口和四步端到端验收。
本次没有使用 editable 安装，不代表另一台设备、Linux 或 GPU 的 wheel 已验收。

## 环境与安装来源

- 新目录：`D:/Python/wheel-acceptance-20261028`，所有运行命令的工作目录均在这里。
- 新环境：目录下的 `.venv`，由标准 CPython 的 `venv` 新建，未复制旧环境或启用系统包。
- 本机硬件沿用同日[设备采集记录](cross-device-checks.md)：Intel N150、4 核 / 4 线程、约 15.75 GiB 内存。
- 本次实测 Python 3.12.10、Windows 11 10.0.26100、torch 2.11.0+cpu、numpy 2.5.4、regex 2026.9.29、PyYAML 6.0.3。
- OMP_NUM_THREADS=1、MKL_NUM_THREADS=1；实测 torch intra-op=1、interop=4，未使用 CUDA。
- 使用 `tiny_language_model-0.1.0-py3-none-any.whl`，SHA-256 为
  `51116e78da2f740f86bac1dba6cd9db704639ed776884cf8c23ac6aed42b9551`，与仓库旁置校验文件一致。

验收目录只复制 wheel、`configs/smoke.yaml` 和 `examples/toy_corpus.txt` 作为输入，未复制项目源码。
安装命令如下（PowerShell，详细命令与输出另存日志）：

```powershell
Set-Location D:/Python/wheel-acceptance-20261028
$WheelPython = './.venv/Scripts/python.exe'
& $WheelPython -I -X utf8 -m pip install --disable-pip-version-check --no-input torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
& $WheelPython -I -X utf8 -m pip install --no-compile --disable-pip-version-check --no-input './tiny_language_model-0.1.0-py3-none-any.whl[compute,tokenizer]' numpy==2.5.4 regex==2026.9.29 PyYAML==6.0.3
& $WheelPython -I -m pip check
```

安装访问包索引，依赖 wheel 命中本地缓存；不能将此次记录称为完全离线安装。
没有使用 `--no-deps` 跳过依赖解析。两条安装命令及 `pip check` 均成功。

## 实际通过的检查

| 检查 | 结果 |
| --- | --- |
| 安装来源 | `direct_url.json` 指向上述本地 wheel，包含匹配的 SHA-256，无 editable 标记 |
| 模块来源 | tiny_lm、torch、numpy、regex、yaml 均来自新环境的 site-packages |
| 许可文件 | 项目 MIT、第三方声明、Stanford 与 OpenAI 许可均出现在已安装的 dist-info 中 |
| console 入口 | `tiny-lm.exe --help`、`--version`、`check-config` 全部成功 |
| module 入口 | `python -I -m tiny_lm --help`、`--version` 成功 |
| 模型分支 | 安装源码保留 GELU 分支；本次 smoke 模型仍使用 SwiGLU |
| tokenizer 与数据 | 256 字节词表创建成功，训练/验证文本分别编码并加载成功 |
| 训练 | 初次完成 2 步，生成 step_00000002.pt |
| 恢复 | 从第 2 步恢复到第 4 步，生成 step_00000004.pt |
| 独立验证 | 4 batches，128 个目标 token，loss=5.756909012794495 |
| 生成 | Hello 提示，seed=42，32 个新 token，以 length 结束 |

示例产物位于 `e2e-demo-wheel-v1/`；只执行了四步训练，没有追加正式语料训练。
配置保留 max_steps=4，初次用 `--stop-after 2`，恢复时不改配置；验证使用 float32、seed=42。
生成使用 temperature=0.8、top_k=20、top_p=1.0。四步字节模型的乱码输出不用于评价模型质量。

端到端的 8 个进程通过 `installed_cli.py` 调用已安装的 `tiny_lm.__main__`，
没有手动插入源码搜索路径；逐进程检查所有已加载 tiny_lm 模块路径，均在新环境内。
审计钩子阻止读取旧项目与旧复现环境，记录到的读取尝试为 0。
另有直接调用安装后 console 和 `python -m` 的检查，未用包装器替代入口验证。
完整验收驱动退出码为 0，15 条验收子命令均成功；两条安装命令另计。

## 与此前源码演示的区别

此前源码演示的 loss 为 5.6413843631744385，本次为 5.756909012794495。
核对发现两次输入字节不同：

| 输入 | 本次 | 此前材料包演示 |
| --- | --- | --- |
| toy_corpus.txt | 381 字节，LF | 387 字节，6 个 CRLF |
| validation.txt | 61 字节，1 个 CRLF | 60 字节，LF |

统一换行后两组文本分别相同，但默认预处理保留原始换行，因此 token 序列不同。
本次外部验收脚本写验证文本时使用 Windows 默认换行；没有改变项目预处理行为。
这些运行不是同一输入的精度对照，不把 loss 差异当作 wheel 数值回归或跨设备误差。
没有额外运行逐张量对照来证明差异的全部原因。

## 保存位置与后续范围

验收目录保留 `acceptance.py`、`run_check.py`、`installed_cli.py`、`probe.py`、
完整 `acceptance.json` 和 `logs/`。日志包含每条命令的参数、cwd、UTC 时间、退出码、耗时、
stdout/stderr，以及端到端进程的导入来源记录。
仓库只新增文档和[结果摘要](wheel-install-checks.json)，环境、训练产物和运行日志不加入 Git。

本次未改项目源码、依赖配置或 wheel 文件，未执行全量 pytest/Ruff、长期训练、Linux/GPU
或另一台设备上的 wheel 验收。下一步可整理首版安装说明、更新日志和 Release；
单独分发实验材料仍需相应许可核查。
