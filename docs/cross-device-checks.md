# 跨设备 CPU 复现验收（2026-10-28）

[文档索引](README.md) · [复现指南](reproduction.md) · [机器可读结果](cross-device-checks.json)

在 Intel N150 开发设备上，新建虚拟环境并安装材料包源码，完成四步端到端示例及已有
50k 实验 checkpoint 的生成、验证。13 条项目 CLI 命令均成功；没有启动长时间训练。
本次为 editable 源码安装验收，**不代表发布 wheel 安装验收通过**。

## 材料与隔离

- 收到的文件实际名为 `artifacts/reproduction-50k-v1..zip`，校验文件为
  `artifacts/reproduction-50k-v1.zip..sha256`；保留原文件名，未覆盖已有材料。
- ZIP 大小：176064894 字节。
- SHA-256：`6cf5922589e0aaad55c6e7fb65ab983f70798c377c63acaa72dc274177ff7bd5`，
  与旁置校验文件及材料包原记录一致。
- 新工作目录：`D:/Python/reproduction-acceptance-20261028`。
- 解压根目录：工作目录下的 `reproduction-50k-v1`。
- 独立环境：工作目录下的 `.venv`，使用现有 CPython 3.12.10 创建，未复制旧环境，未启用系统包。
- 原始命令、脚本、日志：工作目录下的 `verification`；临时文件使用独立 `temp`。

执行前后各运行一次包内 `verify_bundle.py`，两次均验证 338 个清单文件的大小和 SHA-256。
原清单文件未改变；安装生成的元信息和新演示产物不在原清单内，校验器不会拒绝这些新增文件。

通过新环境的 editable 安装解析 `tiny_lm`，没有手动注入源码搜索路径。
每个 CLI 进程均记录并检查所有已加载的 `tiny_lm` 模块路径，全部位于解压后的 `src`；
审计钩子阻止读取原项目目录，本次记录到的读取尝试为 0。
PyTorch、NumPy、regex、PyYAML 均来自新虚拟环境。

## 设备与安装

| 项目 | 本次实际采集 |
| --- | --- |
| CPU | Intel N150，4 核 / 4 逻辑处理器 |
| 内存 | 16912056320 字节，约 15.75 GiB |
| 系统 | Windows 11 专业版，10.0.26100，64 位 |
| 显卡 | Intel Graphics，驱动 32.0.101.6127；未用于计算 |
| Python | CPython 3.12.10，64 位 |
| 核心依赖 | torch 2.11.0+cpu、numpy 2.5.4、regex 2026.9.29、PyYAML 6.0.3 |
| 计算 | CPU / float32；torch CUDA 版本为空 |
| 线程 | OMP_NUM_THREADS=1、MKL_NUM_THREADS=1；实测 intra-op=1、interop=4 |

原训练设备记录为 Ryzen 9 7945HX、Windows 11 10.0.26300；Python 和上述核心依赖版本一致。
完整包版本与导入来源保存在机器可读结果和原始环境日志中。

安装过程仅写入新环境。以下为对应 PowerShell 命令；实际逐条执行参数、时间与退出码另存日志：

```powershell
$Work = 'D:/Python/reproduction-acceptance-20261028'
$Py = "$Work/.venv/Scripts/python.exe"
Set-Location "$Work/reproduction-50k-v1"
& $Py -I -X utf8 -m pip install --disable-pip-version-check --no-input torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
& $Py -I -X utf8 -m pip install --no-compile --disable-pip-version-check --no-input numpy==2.5.4 regex==2026.9.29 PyYAML==6.0.3 wheel
& $Py -I -X utf8 -m pip install --no-compile --no-deps --no-build-isolation --no-index --no-cache-dir --disable-pip-version-check -e .
& $Py -I -X utf8 -m pip check
& $Py -I -X utf8 -B ./verify_bundle.py
```

`pip check` 通过。pip 在 editable 安装时构建的临时 editable wheel，不等于独立发布 wheel 的安装验收。

## 四步端到端示例

按[端到端示例](end-to-end.md)流程，使用新目录 `artifacts/e2e-demo-v2`，保留旧产物。
配置使用 CPU、256 字节词表、SwiGLU、总计 4 步；训练和验证文本分开准备。

| 操作 | 实际结果 |
| --- | --- |
| 配置、tokenizer 与训练/验证预处理 | 全部成功 |
| 初次训练 | 完成 2 步，保存 step_00000002.pt |
| 恢复训练 | 从第 2 步继续到第 4 步，保存 step_00000004.pt |
| 验证 | 4 batches、128 个目标 token，loss=5.6413843631744385 |
| 生成 | 生成 32 个新 token，以 length 结束 |

四步字节级演示只用于验收流程，生成内容不作为模型质量结论。

## 已有 50k checkpoint 对照

加载 `outputs/tinystories-cpu-50k-v1/step_00015000.pt`，使用包内
`configs/tinystories-cpu-50k.yaml`。已有权重的 `ffn_type=swiglu` 保持不变，
当前源码中的 GELU 支持也保持不变；没有把已有模型转换为 GELU。
对照原报告为 `outputs/tinystories-evaluation-50k-v1/report.json`。

固定提示为 `Once upon a time, there was a little girl`，seed=42、temperature=0.8、
top_k=20、top_p=1.0、最多 128 个新 token。
本次与原报告的**完整文本及全部 token IDs 完全一致**，均生成 128 个新 token，以 length 结束。

验证使用 float32，每个 seed 32 batches，batch_size=8、context_length=128，
每组统计 32768 个目标 token。采样窗口可以重复，这不是完整验证集遍历。

| seed | 原 loss | 本次 loss | 本次减原值 |
| --- | --- | --- | --- |
| 1337 | 2.363883964717388 | 2.3638839572668076 | -7.450580596923828e-09 |
| 2026 | 2.3793849647045135 | 2.379384934902191 | -2.9802322387695312e-08 |
| 42 | 2.3204189836978912 | 2.3204189762473106 | -7.450580596923828e-09 |
| 均值 | 2.354562637706598 | 2.3545626228054366 | -1.4901161193847656e-08 |

指定生成复现一致，验证存在上述微小数值差异；不声称 loss 逐位一致，也未据此定位差异的具体原因。

## 命令、日志与范围

`verification/acceptance.json` 保存汇总、环境及逐条命令；仓库中的
[JSON 记录](cross-device-checks.json)保留这些结果与命令，便于审阅。
本机 `verification` 另保留：

- `archive-verification.json`、硬件与环境采集结果、安装日志。
- 每条命令的 `*.command.json`、`*.stdout.txt`、`*.stderr.txt`。
- 每个 CLI 的 `*.imports.json` 导入来源证明。
- `run_command.py`、`run_bundle_cli.py`、`setup_demo.py`、`environment.py`、`acceptance.py` 等验收脚本。
- `bundle-before` / `bundle-after` 的完整性检查日志。

首次外部验收脚本在解析 `check-config` 输出时，误把 JSON 前的成功提示一起传给 JSON 解析器，
因此以 JSONDecodeError 退出；当时端到端流程和配置 CLI 已成功。
修正外部脚本后复用参数一致且退出码为 0 的既有命令日志，继续执行剩余生成与验证，最终退出码为 0。
失败日志保留，没有为此重复训练，也没有修改项目 CLI 或模型源码。

本次未运行全量 pytest/Ruff、Linux/GPU 验收、正式发布 wheel 安装、50k 跨目录续训，
也未重新执行全部 checkpoint/提示词/独立留出集评估。
语料、权重、材料包、虚拟环境与原始运行产物不纳入 Git；后续优先做独立发布 wheel 安装验收及许可核查。
