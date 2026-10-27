# 材料包换目录复现验收

2026-10-27 将已整理的 ZIP 解压到新目录，在同一台 Windows 笔记本上验证生成与独立评价。
没有重新训练，没有安装或修改依赖，没有改动原材料包及旧实验数据。

## 环境与隔离方式

- ZIP：`artifacts/reproduction-50k-v1.zip`，SHA-256 为 `6cf5922589e0aaad55c6e7fb65ab983f70798c377c63acaa72dc274177ff7bd5`。
- 新目录：`D:/Python/reproduction-check-20261027/reproduction-50k-v1/`。
- 日志目录：`D:/Python/reproduction-check-20261027/verification/`，位于材料包之外，避免改变包内清单文件。
- 复用 `D:/Python/.venvs/tiny-language-model/Scripts/python.exe` 的第三方依赖，CPython 3.12.10、PyTorch 2.11.0+cpu。
- Windows 11 10.0.26300；NumPy 2.5.4、regex 2026.9.29、PyYAML 6.0.3；CPU FP32、OMP/MKL 线程数均为 1。

使用 `python -I -B` 启动日志目录中的 run_bundle_cli.py，移除原项目的 sys.path 项，
将解压包的 src/ 放在导入路径首位，再运行 `tiny_lm` CLI。
运行时审计钩子拒绝打开原项目目录中的文件；每条命令结束后记录并断言所有已导入 tiny_lm 模块均来自包内 src/。
五条命令均未触发原目录读取拦截；配置解析后的数据、词表与输出路径均位于解压包中。

主代码保留已接通的 GELU 分支，没有启用历史快照目录。被加载的现有权重仍是 SwiGLU，未修改 ffn_type。
本次复用已有依赖环境，并非干净安装或另一台设备上的测试。

## 执行与比较结果

执行配置检查、一次生成、三个 seed 的独立验证，共五条命令，全部退出码为 0。
命令合计墙钟耗时 **29.11 秒**，包含每个进程启动；不含解压、哈希校验和文档整理。

使用包内 `configs/tinystories-cpu-50k.yaml` 和 `outputs/tinystories-cpu-50k-v1/step_00015000.pt`。
验证配置 batch=8、context=128，每 seed 32 batches，合计 98,304 个目标 token，与原固定评价一致。

| seed | 原验证 loss | 新目录验证 loss | 差值 |
| --- | ---: | ---: | ---: |
| 1337 | 2.363883964717388 | 2.363883964717388 | 0 |
| 2026 | 2.3793849647045135 | 2.3793849647045135 | 0 |
| 42 | 2.3204189836978912 | 2.3204189836978912 | 0 |
| 均值 | 2.354562637706598 | 2.354562637706598 | 0 |

生成采用原 girl 提示 `Once upon a time, there was a little girl`，seed=42、temperature=0.8、top-k=20、top-p=1，最多新增 128 tokens。
新增 128 tokens，以 length 结束；完整文本与全部 token IDs 均与包内原评估报告第 15000 步 girl 样本完全相同。
这里只核对一条固定提示，不代表重新评价了六个 checkpoint 或全部 60 条样本，也未再次查看留出集来调参。

ZIP 哈希在解压前与旁置清单一致。解压后和执行后两次运行 verify_bundle.py，**338 个文件的大小与 SHA-256 全部通过**。
包内配置、数据、词表、权重、源码及历史报告未被覆盖，未生成包内 Python 字节码缓存。
机器可读结果见 [JSON](relocation-checks.json)。

## 原始记录与编码说明

verification/ 内保存 archive-verification.json、bundle-verification.txt、run.json、verification.json、
实际 CLI 启动器 run_bundle_cli.py，以及五条命令各自的 stdout、stderr 和模块来源 imports.json。
原始记录位于 Git 仓库之外；本说明与精简结果 JSON 纳入版本控制。

隔离模式 `-I` 会忽略 PYTHONIOENCODING 环境变量，因此配置检查的中文 stdout 实际为 Windows 默认 GBK。
第一次日志核对按 UTF-8 读取该文件时报解码错误，随后按 GB18030 正确读取；CLI 本身一直成功，无需重跑。
其余生成和验证输出均为 ASCII 可解析的 JSON。以后需要强制 UTF-8 的隔离启动可显式使用 `-X utf8`。

## 结论与范围

材料包在本机换目录后能够独立定位项目源码和模型材料，所检查的生成与验证结果复现一致。
当前 GELU 代码可加载此前 SwiGLU 权重完成这些操作；这不是新的 GELU 效果实验。
没有验证新设备、不同依赖版本、GPU、干净安装、发布包或移动目录后的严格续训。
下一步可做干净环境安装与构建包验收，不需要追加长时间训练。
