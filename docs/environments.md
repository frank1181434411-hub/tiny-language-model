# 设备与环境清单

[文档索引](README.md) · [实验索引](../experiments/README.md)

材料传输、校验与新设备运行步骤见[跨设备复现指南](reproduction.md)。

本清单区分当前开发设备的采集结果与另一台训练设备的历史记录。
登记环境不代表已在该设备完成模型测试、训练或跨设备复现。

## 当前开发设备

2026-10-26 从项目虚拟环境采集。源码 HEAD 为
`be0fb67e2742eec9680a70b90c53beac108d2408`；这是登记时的版本，不是此前训练使用的版本。

| 项目 | 当前采集结果 |
| --- | --- |
| 系统 | Windows 11，版本 10.0.26100 |
| Python | CPython 3.12.10，64 位 AMD64 |
| 项目解释器 | `D:/Python/tiny-language-model/.venv/Scripts/python.exe` |
| PyTorch | 2.11.0+cpu |
| NumPy | 2.5.4 |
| regex | 2026.9.29 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| Ruff | 0.17.1 |
| OMP_NUM_THREADS / MKL_NUM_THREADS | 本次采集进程均未设置；不据此推断 PyTorch 实际线程数 |
| CPU 型号、物理核心数、内存容量 | 待核实 |
| GPU 型号与驱动 | 待核实；安装 CPU 版 PyTorch 不代表机器没有 GPU |

软件版本来自 `importlib.metadata` 的已安装包元信息，系统版本来自 Python `platform`。
读取硬件的 PowerShell `Get-CimInstance` 查询返回“拒绝访问”（0x80041003），
因此没有将受限进程中的逻辑处理器数量当作整机规格，也没有猜测 CPU、内存或 GPU。
本次只采集信息，未导入模型、运行测试或训练。

## 另一台训练设备

以下信息转录自 [5 万篇、15000 步实验报告](../experiments/tinystories-50k.md)，
本次没有访问原设备或核对其运行产物。当前开发设备的软件版本不能用于补齐这张表。

| 项目 | 训练报告记录 |
| --- | --- |
| 实验日期 | 2026-10-25 |
| CPU | AMD Ryzen 9 7945HX |
| 系统 | Windows 11；具体构建号未记录 |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cpu |
| 计算设备与精度 | CPU、FP32；本轮没有 GPU 参与 |
| 线程设置 | OMP_NUM_THREADS=1、MKL_NUM_THREADS=1 |
| 训练规模 | 50000 篇故事、15000 步、15,360,000 个训练目标 token |
| 训练耗时 | 1491.07 秒；包含启动、数据校验、训练内验证与 checkpoint 保存 |
| 运行时源码 HEAD | `8516dccdcb95a7c739ed23e0bcce86d02a6ea7d8`，另有未提交修改 |
| NumPy、regex、PyYAML 等完整依赖版本 | 待从原设备的运行记录核对 |
| 内存容量、GPU 型号与驱动 | 报告未记录；GPU 未参与本轮 CPU 实验 |
| 峰值内存、温度、功耗 | 本轮未测量，不能事后补写为实测值 |

## 回到训练设备后补齐的材料

优先保存训练当时的记录，再补充现在的机器信息：

1. 从 `outputs/tinystories-cpu-50k-v1/` 保存 `run.json`、`working-tree.diff`、
   `config.source.yaml`、`config.resolved.json`、`train.json` 和 `train.stderr.jsonl`。
   报告记录这些文件已生成；Git 拉取不会自动带回它们。
2. 保存选定 checkpoint、匹配 tokenizer、训练/验证数据与旁置 JSON，
   以及数据 manifest 和评估输出；核对传输前后的 SHA-256。
3. 检查 `run.json` 是否已保存完整依赖与环境信息；没有记录的项目继续标为未知。
   现在导出的依赖列表只能证明导出时状态，不能证明训练后环境从未改变。
4. 补充 CPU、内存、系统构建号；若以后开展 GPU 实验，再记录 GPU、驱动及实际使用的 PyTorch/CUDA 构建。

可在原设备的普通 PowerShell 中读取硬件信息：

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
```

这些命令只读取信息；若查询被限制，记录具体缺项即可。
不采集设备序列号、机器名、账户或网络信息。

## 如何使用这份清单

本清单用于识别设备和环境差异，不等于环境锁定文件或复现验收结果。
实际复现还需对应源码（含训练时未提交修改）、配置、数据、tokenizer 与权重。
严格续训会比较完整配置中的路径和设备；相同文件内容不代表换目录后可直接恢复，
见 [checkpoint 输入绑定](checkpoint-binding.md)。跨设备逐位一致尚未验证。
