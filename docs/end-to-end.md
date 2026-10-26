# 小型端到端示例

[项目首页](../README.md) · [文档索引](README.md) · [跨设备复现](reproduction.md)

本例把已有 CLI 串成一条完整流程：准备文本 → 创建 tokenizer → 编码数据 → 训练两步 →
从 checkpoint 恢复到四步 → 独立验证 → 文本生成。
使用仓库自带语料、CPU FP32 和极小模型，不下载语料或模型；执行第 5、6 节会真实更新模型参数。
四步训练只用于检查流程，不应期待通顺的生成结果或据此评价泛化能力。

下面各节请在同一个 PowerShell 窗口中按顺序执行，任一步失败先停止排查。
所有产物放入新的 `artifacts/e2e-demo-v1/`，不使用已有 TinyStories 数据和权重。

## 1. 准备项目环境

需要 Python 3.11+、PyYAML、PyTorch、NumPy 和 regex。已有项目环境时直接设置解释器路径：

```powershell
Set-Location D:\Python\tiny-language-model
$ProjectPython=".\.venv\Scripts\python.exe"
& $ProjectPython --version
if ($LASTEXITCODE -ne 0) { throw "Project Python is unavailable" }
```

尚无环境时按 [README 安装步骤](../README.md)创建 `.venv`，
并按 [依赖说明](dependencies.md)安装对应平台的 PyTorch；本例选择 CPU 构建即可。
随后安装项目依赖；已有完整依赖时可跳过安装命令：

```powershell
& $ProjectPython -m pip install -e ".[dev,compute,tokenizer]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $ProjectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
```

以下线程设置只影响当前 PowerShell 进程及随后启动的子进程，不修改持久系统设置：

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$DemoRoot="artifacts/e2e-demo-v1"
$DemoConfig="$DemoRoot/config.yaml"
$Tokenizer="$DemoRoot/tokenizer.json"
$TrainData="$DemoRoot/train.bin"
$ValidationText="$DemoRoot/validation.txt"
$ValidationData="$DemoRoot/val.bin"
$Checkpoint2="$DemoRoot/checkpoints/step_00000002.pt"
$Checkpoint4="$DemoRoot/checkpoints/step_00000004.pt"
```

## 2. 创建独立配置与验证文本

以下片段以 [smoke.yaml](../configs/smoke.yaml)为模板，只调整数据和输出路径，
保留 CPU、256 词表、上下文 16、单层 32 维、batch=2、max_steps=4、每两步验证/保存的设置。
模板中预热为一步，第一次更新的学习率为零；四步是训练流程的更新计数。

脚本使用 `pathlib` 创建目录；目标已存在会报错，不覆盖旧演示结果。
重新开始一轮时，将上一节的 `$DemoRoot` 改为 `artifacts/e2e-demo-v2`，重新设置全部派生路径。

```powershell
@'
from pathlib import Path
import sys
import yaml
root=Path(sys.argv[1])
config=yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
config["data"]={"train_data_path":"train.bin","val_data_path":"val.bin","tokenizer_path":"tokenizer.json"}
config["output"]={"output_dir":"checkpoints"}
root.mkdir(parents=True,exist_ok=False)
with (root/"config.yaml").open("x",encoding="utf-8",newline="\n") as stream:
    yaml.safe_dump(config,stream,sort_keys=False)
with (root/"validation.txt").open("x",encoding="utf-8",newline="\n") as stream:
    stream.write("A small validation text. The model predicts the next token.\n")
print(f"Created demo files in {root.resolve()}")
'@ | & $ProjectPython - $DemoRoot
if ($LASTEXITCODE -ne 0) { throw "Demo setup failed" }
& $ProjectPython -m tiny_lm check-config --config $DemoConfig
if ($LASTEXITCODE -ne 0) { throw "Config validation failed" }
```

这里的相对路径以新 config.yaml 所在目录为基准，因此 train.bin、val.bin、tokenizer.json
和 checkpoints 都位于 `$DemoRoot` 下。
`check-config` 此时成功只说明配置合法；尚未创建 tokenizer 和编码数据。
验证文本独立编写，仅用于运行流程，不能视为有统计意义的验证集。

## 3. 创建匹配的 tokenizer

```powershell
& $ProjectPython -m tiny_lm train-tokenizer --input examples/toy_corpus.txt --vocab-size 256 --output $Tokenizer
if ($LASTEXITCODE -ne 0) { throw "Tokenizer creation failed" }
```

成功摘要应显示 `vocab_size=256`。本例使用完整的 256 字节基础词表，不注册特殊 token，
因此没有 BPE 合并，也没有 EOS；这是与现有 smoke 配置匹配的最小 tokenizer 示例。
它覆盖 tokenizer 创建/保存/加载入口，不检验学习合并规则的效果。
不要替换成 README 中 300 词表的预处理产物，否则模型配置将不匹配。

## 4. 分别编码训练与验证文本

```powershell
& $ProjectPython -m tiny_lm prepare-data --input examples/toy_corpus.txt --tokenizer $Tokenizer --output $TrainData
if ($LASTEXITCODE -ne 0) { throw "Training data preparation failed" }
& $ProjectPython -m tiny_lm prepare-data --input $ValidationText --tokenizer $Tokenizer --output $ValidationData
if ($LASTEXITCODE -ne 0) { throw "Validation data preparation failed" }
```

两个文本使用同一 tokenizer，各自产生 bin 文件与旁置 JSON，记录 dtype、token 数、文件哈希和词表指纹。
预处理默认保留原始换行、不追加 EOS。两份文本编码后都应至少有 context_length+1，即 17 个 token；
本例使用的文本满足该长度要求，实际加载仍会校验。

## 5. 训练到第二步并保存

```powershell
& $ProjectPython -m tiny_lm train --config $DemoConfig --stop-after 2 --dtype float32 --eval-batches 4
if ($LASTEXITCODE -ne 0) { throw "Initial training failed" }
if (-not (Test-Path -LiteralPath $Checkpoint2)) { throw "Step 2 checkpoint is missing" }
```

配置的 max_steps 仍为 4；`--stop-after 2` 让本次进程在完成第二步后正常结束。
学习率计划仍按四步安排，这样才能在下一节延续同一训练计划。
stdout 的 JSON 摘要应包含 `completed_updates=2`，训练指标写入 stderr；两者同时显示在终端是正常行为。

本节演示的是主动停止后的恢复准备，不模拟突然断电或进程崩溃。

## 6. 恢复到原定第四步

保持配置、tokenizer、两份数据、dtype 和 eval_batches 不变，运行：

```powershell
& $ProjectPython -m tiny_lm train --config $DemoConfig --resume $Checkpoint2 --dtype float32 --eval-batches 4
if ($LASTEXITCODE -ne 0) { throw "Training resume failed" }
if (-not (Test-Path -LiteralPath $Checkpoint4)) { throw "Step 4 checkpoint is missing" }
```

恢复模型、优化器和随机状态后，继续完成第三、第四步。摘要应显示 `completed_updates=4`。
这不是额外再训练四步，也不需要把 max_steps 改成 6。
不要为了续训另改 output_dir；严格恢复会核对完整配置与历史输入。

本例只检查恢复入口与步数衔接，不与另一条连续训练轨迹逐张量对照，
所以不能仅凭成功运行就声称已经验证逐位复现。

## 7. 独立验证

```powershell
& $ProjectPython -m tiny_lm evaluate --config $DemoConfig --checkpoint $Checkpoint4 --dtype float32 --eval-batches 4 --seed 42
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
```

成功输出有限的 loss、`completed_updates=4`、`num_batches=4`，
以及 `num_tokens=128`（4 batches×2 条序列×16 个预测位置）。
该指标来自有放回随机窗口，不能把 128 次目标预测理解成 128 个互不重复的语料 token。
独立验证的采样状态与训练期间不同，不要求 loss 与训练日志中的 validation_loss 完全相同。
本示例不预设具体 loss 数值，也不要求四步内单调下降。

## 8. 从 checkpoint 生成文本

```powershell
& $ProjectPython -m tiny_lm generate --config $DemoConfig --checkpoint $Checkpoint4 --prompt "Hello" --max-new-tokens 32 --temperature 0.8 --top-k 20 --top-p 1 --seed 42
if ($LASTEXITCODE -ne 0) { throw "Text generation failed" }
```

输出 JSON 包含 text、token_ids、num_prompt_tokens、num_generated_tokens 和 finish_reason。
本例未注册 EOS，因此正常完成时新增 32 个 token，finish_reason 为 `length`。
字节级小模型可能生成控制字符、乱码或无效 UTF-8 的替换字符，这不代表已经学会语言。
判断流程是否成功应查看退出码与输出结构，不把生成是否通顺当作本例验收标准。

## 9. 产物与检查范围

```text
artifacts/e2e-demo-v1/
  config.yaml
  validation.txt
  tokenizer.json
  train.bin
  train.bin.json
  val.bin
  val.bin.json
  checkpoints/
    step_00000002.pt
    step_00000004.pt
```

这些文件由 Git 忽略。训练、验证与生成摘要默认输出到终端，不会自动保存为日志文件。
需要记录实际运行时，可以使用新日志文件名保存输出；重复使用重定向文件名可能截断旧记录。

完成后可以记录以下事实：

- tokenizer 与两份数据成功创建、加载，词表和数据元信息匹配。
- 两个训练进程分别停在第二步和第四步，对应 checkpoint 存在。
- 独立验证返回有限 loss 与预期目标 token 数。
- 生成返回完整 JSON、新增 token 数与预期长度结束原因。

这不包括正式语料效果、跨设备数值一致、GPU 性能或故障恢复可靠性验收。
换目录继续使用已有产物前，阅读[跨设备复现指南](reproduction.md)中的生成、评估和严格续训区别。

本示例于 2026-10-26 根据已有文档、smoke 配置与 CLI 参数整理。
本次仅检查文档链接和命令片段语法，没有实际执行上述预处理、四步训练、验证或生成；
后续运行结果应单独记录，不将已有其他测试的通过状态算作本示例实测。
