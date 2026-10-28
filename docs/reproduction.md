# 跨设备复现指南

[项目首页](../README.md) · [文档索引](README.md) · [设备与环境清单](environments.md) · [实验索引](../experiments/README.md)

本指南说明如何把已有实验材料转移到另一台设备，运行生成、评估、从头训练或恢复训练。
命令以 Windows PowerShell 为主，末尾提供 Linux 环境写法。
以下为操作说明；实际验收范围按下列日期记录，训练来源见设备清单。
2026-10-26 已在原训练设备整理[材料包](material-bundle.md)，保留当前 GELU 源码与单独的历史源码快照；尚未完成跨设备运行。
2026-10-27 已完成[同机换目录生成与验证](relocation-checks.md)，复用已有依赖环境；另一台设备及干净安装仍未验收。
2026-10-28 已完成[另一台设备的独立环境 CPU 验收](cross-device-checks.md)：四步示例成功，固定生成一致，
三组验证均值差异为 -1.4901161193847656e-08。安装对象为材料包 editable 源码，发布 wheel 仍未验收。

## 1. 先确定复现目的

| 目的 | 必需材料 | 当前支持范围 |
| --- | --- | --- |
| 使用已有模型生成 | 对应源码、完整 YAML、原 tokenizer.json、选定 checkpoint | 可调整 tokenizer 路径和运行设备；模型配置须匹配 |
| 独立验证 loss | 上述材料、验证 bin 与旁置 JSON、评估参数 | 允许新的验证路径；不要求训练数据存在 |
| 复现整轮固定评估与生成样本 | 完整实验源码、配置、训练/验证数据及 JSON、tokenizer、全部候选 checkpoint、固定提示 | TinyStories 脚本固定使用 CPU FP32 和既定评估协议 |
| 从头重复训练 | 对应源码、配置、tokenizer、训练/验证数据及 JSON、运行环境记录 | 使用新 output_dir，从零开始；需另行安排训练预算 |
| 从中断位置恢复训练 | 完整训练材料、包含优化器与随机状态的 checkpoint | 完整配置含解析后的路径和设备必须匹配；跨目录/CPU→GPU 续训尚无便捷入口 |

生成只需模型与词表，不必复制全部语料。数据文件必须与元信息配对，例如 `val.bin` 和 `val.bin.json`。
同样大小的词表不代表同一个 tokenizer；不得重新训练一个 2048 词表来替代原文件。

## 2. 原设备应保存什么

以 [5 万篇、15000 步实验](../experiments/tinystories-50k.md)为例，完整复现材料建议保持以下相对结构：

```text
configs/tinystories-cpu-50k.yaml
experiments/tinystories-prompts.json
data/tinystories-expanded-50k-v1/
  tokenizer.json
  train.bin / train.bin.json
  val.bin / val.bin.json
  heldout.bin / heldout.bin.json
  manifest.json
outputs/tinystories-cpu-50k-v1/
  step_00002500.pt ... step_00015000.pt
  config.source.yaml / config.resolved.json
  run.json / working-tree.diff
  train.json / train.stderr.jsonl
outputs/tinystories-evaluation-50k-v1/
  report.json / metrics.csv / samples.md / heldout.json
```

另保留原始文本分片、下载缓存与来源清单，便于将来复核数据准备；仅使用已经编码的数据时不要求重新下载。
跨设备比较旧 6133 篇模型时，还需旧模型的配置、tokenizer、训练/验证数据及候选 checkpoint。
目录和文件存在性以原设备为准；如果缺少某项，记录缺失，不补造运行历史。

### 源码与运行记录

Git 拉取只同步已提交的源码和说明，被忽略的权重、数据和运行记录需要单独传输。
不要复制 `.venv`；新设备重新创建环境。

- 保存训练时的 Git commit、配置快照、完整命令、依赖版本、硬件与线程设置。
- 本轮报告记录的训练 HEAD 是 `8516dccdcb95a7c739ed23e0bcce86d02a6ea7d8`，当时另有未提交修改。
  仅 checkout 这个 commit 不足以重建完整运行源码。
- 保存对应 `working-tree.diff`，以及当时未跟踪的新脚本/文件。普通 Git diff 不包含未跟踪文件。
  如需还原历史源码，在独立目录应用已核对的补丁与新增文件，再核对 `run.json` 中的源码哈希。
- 当前分支可能已包含这些修改；应通过源码文件哈希确认，不能仅凭文件名或版本号判断一致。
  若历史文件无法找回，只能标为基于当前源码的复跑，不能声称精确重建了原实验。
- 从原设备现在导出的包列表只代表导出时环境，优先使用训练当时的环境记录。

在原设备创建一个**尚不存在**的材料目录，并按上表复制所需文件，保留相对目录结构。
例如使用 `artifacts/reproduction-50k-v1/`，通过移动硬盘或其他文件传输方式带到新设备。
这个目录已被 Git 忽略；不要为了传输权重而使用 `git add -f`。

下面只记录导出时环境，不启动模型。先将 `$ProjectPython` 改为原设备实际使用的解释器：

```powershell
$ProjectPython="D:/Python/.venvs/tiny-language-model/Scripts/python.exe"
$BundleRoot="artifacts/reproduction-50k-v1"
New-Item -ItemType Directory -Path $BundleRoot -ErrorAction Stop | Out-Null
& $ProjectPython --version
& $ProjectPython -m pip freeze --all | Out-File -Encoding utf8 "$BundleRoot/environment-at-export.txt"
git rev-parse HEAD | Out-File -Encoding utf8 "$BundleRoot/source-head-at-export.txt"
git status --short | Out-File -Encoding utf8 "$BundleRoot/source-status-at-export.txt"
```

随后复制材料进该目录，再生成下节的校验清单。导出时的 Git 状态文件并不是源码备份。
不在材料包中放账户凭据、访问令牌或无关私人文件。

## 3. 传输前后核对文件

以下 PowerShell 示例调用项目 Python，使用 `pathlib` 与标准库 SHA-256。
先在原设备完成材料复制，再运行；清单不包含自身，已有清单会报错。

```powershell
@'
from pathlib import Path
import hashlib
import json
root=Path("artifacts/reproduction-50k-v1").resolve()
if not root.is_dir():
    raise FileNotFoundError("Bundle directory does not exist")
manifest=root/"transfer-manifest.json"
if manifest.exists():
    raise FileExistsError("Transfer manifest already exists")
entries=[]
for path in sorted(root.rglob("*")):
    if path.is_file():
        with path.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        entries.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":digest})
if not entries:
    raise ValueError("Bundle is empty")
with manifest.open("x",encoding="utf-8") as stream:
    json.dump(entries,stream,ensure_ascii=True,indent=2)
print(f"Recorded {len(entries)} files")
'@ | & $ProjectPython -
```

传输期间不要修改材料。在新设备将完整材料包放入相同的项目相对位置，创建好虚拟环境后运行：

```powershell
@'
from pathlib import Path
import hashlib
import json
root=Path("artifacts/reproduction-50k-v1").resolve()
entries=json.loads((root/"transfer-manifest.json").read_text(encoding="utf-8"))
if not entries:
    raise ValueError("Transfer manifest is empty")
for item in entries:
    path=(root/item["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid file: {item['path']}")
    with path.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    if path.stat().st_size!=item["bytes"] or digest!=item["sha256"]:
        raise ValueError(f"File checksum mismatch: {item['path']}")
print(f"Verified {len(entries)} files")
'@ | & $ProjectPython -
```

这检查清单中的文件是否完整传输，不证明包内源码就是训练当时的源码，也不代替模型输入绑定检查。
校验后将所需数据与权重复制到工作目录中对应位置，或在新的本地配置中指向材料包；保留原始材料包用于核对。
目标位置已有文件时先比较哈希，避免覆盖其他实验。

## 4. 新设备建立环境和配置

已有仓库先按正常 Git 流程同步；要还原历史源码时另建目录，避免改动当前工作区。
新设备使用 Python 3.11+；复现本轮 CPU 实验优先与报告的 Python 3.12.10、PyTorch 2.11.0+cpu 对齐。

```powershell
Set-Location D:\Python\tiny-language-model
python --version
python -m venv .venv
$ProjectPython=".\.venv\Scripts\python.exe"
```

已有虚拟环境时跳过创建步骤。按 [依赖说明](dependencies.md)安装对应 CPU PyTorch 构建，
GPU 使用实际硬件适配的构建，不照搬另一台设备的 CUDA 环境。
随后安装项目功能依赖；原训练环境的精确依赖记录可用时，应逐项对齐，不能把下限约束视作锁定版本：

```powershell
& $ProjectPython -m pip install -e ".[dev,compute,tokenizer]"
& $ProjectPython -m pip check
& $ProjectPython -m tiny_lm --version
```

完整环境记录与缺失项见 [设备清单](environments.md)。复制整个旧 `.venv` 不能保证解释器路径和二进制依赖可用。

### 生成与评估使用本地配置副本

下例假设材料已按第 2 节放入项目目录。副本仍放在 `configs/`，相对路径解析基准保持一致：

```powershell
$Config="configs/reproduction-50k.local.yaml"
if (Test-Path -LiteralPath $Config) { throw "Local config already exists; choose a new name" }
Copy-Item -LiteralPath configs/tinystories-cpu-50k.yaml -Destination $Config -ErrorAction Stop
& $ProjectPython -m tiny_lm check-config --config $Config
```

按实际位置修改副本中的 data 路径和 device。模型字段须匹配 checkpoint，词表必须使用原文件。
本地配置的 `.local.yaml` 后缀已被 Git 忽略；原实验配置与快照保留不改。
检查规范化后的路径；`check-config` 不检查数据存在性、内容或硬件，成功不代表已经可以训练。
YAML 中的路径相对于 YAML 所在目录；命令参数中的路径相对于当前命令目录。

## 5. 使用已有权重生成

只需 YAML、原 tokenizer 和选定 checkpoint；训练/验证文件及 output_dir 可以不存在。
下例复现固定提示集中的第一条提示，生成精度自动从 checkpoint 读取：

```powershell
$Checkpoint="outputs/tinystories-cpu-50k-v1/step_00015000.pt"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
& $ProjectPython -m tiny_lm generate --config $Config --checkpoint $Checkpoint --prompt "Once upon a time, there was a little girl" --max-new-tokens 128 --temperature 0.8 --top-k 20 --top-p 1 --seed 42
```

输出 JSON 包含完整文本、token IDs、新增 token 数与结束原因。保存结果时使用新的日志文件名。
相同种子不保证不同硬件/软件下逐字一致；先核对全部采样参数、tokenizer 与权重哈希。
该命令只覆盖一条提示，不能代替整轮生成评估。

## 6. 复现固定评估

### 单 checkpoint 的独立验证

保持配置中 batch_size=8、context_length=128，验证数据使用原 val.bin 与元信息。
每个 seed 32 batches；三次均成功后，对三个 loss 取算术平均，才对应报告的固定验证均值：

```powershell
foreach ($EvalSeed in @(1337,2026,42)) {
    & $ProjectPython -m tiny_lm evaluate --config $Config --checkpoint $Checkpoint --dtype float32 --eval-batches 32 --seed $EvalSeed
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
}
```

三组共 98,304 个目标 token，窗口有放回且可能重叠，不代表完整遍历验证集。
若要独立评估 heldout，另复制配置，将 val_data_path 指向 heldout.bin；保留原验证配置。
不得根据 heldout 结果反向选 checkpoint 或反复调整采样设置。

### 六个保存点与全部十条提示

这一步另需原训练数据、原验证数据与六份 checkpoint。
本地配置的 output.output_dir 必须指向已有 checkpoint 目录；脚本的 `--output-dir` 才是新评估结果目录。
脚本会核对历史数据绑定，固定使用 CPU FP32、训练时 eval_batches=4 的绑定，以及报告的评估/生成参数。
它不是适用于任意模型和任意训练设置的通用评估器。

```powershell
& $ProjectPython scripts/evaluate_tinystories.py --config $Config --prompts experiments/tinystories-prompts.json --steps 2500 5000 7500 10000 12500 15000 --output-dir outputs/reproduction-50k-evaluation-v1
```

输出目录必须不存在；成功产生 report.json、metrics.csv 与 samples.md。
与 [原报告](../experiments/tinystories-50k.md)和[指标 CSV](../experiments/tinystories-50k-metrics.csv)逐项比较，
记录运行环境、实际 loss 差值和生成 token 是否一致，不预设未经验证的误差阈值。

### 新旧模型的独立留出对照

留出脚本会检查传入 report.json 内 `source_sha256` 的所有文件，并通过其中的绝对路径定位权重。
**换目录后不能直接复用另一台设备的原 report.json，也不要手改其中的路径或哈希来伪装原报告。**
先在新设备为旧小语料模型和新 5 万篇模型各生成一次固定评估报告，
使用各自对应的训练/验证数据与配置，再运行：

```powershell
& $ProjectPython scripts/evaluate_tinystories_heldout.py --reports outputs/reproduction-small-evaluation-v1/report.json outputs/reproduction-50k-evaluation-v1/report.json --heldout data/tinystories-expanded-50k-v1/heldout.bin --tokenizer data/tinystories-expanded-50k-v1/tokenizer.json --output outputs/reproduction-50k-evaluation-v1/heldout.json
```

旧模型报告用上一节同一脚本生成，将 config 换为旧 `tinystories-cpu-15000.yaml` 的本地副本，
将结果目录换为 `outputs/reproduction-small-evaluation-v1`，steps 保持相同。
报告生成后不再修改其引用的配置、提示、数据或权重；输出 heldout.json 必须不存在，父目录须存在。
这属于复核历史对照，不得把重复查看留出结果用于调参；原报告和新报告都保留。

## 7. 从头重复训练与中断恢复

### 从头重复训练

本节会真正启动训练，仅在已经决定重跑该实验时执行。
另建训练配置副本，保留原模型、数据、训练参数和线程设置，将 output_dir 改为新的独立目录。
复现原 5 万篇实验时 eval_interval 仍为 500；未来改为 2500 是新实验设置，不能回改历史协议。

例如把副本命名为 `configs/retrain-50k.local.yaml`，并设置：

```yaml
output:
  output_dir: ../outputs/retrain-50k-v1
```

其余字段保留，数据路径按实际位置配置。确认目录尚不存在，再执行：

```powershell
& $ProjectPython -m tiny_lm check-config --config configs/retrain-50k.local.yaml
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
& $ProjectPython -m tiny_lm train --config configs/retrain-50k.local.yaml --dtype float32 --eval-batches 4
```

不传 `--resume`，从零开始。CLI 会创建训练输出目录；这条命令没有自动的 1800 秒终止机制。
原实验的限时来自外层进程控制，若需要相同时间预算，须另行安排监控和停止方式。
保存本次源码状态、环境、命令、配置与日志，使用新实验名称，不覆盖原结果。

### 恢复尚未结束的训练

当前 CLI 严格比较完整 ProjectConfig，包含解析后的绝对数据路径、tokenizer 路径、output_dir 和 device，
并核对实际数据内容、tokenizer、dtype、eval_batches 等。文件内容一致也不能自动放行路径变化。

- 同目录布局的另一台 Windows 设备：满足全部配置与输入约束后可以尝试恢复，但不同硬件仍不保证逐位一致。
- 换盘符、改 output_dir、Windows→Linux 或 CPU→CUDA：当前没有路径映射或设备迁移的续训命令；
  需要单独设计并验证兼容规则，不能在指南中承诺直接恢复。
- `--allow-unbound-checkpoint` 只兼容历史缺失绑定，不允许绕过已知配置、路径或内容不匹配。
- 保存步数等于 max_steps 时恢复不会再更新；15000 步终点不是可直接追加任意步数的入口。
  修改 max_steps 会改变计划并触发配置不匹配。

只有已核对上述条件、且另一个设备尚无同名后续产物时，才使用中间 checkpoint，例如：

```powershell
& $ProjectPython -m tiny_lm train --config configs/tinystories-cpu-50k.yaml --resume outputs/tinystories-cpu-50k-v1/step_00012500.pt --dtype float32 --eval-batches 4
```

这会继续到原定 15000 步，不是再训练 15000 步。同名后续 checkpoint 可能被替换，
不要在已经保存完整历史结果的训练目录里重放；原始备份应放在独立材料目录。
仅为使用模型或复核 loss 时，选择前面的生成/评估入口即可。

## 8. Linux 环境写法

新建独立环境，先按目标平台准备 PyTorch，再安装项目。以项目根目录为当前目录：

```bash
python3.12 -m venv .venv
PYTHON=.venv/bin/python
"$PYTHON" -m pip install -e ".[dev,compute,tokenizer]"
"$PYTHON" -m pip check
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PYTHON" -m tiny_lm check-config --config configs/reproduction-50k.local.yaml
```

相应功能命令把 PowerShell 的 `& $ProjectPython` 换为 `"$PYTHON"`，参数保持不变。
第 3 节两个 Python 校验片段也可保存为临时 `.py` 文件，用这个解释器执行。
这里提供可移植的路径与命令写法，不代表 Linux/GPU 已验收；Windows 的绝对路径不适用于 Linux。
GPU 可以单独评估加载能力和数值表现，TinyStories 固定评估脚本仍明确使用 CPU。

## 9. 如何记录复现结论

| 检查层次 | 应记录的事实 |
| --- | --- |
| 材料完整性 | 源码版本与未提交文件是否还原，传输文件数、哈希是否匹配，缺少哪些材料 |
| 环境 | OS、硬件、Python、依赖、设备、dtype、线程；与原实验的差异 |
| 功能运行 | 执行了哪些命令、退出码、输出位置；未执行的项目明确标注 |
| 固定评估 | 相同窗口协议下逐 seed loss、均值与差值，不能与训练日志的随机窗口 loss 混比 |
| 生成 | 固定提示与采样设置下 token IDs、结束原因、文本；不能因不逐字相同就直接判为代码错误 |
| 恢复 | checkpoint 步数、实际后续步数、配置与随机状态约束；逐位恢复需专门对照实验 |

文件完全一致、流程成功运行、指标接近和数值逐位一致是不同结论，分别报告。
具体数值差异需要结合环境及运算路径调查，当前没有经过验证的跨设备容差标准。

常见问题：

- 文件不存在：Git 不包含运行产物；核对传输清单与配置解析后的绝对路径。
- tokenizer/数据指纹不匹配：确认原产物与配对 JSON，不通过替换元信息或开启兼容选项绕过。
- 续训配置不匹配：先检查解析后的路径、设备、max_steps、验证间隔等完整配置，保留报错。
- 留出脚本仍访问旧盘符：先在当前路径重新生成固定评估报告。
- loss 或生成不同：核对源码与权重哈希、依赖版本、线程、dtype、数据、随机种子和采样参数，记录实际差异。

本指南整理于 2026-10-26，仅核对现有接口、文档链接和命令示例；没有传输真实训练产物，
也没有执行跨设备训练、生成或数值复现。完成上述步骤后，再另写实际验收记录。
