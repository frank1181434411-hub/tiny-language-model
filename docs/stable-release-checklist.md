# 首个正式版范围与验收清单

2026-11-02 根据当前源码、格式文档及 alpha.2 验收记录整理。
[文档索引](README.md) · [路线图](roadmap.md) · [alpha.2 发布记录](release-alpha2.md)

本清单定义下一阶段的发布范围，不表示正式版已经通过验收。2026-11-03 源码版本更新为 `0.1.1`（未发布）；
配置整理见 [C 阶段发布准备](release-0.1.1.md)，最终候选回归与最终 wheel 已通过[验收](final-0.1.1-checks.md)，正式发布仍待完成。

## 1. 首版支持范围

正式版定位为个人学习与实验工具，交付可安装、可运行、可保存恢复的小型语言模型流程。
不以生成故事的连贯性或达到新的 loss 为发布门槛，也不要求追加长时间训练。

| 项目 | 本阶段承诺范围 |
| --- | --- |
| 操作系统 | Windows 11 x64 |
| Python | 标准 CPython 3.12 x64，以实际验收的小版本记录为准 |
| 计算 | CPU、FP32 的完整命令行流程 |
| 文本 | BPE 训练，原生 tokenizer 保存/加载，编码/解码，单文件及有序多文件预处理 |
| 模型 | TransformerLM、RoPE、RMSNorm、SwiGLU/GELU；正式语料历史实验仍为 SwiGLU |
| 运行 | 配置检查、训练、独立验证、保存与恢复、单提示词续写 |
| 安装 | wheel 用户安装；源码 editable 开发安装；分别说明运行依赖和开发依赖 |

Python `>=3.11` 是当前包的最低版本声明，不等于该范围每个版本都已完成完整流程验收。
Linux CI 已有配置，但不能仅凭工作流文件存在就宣称 Linux wheel 已验收。
Linux、GPU、MPS、MSYS2 Python 与其他精度不列入首版完整流程承诺；已有实现或专项测试保留，
不因此删除相应能力，也不把未验收的平台描述为已证明不可用。

Windows 上的目标环境需记录 Python、PyTorch、NumPy、regex、PyYAML 的实际版本和硬件信息。
当前历史参考为 CPython 3.12.10、torch 2.11.0+cpu；最终候选版另存实际环境记录。

## 2. 当前格式与兼容约定

以下按现有实现描述，不增加未来版本永久兼容的承诺。未来变更持久化格式时须更新版本及迁移说明，
不以改文件名或忽略校验来声称兼容。

| 对象 | 当前格式及加载边界 | 首版验收要求 |
| --- | --- | --- |
| YAML 配置 | dataclass 对应 model/data/training/output，无单独 schema 版本；拒绝未知/重复字段和非法参数 | 配置内相对路径按配置文件目录解析；check-config 不要求数据存在、不创建输出 |
| Tokenizer | `tiny-lm-bpe`，version=1，预分词标识 `gpt2-regex-v1` | 原生 JSON 往返及指纹匹配；未知版本拒绝；不自动导入旧作业双文件格式 |
| Token 数据 | `tiny-lm-tokens`，version=1；bin 与旁置 JSON 配套，小端 uint16/uint32 | 校验 dtype、长度、哈希、词表与 tokenizer 指纹；不可仅复制 bin 后宣称完整可用 |
| Checkpoint | 当前写入 format_version=2，低层读取兼容 v1 | 绑定 v2 正常恢复；未知版本和旧作业无版本格式拒绝；兼容读取不等于允许默认续训 |
| Python API / CLI | 以当前公开文档和帮助列出的接口为准 | 不承诺内部下划线函数、全部日志文字或精确报错文案长期不变；已知变更需写更新日志 |

详细依据：[Tokenizer](tokenizer.md)、[预处理](preprocess.md)、[checkpoint](checkpoint.md)及[输入绑定](checkpoint-binding.md)。

严格恢复训练保持完整配置一致，包括已解析路径、设备、max_steps 和学习率计划，并检查模型结构、
权重 dtype、优化器映射、训练/验证 token 内容、已记录的 tokenizer 指纹、eval_batches 和随机状态。
完成 k 步的快照继续执行第 k 次调度对应的下一次更新，不能将恢复误解为额外再跑 max_steps 步。
不承诺移动目录后无条件续训，也不承诺不同设备或数值内核逐位一致。

v1 或未绑定 v2 的续训/文本命令默认拒绝缺失绑定；`--allow-unbound-checkpoint` 仅允许缺失的历史信息，
不能绕过已知不匹配。独立验证允许新验证集和采样参数；生成核对 tokenizer，不加载训练数据或恢复训练 RNG。
SwiGLU checkpoint 不能只改配置为 GELU 后继续加载；同 d_ff 下两者参数结构与数量不同。

## 3. 分发范围

- wheel：正式 tiny_lm 代码、依赖与入口元信息、项目许可和第三方声明。
- 同标签源码包：配置模板、文档、测试及自行编写的小型演示文本，配合 wheel 使用。
- 配置和演示文本当前不在 wheel 内；README 必须说明获取方式，不能暗示单个 wheel 自带演示资源。
- 训练语料、派生 token、训练 tokenizer、权重、复现 ZIP、虚拟环境和运行缓存不作为本次正式版附件。
- 历史 0.1.0 / 0.1.0a2 wheel 保持原身份与哈希，不能重命名为新正式版替代重新构建。

软件许可范围沿用[来源与许可](provenance.md)；若以后单独分发实验材料，另做许可与来源核查。

## 4. 发布门槛与顺序

下列未勾选项都是正式版候选的待验收任务；alpha.2 的历史通过记录不直接代替它们。

### A. 范围与现状（本轮）

- [x] 明确 Windows / CPython 3.12 / CPU FP32 的完整流程范围。
- [x] 对照配置、tokenizer、数据、checkpoint 实现，记录格式和恢复边界。
- [x] 区分软件附件与实验材料，保留现有功能限制。

### B. 候选代码整理与回归（已完成）

- [x] 2026-11-02 将 config.py 异常及 CLI 配置错误前缀统一为英文，并同步既有断言；配置/CLI 专项测试 85 passed（10.54s），相关文件 Ruff 检查通过。未改成功提示、帮助语言或校验逻辑。
- [x] 2026-11-03 核对配置与完整依赖后，Ruff 两项检查及全量 pytest 通过：1414 passed、0 failed、0 errors、0 skipped，退出码均为 0。
- [x] 记录提交 `e305e47`、环境、精确命令及耗时，见[完整回归记录](regression-20261103.md)；没有缺依赖跳过。
- [x] 没有实际失败，无需修代码；保留既定格式排除配置，未进行全库格式重写。

### C. 最终版本与包元信息

- [x] 确定正式版本号 `0.1.1`，计划标签 `v0.1.1`，避免复用已有 0.1.0 历史快照标识。
- [x] 迁移新版许可字段，声明 `MIT AND LicenseRef-OpenAI-GPT2` 并映射原始文本；构建后端最低版本为 77.0.3，配置检查构建没有弃用警告。
- [x] 更新 CHANGELOG、安装说明及[发布草稿](release-0.1.1.md)；34 项版本/导入专项测试、Ruff 与构建元数据核对通过。
- [x] 0.1.1 变更后重新完成全量 1414 项回归，0 失败/跳过，Ruff 通过，见[最终记录](final-0.1.1-checks.md)。

### D. 最终产物验收

- [x] 构建 0.1.1 wheel，保存工具版本、命令、日志、文件清单和 SHA-256，见[最终记录](final-0.1.1-checks.md)。
- [x] 包版本、入口、依赖、四份许可、23 个源码文件和 RECORD 核验通过，无语料或权重。
- [x] 新独立环境安装最终 wheel，仓库外 site-packages 来源、安装哈希及 pip check 通过。
- [x] 两种 CLI 入口及配置检查、预处理、两步保存、恢复到四步、验证和生成通过。
- [x] 输入哈希与 LF 换行已记录，没有设置任意 loss 阈值。
- [x] 错误配置、格式/指纹不匹配和未绑定恢复拒绝行为由本次完整测试覆盖。

### E. 发布

- [ ] 核对最终提交、版本与标签关系；历史标签不移动。
- [ ] 创建非预发布 Release，上传同一已验收 wheel 和校验文件，提供配套源码与安装说明。
- [ ] 下载 Release 附件核对实际 SHA-256，更新真实发布日期及链接。

## 5. 明确不阻塞首版的事项

Linux/GPU 扩展验收、更长训练、GELU 正式语料效果对照、AMP、梯度累积、KV cache、批量生成、
任意文本分块的等价流式预处理均可后续独立规划。它们不得作为已支持功能宣传。
当前生成质量有限、窗口可跨故事边界、EOS 不自动隔离 attention、固定随机窗口不等于全验证集遍历，
这些限制继续保留在使用说明中。

2026-11-03 B 阶段完整回归及 C 阶段版本/包元信息整理已完成。
随后完成 C 的最终回归与 D 的 wheel 验收。下一项为 E 正式发布，尚未创建标签或上传正式 Release。

英文报错专项检查使用项目 `.venv`：

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_cli.py -q -p no:cacheprovider --basetemp=D:/Python/.tmp/config-errors-20261102
& ./.venv/Scripts/python.exe -m ruff check --no-cache src/tiny_lm/config.py src/tiny_lm/cli.py tests/test_config.py tests/test_cli.py
& ./.venv/Scripts/python.exe -m ruff format --no-cache --check src/tiny_lm/config.py tests/test_config.py tests/test_cli.py
```

首次 Ruff 因缓存权限失败，改为 `--no-cache` 后通过；首次 pytest 显示 100% 后未及时退出，已中断，
不以其进度条作为通过证据。上述关闭缓存、独立临时目录的复跑退出码为 0。
这不是全量回归记录，也不替代正式版最终 wheel 验收。
