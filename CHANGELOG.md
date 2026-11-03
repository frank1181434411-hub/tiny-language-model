# 更新日志

本文件汇总可追溯的项目里程碑；日期以实际记录为准，未发布内容不视为已发行版本。

## v0.1.1 — 2026-11-03（正式版）

- 源码版本设为 `0.1.1`，避免与历史 `0.1.0` wheel 混淆；不移动旧标签或覆盖旧附件。
- 配置异常及 CLI 配置错误前缀统一为英文，校验逻辑不变。
- 许可元数据迁移为 `project.license` SPDX 表达式与 `project.license-files`；构建后端最低版本提升为 setuptools 77.0.3。
- 表达式为 `MIT AND LicenseRef-OpenAI-GPT2`，自定义引用对应原有 Modified MIT 文件；保留四份许可/声明，未改变组件授权条款。
- 明确首版 Windows 11 / CPython 3.12 / CPU FP32 范围，区分 wheel 用户安装与源码开发安装。
- 版本调整前的完整回归为 1414 passed，见[记录](docs/regression-20261103.md)；该历史结果不代替最终候选验收。
- 0.1.1 最终候选完整回归为 1414 passed、0 失败/跳过；最终 wheel 独立安装与仓库外四步流程通过，见[验收](docs/final-0.1.1-checks.md)。
- 已正式发布，标签指向 `08bb5c5`；两个附件重新下载核对通过。详见[发布记录](docs/release-0.1.1.md)。

## v0.1.0-alpha.2 — 2026-10-28（预发布）

2026-10-28 整理。以下是当前阶段的能力与验收汇总，不声称每项均首次出现在 alpha.2。
本地尚无 alpha.1 标签可用于核对完整版本差异；历史版本内容不在此补造。
已发布的 [GitHub Release](https://github.com/frank1181434411-hub/tiny-language-model/releases/tag/v0.1.0-alpha.2) 与本地[发布记录](docs/release-alpha2.md)可供核对。

### 功能与实验

- 已打通 BPE、Tokenizer、预处理、TransformerLM、训练/验证、checkpoint 恢复和单提示词生成。
- 在 SwiGLU 之外接通 GELU 前馈分支；GELU 使用 PyTorch 精确激活，相关数值及集成验收记录为 575 项通过。
  这是历史专项记录，不是本次重新运行的全量测试。详见 [Transformer 文档](docs/transformer.md)。
- 完成 TinyStories 6133 篇与 50000 篇、相同模型和 15000 步预算的数据对照：
  原验证 loss 为 2.3680 → 2.3546，独立留出 loss 为 2.7094 → 2.4873。
  保留全部固定生成样本；生成连贯性仍不稳定。详见[实验报告](experiments/tinystories-50k.md)。

### 复现与安装

- 2026-10-26：[四步端到端实跑](docs/end-to-end-checks.md)通过，覆盖预处理、保存、恢复、验证和生成。
- 2026-10-27：[同机换目录验收](docs/relocation-checks.md)通过，指定生成及三组验证结果与原记录一致。
- 2026-10-28：[另一台设备独立环境的源码复现](docs/cross-device-checks.md)通过，指定文本和 token IDs 一致，
  三组验证均值相差约 -1.49e-8；不声称跨设备浮点结果逐位一致。
- 2026-10-28：[0.1.0 wheel 构建](docs/wheel-build.md)及[本机独立安装](docs/wheel-install-checks.md)通过，
  包含安装来源、许可文件、两种 CLI 入口、四步训练恢复与验证生成检查。
  wheel 演示与此前源码演示的输入换行不同，不以两次 loss 差异判定数值回归。

### 许可与文档

- 为项目原创软件和文档添加 [MIT 许可](LICENSE)，保留[第三方声明](THIRD_PARTY_NOTICES.md)以及 Stanford、OpenAI 原始许可文本。
- 整理使用索引、设备清单、跨设备指南、实验报告和带 SHA-256 清单的个人复现材料包。
  软件许可不替代语料、词表、权重和材料包的单独分发审查。

### 发布状态

- Python 包版本已统一为 `0.1.0a2`，对应 Git 标签 `v0.1.0-alpha.2`。
- 新 wheel 的内容、SHA-256、独立环境安装和四步流程已通过[验收](docs/wheel-alpha2-checks.md)；旧 0.1.0 wheel 保留为历史快照。
- 2026-11-02 本地核对：版本、附件与验收记录已随 `8750f77` 拉取；补齐 wheel 用户安装入口及发布操作说明。
- 2026-11-02 通过 GitHub API 核实：Release 于 2026-10-28 13:50:14 UTC（北京时间 21:50:14）发布，prerelease=true、draft=false。
- 2026-11-02 用户下载的 wheel 完整 SHA-256 与下载校验文件、本地构建及原验收记录一致，见[附件校验](docs/release-alpha2.md)。后续文档收尾只更新 main，不移动已有版本标签。

### 已知限制

- 主要运行验收范围为 Windows CPU FP32；Linux、GPU 及其他环境未据此获得验收。
- 已有 TinyStories 实验使用 SwiGLU，未做 GELU 正式语料效果对照；同 d_ff 下两种前馈参数量不同。
- 仅有一个训练种子的规模对照；固定随机窗口不是完整验证集遍历，loss 改善不代表生成质量全面提升。
- 文本生成仅支持单提示词，尚无 KV cache、AMP 或梯度累积；预处理不支持任意分块的等价流式编码。
