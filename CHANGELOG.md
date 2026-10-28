# 更新日志

本文件汇总可追溯的项目里程碑；日期以实际记录为准，未发布内容不视为已发行版本。

## 未发布 — 计划 v0.1.0-alpha.2

2026-10-28 整理。以下是当前阶段的能力与验收汇总，不声称每项均首次出现在 alpha.2。
本地尚无 alpha.1 标签可用于核对完整版本差异；历史版本内容不在此补造。
Release 草稿见 [发布说明](docs/release-alpha2.md)。

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

### 发布前仍需完成

- Python 包版本已统一为 `0.1.0a2`，对应计划 Git 标签 `v0.1.0-alpha.2`。
- 新 wheel 的内容、SHA-256、独立环境安装和四步流程已通过[验收](docs/wheel-alpha2-checks.md)；旧 0.1.0 wheel 保留为历史快照。
- 剩余工作：提交并推送最终文件，核对发布提交与标签，使用预发布标记；尚未创建标签或发布 Release。

### 已知限制

- 主要运行验收范围为 Windows CPU FP32；Linux、GPU 及其他环境未据此获得验收。
- 已有 TinyStories 实验使用 SwiGLU，未做 GELU 正式语料效果对照；同 d_ff 下两种前馈参数量不同。
- 仅有一个训练种子的规模对照；固定随机窗口不是完整验证集遍历，loss 改善不代表生成质量全面提升。
- 文本生成仅支持单提示词，尚无 KV cache、AMP 或梯度累积；预处理不支持任意分块的等价流式编码。
