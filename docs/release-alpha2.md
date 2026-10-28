# v0.1.0-alpha.2 发布说明草稿

2026-10-28 整理，未创建标签或发布 Release。
当前包版本为 `0.1.0a2`，已重新构建并通过[独立安装验收](wheel-alpha2-checks.md)。旧 0.1.0 wheel 保留为历史快照。

## 页面填写

- Tag：`v0.1.0-alpha.2`；若标签已经存在，先核对其提交，不能移动旧标签来替换历史版本。
- Target：包含最终版本号、文档和构建记录的提交；不要将当前草稿误当作最终发布提交。
- Title：`v0.1.0-alpha.2 — GELU 支持、TinyStories 基线与复现验收`。
- Release label：Pre-release。
- 附件：`dist/tiny_language_model-0.1.0a2-py3-none-any.whl` 及同目录下的 `.whl.sha256` 文件，均已生成并核验。
- 不附带包含语料、tokenizer、权重及缓存的个人复现材料包。

## 可粘贴的 Release notes

以下正文可用于本次预发布；发布前应先推送最终提交，并将标签指向该提交。

```markdown
这是 tiny-language-model 的阶段性预发布，面向个人学习、实验与流程复现。

### 功能
- BPE 与 Tokenizer、文本预处理、token 数据加载。
- TransformerLM、RoPE、RMSNorm，以及 SwiGLU/GELU 前馈分支。
- 交叉熵、AdamW、学习率调度和梯度裁剪。
- 训练、独立验证、checkpoint 保存与恢复，以及历史输入绑定。
- 单提示词续写，支持 temperature、top-k/top-p、EOS 和独立采样种子。

### 实验结果
- 完成 TinyStories 6133 篇与 50000 篇训练故事的对照，保持相同模型、词表和 15000 步预算。
- 原验证集 loss 从 2.3680 降到 2.3546；独立留出集 loss 从 2.7094 降到 2.4873。
- 保留训练曲线、固定生成样本及实验限制。生成仍有角色混乱、重复和情节跳跃。
- 已有 TinyStories 权重使用 SwiGLU；GELU 已接通并有专项数值/集成测试记录，但尚未做正式语料效果对照。

### 已完成的复现与安装验收
- 四步端到端示例通过：准备数据、训练两步、恢复到四步、独立验证和生成。
- 材料包在同机新目录中运行，指定生成结果和三组验证 loss 与原记录一致。
- 在另一台 Windows CPU 设备上新建独立环境并安装材料包源码：指定生成文本与 token IDs 一致，三组验证均值差异约 -1.49e-8。
- 0.1.0a2 wheel 已完成独立环境安装、导入来源、许可文件、CLI、四步训练恢复及验证生成验收；GELU 前向输出有限。

### 发行附件
- tiny_language_model-0.1.0a2-py3-none-any.whl（60127 字节）及对应 .whl.sha256。
- SHA-256：947509a8677d1c20ec3fa3a24cc8e1e3d9e317529f3f62b317fae08e52ed33b2。

### 许可
- 项目原创软件与文档采用 MIT 许可。
- 保留 Stanford CS336 与 OpenAI GPT-2 相关原始声明，详见 LICENSE、THIRD_PARTY_NOTICES.md 和 docs/provenance.md。
- 软件授权不覆盖外部依赖、TinyStories 语料及派生数据、训练词表、模型权重和复现材料包的单独分发。

### 当前限制
- 主要运行验收为 Windows CPU FP32；未完成 Linux/GPU 验收，不保证不同环境逐位一致。
- 单个训练种子、固定抽样窗口的实验结论不代表统计显著或生成质量全面改善。
- 尚无 KV cache、AMP 或梯度累积；不支持任意文本分块的等价流式预处理。
- 语料、训练词表、模型权重和运行缓存不作为本次 Release 附件提供。

使用说明见 README.md，完整实验见 experiments/README.md，复现步骤见 docs/reproduction.md。
跨设备和本版本 wheel 安装的检查范围分别见 docs/cross-device-checks.md 与 docs/wheel-alpha2-checks.md。
```
