# v0.1.0-alpha.2 发布记录

已发布：[v0.1.0-alpha.2](https://github.com/frank1181434411-hub/tiny-language-model/releases/tag/v0.1.0-alpha.2)。
2026-11-02 通过 GitHub API 核实发布时间为 2026-10-28 13:50:14 UTC（北京时间 21:50:14），
prerelease=true、draft=false。本轮未修改线上 Release 或标签；随后已校验用户下载的附件，结果见下文。
当前包版本为 `0.1.0a2`，已重新构建并通过[独立安装验收](wheel-alpha2-checks.md)。旧 0.1.0 wheel 保留为历史快照。

## 发布字段参考

以下保留发布准备时的填写说明，不是要求重建现有版本；后续 main 文档更新不应移动已有标签。

- Tag：`v0.1.0-alpha.2`；若标签已经存在，先核对其提交，不能移动旧标签来替换历史版本。
- Target：推送本轮文档后的 `main` 最新提交，发布前核对提交 SHA；`8750f77` 已包含版本、附件与验收记录。
- Title：`v0.1.0-alpha.2 — GELU 支持、TinyStories 基线与复现验收`。
- Release label：Pre-release。
- 附件：`dist/tiny_language_model-0.1.0a2-py3-none-any.whl` 及同目录下的 `.whl.sha256` 文件，均已生成并核验。
- 不附带包含语料、tokenizer、权重及缓存的个人复现材料包。

## 可粘贴的 Release notes

以下保留本地整理的发布文案；后续补充内容不代表已同步修改线上正文。

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
README 提供 wheel 用户安装和源码开发安装两条路径；配置与示例文本请从同标签源码包的 configs/、examples/ 获取，未打包进 wheel。
跨设备和本版本 wheel 安装的检查范围分别见 docs/cross-device-checks.md 与 docs/wheel-alpha2-checks.md。
```

## 发布准备流程（历史参考）

alpha.2 已发布，以下步骤仅保留作流程参考，不需要为本次状态同步重新执行。

1. 提交并推送本轮文档；用 `git rev-parse HEAD` 记下最终提交，在 GitHub 的 main 分支核对。
2. 打开仓库 Releases，先检查是否已有 `v0.1.0-alpha.2`。若已存在，核对目标提交与附件，不移动旧标签或直接替换已有发行包。
3. 若不存在，选择 Draft a new release，创建标签 `v0.1.0-alpha.2`，Target 选择刚核对的 main。
   操作期间如 main 又有新提交，重新核对目标；不要误发布未验收的代码版本。
4. 填入上面的标题，复制“可粘贴的 Release notes”代码块内部正文，不复制外层代码围栏。
5. 上传 `dist/tiny_language_model-0.1.0a2-py3-none-any.whl` 和对应 `.whl.sha256`，不要选择旧 0.1.0 包。
6. 勾选 This is a pre-release，先 Save draft 检查附件、标题与正文，确认后 Publish release。
7. 发布后下载附件核对 SHA-256，并确认版本标签指向预期提交。GitHub 自动生成的 Source code ZIP 可提供配套配置与示例文本。

操作依据：[GitHub 官方 Release 管理说明](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)。
本轮只修改文档，已验收 wheel 保持原哈希；其中内嵌 README 是构建时的版本，最新安装说明以标签源码中的 README 为准。
本地更新日志现已记录实际发布日期与 Release 链接，下载附件的完整哈希核对已完成。

## 2026-11-02 下载附件校验

用户提供的下载文件实际位于 `C:/Users/ERAZER/Downloads`：
`tiny_language_model-0.1.0a2-py3-none-any.whl` 与对应 `.whl.sha256`。
以 PowerShell `Get-FileHash -Algorithm SHA256` 计算下载 wheel，大小为 60127 字节，结果为：

```text
947509a8677d1c20ec3fa3a24cc8e1e3d9e317529f3f62b317fae08e52ed33b2
```

下载的校验文件格式和文件名匹配；下载 wheel 的实际哈希、下载校验文件中的哈希、
仓库 dist 内原 wheel 的实际哈希以及 alpha.2 验收记录中的哈希四者一致。
本次只读校验用户下载的文件，未重新下载、安装或训练；结果确认该文件与已验收构建一致。
