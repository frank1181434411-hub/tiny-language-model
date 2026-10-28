# 来源、归属与许可

- 学习来源：Stanford CS336 Spring 2025 Assignment 1，本地 README 标明该课程年份；
  [上游仓库](https://github.com/stanford-cs336/assignment1-basics)与
  [课程归档](https://cs336.stanford.edu/spring2025/)可供核对。具体上游 commit 未从本地导入历史确定。
- 用户说明：已自行实现 BPE、Tokenizer、Transformer、损失、AdamW、调度、
  采样、checkpoint 与生成代码；当前主要数值组件、文本处理、训练/验证及单提示词生成已迁移，
  归属仍需逐文件核对。
- loss 来源：`D:/Python/CS336-Private/assignment1-basics/cs336_basics/losses.py`，
  按用户要求迁至 `src/tiny_lm/losses.py`。保留手写计算流程与主要变量名，
  删除未使用变量、精简注释、调整数值精度并补输入检查；旧源码未修改。
  本次 loss 测试为新项目编写，没有复制课程测试文件或 fixtures。
- optimizer 来源与原算法、精度处理和状态恢复见 [迁移记录](optimizer.md)。
- scheduler 来源：`cs336_basics/scheduler.py`，迁至 `src/tiny_lm/schedule.py`。
  保留原函数名、主要变量与公式，补参数检查、端点直接返回和预热运算顺序调整。
  新测试独立编写，未复制课程测试或 fixtures，未修改旧源码。
- data 来源：`cs336_basics/data.py`，迁至 `src/tiny_lm/data.py`。
  保留均匀、有放回的窗口采样和错位目标，补输入/token 检查与可选 CPU generator。
  先在 CPU 转换为 int64 再移动设备；计算依赖延迟到调用时导入。
  新测试独立编写，未修改旧源码，文件加载与预处理尚未迁移。
- 基础层来源：`cs336_basics/nn.py` 中的 Linear、Embedding、RMSNorm、silu_fn、SwiGLU，
  迁至 `src/tiny_lm/model/layers.py`，保留初始化、手写计算与参数名。
  补输入检查，RMSNorm 的 FP64 输入不再降为 FP32；测试独立编写，旧源码未修改。
  具体数值约定见 [迁移记录](layers.md)。
- 注意力来源：同一 `nn.py` 中的 RotaryPositionalEmbedding、softmax、
  scaled_dot_product_attention 和 CausalSelfAttention，迁至 `src/tiny_lm/model/attention.py`。
  保留手写算法、奇偶维旋转及四个投影参数名，补精度处理、参数校验与缓存重建。
  测试独立编写，旧源码未修改；完整模型现已组装，详见 [注意力迁移记录](attention.md) 和 [模型迁移记录](transformer.md)。
- BPE 来源：`cs336_basics/train_bpe.py`，迁至 `src/tiny_lm/tokenizer/bpe.py`。
  保留增量计数、合并与同频排序；补校验、最长特殊 token 匹配并保留原始换行。
  共享规则与原生 v1 保存格式为本项目整理实现；新测试独立编写，旧源码未修改。
  细节见 [BPE 迁移记录](bpe.md)，许可核查结果见下文。
- Tokenizer 来源：`cs336_basics/tokenizer.py` 中的 `BPETokenizer`。
  保留手写最小 rank 合并流程、UTF-8 拼接解码和逐记录编码；补输入校验、
  输入容器复制、独立特殊 token 映射、普通文本模式和原生文件接口。
  测试独立编写，旧源码未修改，详见 [Tokenizer 迁移记录](tokenizer.md)。
- 预处理来源：`cs336_basics/preprocess.py`，迁至 `src/tiny_lm/preprocess.py`。
  保留编码和批量写盘思路；修正任意文本分块，增加单/多文件记录、显式 EOS、
  dtype 范围检查、版本化元信息和不覆盖发布；`data.py` 增加校验后的只读加载。
  测试独立编写，旧源码未修改，详见 [预处理迁移记录](preprocess.md)。
- 生成来源：`cs336_basics/nn.py` 的 generate、_top_p_filter 与 `inference.py` 的加载/编解码流程，
  迁至 `src/tiny_lm/generate.py`。保留滑动上下文、温度的 1e-8、核采样和 multinomial；
  增加可选 top-k、独立 CPU generator、模式恢复、精度和参数检查，限定首版单提示词。
  模型加载使用项目版本化 checkpoint 和原生 tokenizer，不迁入旧格式猜测及交互循环；
  测试独立编写，原作业未修改，详见 [文本生成](generation.md)。
- 本次新增：工程框架、配置/CLI、测试和项目文档。
  GELU 前馈为本项目新增分支，通过已有 Linear 与 `torch.nn.GELU(approximate="none")` 组装，
  不声称该激活是手写或从旧作业迁入；相关数值与集成测试独立编写。
  checkpoint v2 输入绑定为项目工程扩展，复用原生 tokenizer 指纹，并新增实际 token 序列哈希；
  相应测试独立编写，旧作业未修改，见 [输入绑定](checkpoint-binding.md)。
  `examples/toy_corpus.txt` 为本次自行编写的小段中英文示例。
- 后续应区分个人实现、课程骨架、课程测试/adapters、第三方片段，
  记录作者、来源版本、改动及原始许可/声明。
- 第三方依赖与数据集保留各自许可；后续新增第三方材料需继续记录来源、署名及用途限制。

## 2026-10-28 本地来源核查与 MIT 授权

依据项目所有者关于个人实现的说明及本次授权，原创软件和文档采用根目录
[MIT LICENSE](../LICENSE)，版权署名为 `2026 frank1181434411-hub`。
[第三方声明](../THIRD_PARTY_NOTICES.md)及 `licenses/` 保留已识别上游的原始许可，
不把第三方部分的版权改署为项目所有者。

核查对象为当前项目的 Git 跟踪文件、已有迁移记录，以及
`D:/Python/CS336-Private/assignment1-basics` 的许可证、README、源码和 Git 历史。
旧仓库未修改，未运行作业代码。证据定位如下：

| 项目 | 本地证据与结论 |
| --- | --- |
| 当前项目核查基点 | `9ca9226f13701e016078174d061dd66a6b1486f4` |
| 旧仓库 HEAD | `1d64a631e13f2fc9a5f5c5dcc5d685026aafc01d` |
| 初始骨架 | `b879f4f` 下 `cs336_basics/` 仅有 `__init__.py` 和 `pretokenization_example.py` |
| 后续实现记录 | `fb53bae` / `549f511` 为 nn，`5dcc4d9` 为 BPE，`795e7e8` 为 tokenizer，`a415235` 为采样，`1d64a63` 为预处理；作者名为 frank1181434411-hub |
| 本地课程许可 | `assignment1-basics/LICENSE` 为 MIT，署名 `Copyright 2025 Stanford University`；[官方许可](https://github.com/stanford-cs336/assignment1-basics/blob/main/LICENSE)与其一致 |
| 原许可文件 SHA-256 | `c9b91dfeafb7a8690279a75f68d93b6b8c7acb7337d1ac66162f8bc276bd30fe` |
| 新项目课程测试 | Git 文件清单没有课程 adapters、fixtures 或作业 PDF；已有迁移记录说明新测试独立编写 |

这些证据支持迁移来源说明，但 Git 作者名与文件名检查不能证明每行代码的独创性。
部分本地实现文件缺少单独的提交历史；不将它们标为已通过完整逐行第三方比对。
若以后发现其他引用，应追加相应声明。

`src/tiny_lm/tokenizer/rules.py` 的正则来自旧实现中注明的 GPT-2 预分词规则，
缩写分支做了分组改写。对照 [OpenAI encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)，
在 `licenses/OpenAI-GPT2-LICENSE.txt` 保留其 Modified MIT 许可；没有迁入完整 GPT-2 编码器或权重。
Stanford 原许可文本保存在 `licenses/Stanford-CS336-LICENSE.txt`。
构建配置显式收录项目许可及上述声明，实际 wheel 安装验收仍是后续任务。

本次 MIT 授权不覆盖外部依赖、TinyStories 原始语料、派生数据文件、训练 tokenizer、
模型权重及复现材料包。报告中的第三方文本或生成样本也不构成对原语料的重新授权。
未来分发这些材料需分别核查来源与许可；仓库的软件授权不能替代材料包分发审查。

课程软件许可与课程学术要求分别记录：[2025 课程页面](https://cs336.stanford.edu/spring2025/)
列有协作、AI 工具和现成实现的使用要求，本地课程 AGENTS.md 也强调教学用途。
本次是在作业之外整理项目来源与许可证，不代表课程方认可公开作业解答，
也不代表已核实项目所有者是否另有选课或提交义务。
