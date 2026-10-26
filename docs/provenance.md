# 来源、归属与许可

- 学习来源：Stanford CS336 Assignment 1。课程年份、作业版本和具体引用链接
  仍由项目所有者确认；源码使用用户指定的本地目录。
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
  细节见 [BPE 迁移记录](bpe.md)，课程来源与许可仍待审查。
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
- 待核实：课程材料的许可和解答公开政策、测试与骨架能否再分发、
  第三方依赖与未来数据集的许可、署名及用途限制。

开源是项目目标，不是当前许可结论。本次不添加 MIT 或其他 LICENSE，
也不替未经审查的课程代码选择许可证。完成来源与权利审查后，
由项目所有者确定发布范围和合适许可；本文件本身不授予许可。
