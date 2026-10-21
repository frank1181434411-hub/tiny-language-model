# 来源、归属与许可

- 学习来源：Stanford CS336 Assignment 1。课程年份、作业版本和具体引用链接
  仍由项目所有者确认；源码使用用户指定的本地目录。
- 用户说明：已自行实现 BPE、Tokenizer、Transformer、损失、AdamW、调度、
  采样、checkpoint 与生成代码；目前已迁入 loss、optimizer、scheduler 和采样，其他实现待迁移，
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
- 本次新增：工程框架、配置/CLI、测试和项目文档。
  `examples/toy_corpus.txt` 为本次自行编写的小段中英文示例。
- 后续应区分个人实现、课程骨架、课程测试/adapters、第三方片段，
  记录作者、来源版本、改动及原始许可/声明。
- 待核实：课程材料的许可和解答公开政策、测试与骨架能否再分发、
  第三方依赖与未来数据集的许可、署名及用途限制。

开源是项目目标，不是当前许可结论。本次不添加 MIT 或其他 LICENSE，
也不替未经审查的课程代码选择许可证。完成来源与权利审查后，
由项目所有者确定发布范围和合适许可；本文件本身不授予许可。
