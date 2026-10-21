# Checkpoint 迁移记录

从 Assignment 1 的 `cs336_basics/checkpointing.py` 迁移到 `tiny_lm.checkpoint`。
保留 save_checkpoint 的前四个位置参数及 load_checkpoint 的前三个位置参数；
加载仍返回整数 iteration。扩展为带版本、配置及随机状态的训练快照。
原作业目录未修改，训练与生成 CLI 仍待接入。

## 接口与步数

- `save_checkpoint(model,optimizer,iteration,out,*,config=None,generators=None)`
- `load_checkpoint(src,model,optimizer,*,config=None,generators=None,restore_rng=True)`
- `read_checkpoint(src)` 读取 CPU 上的内容，供查看配置或先构造模型，不恢复运行状态。

iteration 定义为已经完成的 optimizer 更新次数，在一次更新完成后保存。
恢复返回 k 后，下一次更新应使用调度步数 k；循环范围为 `range(k,max_steps)`。
参数可能缺失梯度，因此 AdamW 每个参数的 step 可以小于全局 iteration，但不能大于它。
不保存未完成的梯度累积；成功恢复后清除模型的旧梯度，保留调用方的 train/eval 模式。

## v1 格式

文件包含 format_version、iteration、model_class、model_config、model_state_dict、
optimizer_class、optimizer_parameter_names、optimizer_state_dict、config 与 rng_state。
拒绝未知版本或缺字段；旧作业的无版本三字段文件不自动升级，因为没有足够信息恢复
模型构造语义、随机状态和可复现续训。

TransformerLM 提供 get_config()，保存词表、上下文、层数、维度、RoPE theta、
归一化和前馈选项等构造参数。返回独立字典；设备与 dtype 不写入该字典。
模型 dtype 保存在各权重张量中。RoPE 缓存仍不写入 state_dict，重建匹配模型后按需生成。
不能在构造后手工修改模块结构或 RoPE 属性，并仍假定原构造元信息有效。

config 可选传 ModelConfig 或 ProjectConfig；后者还记录训练、数据和输出配置，
路径以字符串保存。加载传入 config 时要求与保存值完全相等，包括路径与设备字段；
不传时跳过实验配置比较，但仍严格检查 TransformerLM 构造配置。
普通 nn.Module 支持张量 state_dict 与类名/形状检查，不自动推断任意自定义模块的构造参数。

## 随机状态

保存 Python random、NumPy 全局 RandomState、PyTorch CPU RNG。
仅在 CUDA 已初始化时保存各 CUDA 设备 RNG，不为保存 CPU 模型主动初始化 CUDA。
独立的采样 RNG 通过具名字典传入，例如 `{"train":train_rng,"validation":val_rng}`。
当前独立 generator 接口仅支持 CPU torch.Generator；不接受同一对象的重复别名。

默认加载时名字集合须完全一致，状态先在临时 RNG 对象中验证，再恢复真实对象。
带 CUDA RNG 的快照要求本机 CUDA 可用且设备数量一致；该分支尚未在 GPU 实测。
跨设备仅加载状态时可指定 restore_rng=False，此时不改变全局或具名随机状态，
也无需提供保存时的 generator。不能据此声称跨设备训练完全一致。
独立 NumPy Generator、数据加载工作进程、第三方 RNG、GradScaler 和分布式状态不在 v1 范围内。

## 状态与失败处理

- 使用 `torch.load(...,map_location="cpu",weights_only=True)`，不回退到任意 pickle 对象加载。
- 在修改模型前检查类名、构造配置、权重键、shape、dtype 和 layout。
  不自动降低/提升模型权重精度；恢复到不同设备可以，但目标模型 dtype 须匹配。
- 保存 optimizer 参数名及每组顺序，拒绝同形状参数重排、外部参数和重复参数。
- 项目 AdamW 额外检查超参数、step、矩状态 shape 与 dtype；
  FP16/BF16 模型的矩状态保持 FP32，FP64 保持 FP64。
  其他 optimizer 只做通用分组/映射检查，其专用状态由自身加载器处理。
- 在实际加载前备份模型、optimizer 和需要恢复的 RNG；加载失败时尝试回滚。
  备份会增加内存占用；任意自定义加载钩子或设备故障不能保证可恢复。
- 不做全部权重/矩元素的 NaN/Inf 扫描，不将已有数值异常伪装成正常训练状态。
- 路径保存先在目标同目录创建临时文件，序列化、flush 和 fsync 后用 os.replace 替换。
  写入或替换失败会清理本次临时文件并保留旧快照。父目录须已存在。
  该策略不是跨平台断电持久性的保证，也不支持并发写同一目标。
- 二进制文件流保持旧接口，写入/读取当前位置；调用方负责 seek、关闭和失败后的流内容。
  文件流无法提供路径保存的原子替换保证。

## 使用示例

下面假定已有 model、optimizer、解析后的 project_config 和两个 CPU generator：

```python
from tiny_lm.checkpoint import save_checkpoint,load_checkpoint

generators={"train":train_rng,"validation":val_rng}
save_checkpoint(model,optimizer,completed_updates,"latest.pt",config=project_config,generators=generators)
completed_updates=load_checkpoint("latest.pt",model,optimizer,config=project_config,generators=generators)
```

需要先查看模型配置时使用 read_checkpoint；随后可将返回的 model_config
展开传给 TransformerLM，显式选择与权重匹配的 dtype 和所需设备，
再构造参数分组与保存时一致的 optimizer 并调用 load_checkpoint。
数据文件、tokenizer 内容、训练代码和软件环境也须保持一致，才有相同续训条件。

## 验证

使用真实 TransformerLM、采样、loss、调度和 AdamW，比较连续五步训练与第二步后
保存/重建/续训的损失、采样、全部权重、optimizer 状态与后续随机输出。
覆盖 FP16、BF16、FP32、FP64，文件路径与 BytesIO，两个独立 generator。
还覆盖零步恢复、清除残留梯度、保留 eval 模式、完整 ProjectConfig、
参数重排、同形状不同 RoPE、错误 dtype/shape/step/版本/RNG、
原子写入失败和加载中途失败的回滚，以及基础 torch SGD 的兼容性。

模块顶层不导入 torch/NumPy；沿用轻量导入审计。
实现与测试遵循连续 import、紧凑空格、单行调用和无空行文档字符串。

2026-10-21，本机 Python 3.12.10、PyTorch 2.11.0+cpu 验收：

- checkpoint 新增 47 项测试，全项目 750 passed（15.29 秒）。
- 四种 dtype 的中断续训与连续运行逐张量完全一致，不使用近似容差。
- Ruff 静态检查、格式检查、Git diff 空白检查和 AST/token 码风检查通过。
- GPU RNG 分支尚未实测；CPU 环境已验证带 CUDA 状态时明确拒绝默认恢复，
  并可用 restore_rng=False 跳过随机状态恢复。
- 未下载数据、运行正式训练、提交或推送 Git。
