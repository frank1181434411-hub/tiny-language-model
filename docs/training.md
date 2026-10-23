# 数组训练与验证接口

`tiny_lm.train.train` 和 `tiny_lm.evaluate.evaluate` 已连接现有数值组件。
数组 API 接收一维 NumPy token 数组或 memmap，不读取配置中指定的数据文件，不编码原文。
可传入 tokenizer 记录内容指纹；未提供时仍绑定实际数组和运行参数。
现已接通 [训练/验证 CLI](training-cli.md)，由命令入口加载和校验文件后调用这些接口。
没有下载语料或运行正式模型实验。

## 训练接口

`train(config,train_data,validation_data,*,dtype=None,eval_batches=4,resume_from=None,stop_after=None,save_checkpoints=True,on_step=None,tokenizer=None,allow_unbound_checkpoint=False)`

- config 为 ProjectConfig；由它构造 TransformerLM 和项目 AdamW。
  config.model 的 GELU 仍不支持，会在模型构造时报错。
- 默认参数 dtype 为 FP32，也支持 FP16、BF16、FP64；没有 autocast、GradScaler、
  梯度累积或分布式训练。低精度中间计算沿用各组件约定。
- 设备取 config.training.device，本接口接受 CPU/CUDA；CPU 已测试，
  GPU 分支尚未实测，不承诺跨设备逐位一致。MPS 训练暂不支持。
- 两份数据都在开始前分块扫描，检查一维整数类型、最小长度和全部 token 范围。
  支持非连续数组和 memmap，不复制整份语料；初次扫描仍有 O(n) 的读开销。
  运行期间数据须保持不变。
- config.data 的路径作为实验元信息保存，不作为本接口的数据来源，
  checkpoint 不包含实际数组，但会保存其逻辑 token 内容哈希和数量。
- 每次调用先按配置 seed 初始化 Python、NumPy 和 PyTorch 全局 RNG 并创建模型；
  seed 须在 `[0,2**64)`。这是显式训练调用的行为，导入模块不改变随机状态。
- 训练/验证分别使用两个独立 CPU generator，以 seed 与 seed+1（模 2**64）初始化；
  续训时用 checkpoint 的状态覆盖初始化状态。

返回 TrainingResult，包含 model、optimizer、completed_updates、history、
本次写入的 checkpoints 路径，以及两个具名 generators。
history 仅记录本次实际更新，每条 TrainingMetrics 含：

- step：更新后已完成的次数，从 1 开始。
- learning_rate：该次更新采用的学习率。
- train_loss：该次更新前采样 batch 的平均 token loss。
- validation_loss：该次更新后若进行了验证则为平均 loss，否则为 None。

训练接口默认不自动打印日志，history 保存在内存中。返回值可用于检查或自定义展示。
可提供 on_step 回调，在本步验证和必要保存成功后接收 TrainingMetrics；CLI 用它即时输出指标。
回调须为可调用对象，不得改变训练状态或随机状态；回调异常会中止调用，不回滚已完成步骤。

## 更新、验证与保存顺序

每一步依次执行：计算学习率 → 清梯度 → 训练采样 → 前向/loss →
反向 → 全局梯度裁剪 → AdamW 更新 → 检查参数有限性 →
按间隔验证 → 记录指标 → 按间隔保存。

学习率函数的步数为更新前已完成次数，终点始终是 config.training.max_steps。
默认采用所有参数统一的学习率、weight_decay，以及 AdamW 默认 betas/eps。
loss、梯度总范数或更新后参数非有限时直接报错，不为该次异常状态写 checkpoint。
异常不会回滚当前内存中的训练步骤，可从之前成功保存的快照重启。

验证在已完成次数能整除 eval_interval 时执行，也在真正达到 max_steps 时执行。
保存发生在 save_interval 的倍数和本次最后一次更新后；同一步先验证再保存，
确保验证 generator 的推进已计入快照。
文件名为 `step_00000010.pt` 等，保存在 config.output.output_dir，
开始实际更新前按需创建目录，同名快照由现有原子保存接口替换。
save_checkpoints=False 不创建输出目录或 checkpoint。

## 验证接口

`evaluate(model,data,*,batch_size,context_length,num_batches,generator)`

model 为 TransformerLM，generator 必须是独立的 CPU torch.Generator，
不能使用全局 default_generator。输入数组在调用前进行完整范围检查。
内部训练循环复用已验证数组，不在每次间隔验证时重复扫描。

验证使用 eval 模式与 no_grad，按每个 batch 的 token 数加权汇总 loss，
返回 EvaluationResult(loss,num_batches,num_tokens)。
当前窗口长度和 batch_size 固定，因此也等于 batch loss 的算术平均。
这是随机窗口估计，不是完整验证语料遍历，也不是准确率指标。

验证会推进传入的 generator；不会清除或覆盖已有参数梯度。
成功或异常后都逐模块恢复原有 train/eval 标志，包括混合模式。
现有模型无随机前向层，验证不消耗全局 RNG，也不推进训练 generator。
若验证中途失败，已使用的验证采样不自动回滚。

## 提前停止与续训

stop_after 是本次允许达到的总更新次数，范围为 `[0,max_steps]`；
它不是追加步数，也不会修改保存的配置或学习率曲线。
例如 max_steps=30、stop_after=10 会在第 10 次更新后保存并返回；
再次传入相同配置和 resume_from 可继续到第 30 次更新。
若提前停止点不是验证间隔，不额外做验证，避免改变后续验证采样状态。

resume_from 使用 checkpoint 的完整配置、模型、optimizer、步数及两个 generator。
配置不匹配或快照步数超过本次终点会报错；已经达到终点则不更新、不另写快照。
checkpoint 的加载与 dtype、设备约定见 [checkpoint 记录](checkpoint.md)。

精确续训要求相同数据内容、dtype、软件/硬件条件和 eval_batches。
checkpoint v2 自动记录并核对数据内容、dtype 和 eval_batches；提供 tokenizer 时还绑定其指纹。
旧文件缺少绑定时默认拒绝续训，可显式传 allow_unbound_checkpoint=True 允许历史未知信息，
已有内容或参数不匹配仍拒绝。保存时已绑定 tokenizer 的续训必须再次提供它。
软件/硬件条件仍由调用方确认，详情见 [输入绑定](checkpoint-binding.md)。

## 小数据使用示例

在项目根目录、已激活项目环境的 Python 中：

```python
import numpy as np
from dataclasses import replace
from tiny_lm.config import load_config
from tiny_lm.train import train

config=load_config("configs/smoke.yaml")
config=replace(config,model=replace(config.model,vocab_size=8,context_length=4,d_model=8,num_layers=1,num_heads=2,d_ff=16),training=replace(config.training,batch_size=4,max_steps=30,learning_rate=0.03,min_learning_rate=0.003,warmup_steps=2,eval_interval=5,save_interval=10))
train_tokens=np.arange(256,dtype=np.int64)%8
validation_tokens=(np.arange(192,dtype=np.int64)+3)%8
part=train(config,train_tokens,validation_tokens,eval_batches=2,stop_after=10)
result=train(config,train_tokens,validation_tokens,eval_batches=2,resume_from=part.checkpoints[-1])
print(result.completed_updates,result.history[-1])
```

示例只检验重复 token 模式的优化流程，不代表真实语言建模效果或泛化能力。
输出使用 smoke 配置中的目录；只需内存运行时可设置 save_checkpoints=False。

## 验收范围

测试覆盖小序列训练 loss 下降、FP16/BF16/FP32/FP64 的完整循环精确续训、
提前停止点是否触发额外验证、正确的调度步数和保存间隔、改变验证频率不影响更新、
验证 loss 的独立统计对照、混合模式与梯度保护、异常后的模式恢复、
非法输入/参数、非有限数值中止，以及裁剪确实发生在 optimizer.step 之前。
训练与验证模块保持轻量导入；实现和测试沿用紧凑空格、连续 import、
单行调用、三引号独占一行且正文无空行的约定。

2026-10-21，本机 Python 3.12.10、PyTorch 2.11.0+cpu 验收：

- 新增训练/验证 41 项测试，全项目 791 passed（17.40 秒）。
- 重复 token 序列训练 30 步后，同一验证采样下的平均 loss 下降超过 50%。
- 四种 dtype 在两种提前停止位置的续训日志、全部权重、optimizer 与采样器状态，
  均与连续训练完全一致，不使用近似容差。
- Ruff 静态检查、格式检查、Git diff 空白检查、AST/token 码风检查与导入审计通过。
- 未验证 GPU/AMP、真实语料训练、远程 CI、文本生成或 CLI 运行流程。
