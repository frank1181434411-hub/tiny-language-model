# 架构与数据流

当前已实现：配置加载与校验、CLI 帮助/版本、显式失败的预留入口和平均交叉熵。
下列计算流程除明确标注的 loss 外均为规划，尚未连接成训练或生成流程。

```text
原始 UTF-8 文本
  → tokenizer/bpe.py：训练词表与 merges
  → tokenizer/serialization.py：持久化 tokenizer 与元信息
  → tokenizer/tokenizer.py：文本编码为 token IDs
  → 数据预处理：写入 token 数据与 dtype、词表版本等元信息
  → data.py：采样输入 x 与下一 token 目标 y
  → model/layers.py + attention.py + transformer.py：输出 logits
  → losses.py：计算平均交叉熵（已实现并有 CPU 数值/梯度测试）
  → train.py：反向传播与梯度裁剪
  → schedule.py + optim.py：学习率与参数更新
  → evaluate.py / checkpoint.py：验证、保存与恢复

提示文本 → tokenizer 编码 → 模型 → generate.py 采样与 EOS 判定
         → 逐步追加 token → tokenizer 解码 → 续写文本
```

`config.py` 提供结构化参数，未来由运行入口显式构造各组件。
模块导入不能启动任务或访问训练数据；文件存在性与运行设备检查由未来入口负责。
checkpoint 计划包含配置、步数、模型、优化器与随机数状态，
实验记录应关联 tokenizer 和数据版本。具体持久化格式尚待迁移时确定。
