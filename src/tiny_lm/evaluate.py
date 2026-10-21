"""
在 token 数组上采样验证窗口，计算按 token 数加权的平均交叉熵。
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from tiny_lm.model.transformer import TransformerLM


@dataclass(frozen=True)
class EvaluationResult:
    loss:float
    num_batches:int
    num_tokens:int


def _positive_integer(name,value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value<=0:
        raise ValueError(f"{name} must be positive")


def _check_data(data,context_length,vocab_size,name):
    import numpy as np
    if not isinstance(data,np.ndarray) or isinstance(data,np.ma.MaskedArray):
        raise TypeError(f"{name} must be a NumPy ndarray or memmap without masked values")
    if data.ndim!=1 or data.dtype.kind not in ("i","u"):
        raise ValueError(f"{name} must be a one-dimensional integer token array")
    if len(data)<=context_length:
        raise ValueError(f"{name} must contain at least context_length + 1 tokens")
    # 分块检查全部 token，避免非法数据在多次更新后才被随机抽中。
    for start in range(0,len(data),1_000_000):
        chunk=data[start:start+1_000_000]
        if int(chunk.min())<0 or int(chunk.max())>=vocab_size:
            raise ValueError(f"{name} token IDs must be between 0 and vocab_size - 1")


def _evaluate_batches(model,data,batch_size,context_length,num_batches,generator):
    import torch
    from tiny_lm.data import get_batch
    from tiny_lm.losses import cross_entropy
    modes=[(module,module.training) for module in model.modules()]
    weighted_losses=[]
    num_tokens=0
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                x,y=get_batch(data,batch_size,context_length,model.token_embeddings.weight.device,generator=generator)
                loss=cross_entropy(model(x),y).item()
                if not math.isfinite(loss):
                    raise FloatingPointError("validation loss must be finite")
                weighted_losses.append(loss*y.numel())
                num_tokens+=y.numel()
    finally:
        # 逐模块恢复，保留调用方可能设置的混合 train/eval 模式。
        for module,training in modes:
            module.training=training
    return EvaluationResult(loss=math.fsum(weighted_losses)/num_tokens,num_batches=num_batches,num_tokens=num_tokens)


def evaluate(model:TransformerLM,data:np.ndarray,*,batch_size:int,context_length:int,num_batches:int,generator:torch.Generator)->EvaluationResult:
    """
    使用调用方提供的独立 CPU generator 验证，成功后该 generator 向前推进。
    禁用梯度并在成功或异常后恢复各模块模式，不清除或覆盖已有参数梯度。
    返回随机窗口的 token 加权平均 loss，不代表完整语料遍历。
    """
    import torch
    from tiny_lm.model.transformer import TransformerLM
    if not isinstance(model,TransformerLM):
        raise TypeError("model must be a TransformerLM")
    for name,value in (("batch_size",batch_size),("context_length",context_length),("num_batches",num_batches)):
        _positive_integer(name,value)
    if context_length>model.context_length:
        raise ValueError("context_length must not exceed the model context_length")
    if not isinstance(generator,torch.Generator) or generator.device.type!="cpu":
        raise TypeError("generator must be an independent CPU torch.Generator")
    if generator is torch.default_generator:
        raise ValueError("generator must not be the global default generator")
    _check_data(data,context_length,model.vocab_size,"validation_data")
    return _evaluate_batches(model,data,batch_size,context_length,num_batches,generator)
