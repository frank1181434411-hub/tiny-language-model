"""从 Assignment 1 迁移的手写交叉熵损失。"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算所有 token 的平均交叉熵。

    logits 为有限浮点分数，形状 (..., vocab_size)；targets 为对应形状的
    int64 token ID。FP16/BF16 输入使用 FP32 计算并返回 loss。
    """
    # 调用时再加载 PyTorch，保持包导入轻量。
    import torch

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be torch.Tensor instances")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("logits must have dtype float16, bfloat16, float32 or float64")
    if targets.dtype != torch.int64:
        raise TypeError("targets must have dtype int64 (torch.long)")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty vocabulary axis")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets.shape must equal logits.shape[:-1]")
    if targets.numel() == 0:
        raise ValueError("cross_entropy requires at least one target")
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device")

    # 低精度输入提升到 FP32，FP64 输入保留原精度。
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    m = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - m
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # 在平移后的分数上计算，避免加回大数再相减造成精度损失。
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    loss = log_sum_exp - target_logits
    return torch.mean(loss)
