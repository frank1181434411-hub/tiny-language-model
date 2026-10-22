"""
串联 token 数组采样、前向、loss、反向、裁剪、更新、验证和 checkpoint。
提供数组 Python 接口；CLI 负责文件加载、tokenizer 校验与进度输出。
"""

from __future__ import annotations
from collections.abc import Callable
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from tiny_lm.config import ProjectConfig
from tiny_lm.evaluate import _check_data,_evaluate_batches,_positive_integer

if TYPE_CHECKING:
    import numpy as np
    import torch
    from tiny_lm.model.transformer import TransformerLM
    from tiny_lm.optim import AdamW


@dataclass(frozen=True)
class TrainingMetrics:
    step:int
    learning_rate:float
    train_loss:float
    validation_loss:float|None


@dataclass
class TrainingResult:
    model:TransformerLM
    optimizer:AdamW
    completed_updates:int
    history:list[TrainingMetrics]
    checkpoints:list[Path]
    generators:dict[str,torch.Generator]


def train(config:ProjectConfig,train_data:np.ndarray,validation_data:np.ndarray,*,dtype=None,eval_batches:int=4,resume_from=None,stop_after:int|None=None,save_checkpoints:bool=True,on_step:Callable[[TrainingMetrics],None]|None=None)->TrainingResult:
    """
    创建模型和 AdamW，在 token 数组上训练；返回本次更新日志及最终运行对象。
    max_steps 始终是调度终点；stop_after 是本次最多完成的总更新次数，不改变调度。
    checkpoint 保存两个独立采样器和完整配置；续训须使用相同数据、dtype 与 eval_batches。
    on_step 在本步验证和必要的保存完成后调用，用于输出指标；回调不得改变训练状态。
    """
    import numpy as np
    import torch
    from tiny_lm.checkpoint import load_checkpoint,save_checkpoint
    from tiny_lm.data import get_batch
    from tiny_lm.losses import cross_entropy
    from tiny_lm.model.transformer import TransformerLM
    from tiny_lm.optim import AdamW,clip_gradient_norm
    from tiny_lm.schedule import get_lr_cosine_schedule
    if not isinstance(config,ProjectConfig):
        raise TypeError("config must be a ProjectConfig")
    _positive_integer("eval_batches",eval_batches)
    if type(save_checkpoints) is not bool:
        raise TypeError("save_checkpoints must be a bool")
    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable or None")
    training=config.training
    limit=training.max_steps if stop_after is None else stop_after
    if type(limit) is not int:
        raise TypeError("stop_after must be an integer or None")
    if not 0<=limit<=training.max_steps:
        raise ValueError("stop_after must be between 0 and config.training.max_steps")
    if not 0<=training.seed<2**64:
        raise ValueError("seed must be between 0 and 2**64 - 1")
    dtype=torch.float32 if dtype is None else dtype
    if dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise TypeError("dtype must be float16, bfloat16, float32 or float64")
    device=torch.device(training.device)
    if device.type not in ("cpu","cuda"):
        raise ValueError("training currently supports CPU and CUDA devices only")
    if device.type=="cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index>=torch.cuda.device_count())):
        raise ValueError("the configured CUDA device is not available")
    _check_data(train_data,config.model.context_length,config.model.vocab_size,"train_data")
    _check_data(validation_data,config.model.context_length,config.model.vocab_size,"validation_data")
    random.seed(training.seed)
    np.random.seed(training.seed%2**32)
    torch.manual_seed(training.seed)
    model=TransformerLM.from_config(config.model,device=device,dtype=dtype)
    optimizer=AdamW(model.parameters(),lr=training.learning_rate,weight_decay=training.weight_decay)
    generators={"train":torch.Generator(device="cpu").manual_seed(training.seed),"validation":torch.Generator(device="cpu").manual_seed((training.seed+1)%2**64)}
    completed=0
    if resume_from is not None:
        completed=load_checkpoint(resume_from,model,optimizer,config=config,generators=generators)
        if completed>limit:
            raise ValueError("checkpoint iteration exceeds the requested stop_after or max_steps")
    history=[]
    checkpoints=[]
    output_dir=config.output.output_dir
    if save_checkpoints and completed<limit:
        output_dir.mkdir(parents=True,exist_ok=True)
    model.train()
    for step in range(completed,limit):
        lr=get_lr_cosine_schedule(step,training.learning_rate,training.min_learning_rate,training.warmup_steps,training.max_steps)
        for group in optimizer.param_groups:
            group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        x,y=get_batch(train_data,training.batch_size,config.model.context_length,device,generator=generators["train"])
        loss=cross_entropy(model(x),y)
        loss_value=loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError("training loss must be finite")
        loss.backward()
        clip_gradient_norm(model.parameters(),training.grad_clip)
        optimizer.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise FloatingPointError("model parameters must remain finite after an optimizer update")
        completed=step+1
        validation_loss=None
        if completed%training.eval_interval==0 or completed==training.max_steps:
            validation_loss=_evaluate_batches(model,validation_data,training.batch_size,config.model.context_length,eval_batches,generators["validation"]).loss
        history.append(TrainingMetrics(step=completed,learning_rate=lr,train_loss=loss_value,validation_loss=validation_loss))
        if save_checkpoints and (completed%training.save_interval==0 or completed==limit):
            path=output_dir/f"step_{completed:08d}.pt"
            save_checkpoint(model,optimizer,completed,path,config=config,generators=generators)
            checkpoints.append(path)
        if on_step is not None:
            on_step(history[-1])
    return TrainingResult(model=model,optimizer=optimizer,completed_updates=completed,history=history,checkpoints=checkpoints,generators=generators)
