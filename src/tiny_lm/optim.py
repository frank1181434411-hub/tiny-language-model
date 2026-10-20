"""
从 Assignment 1 迁移的 AdamW 与全局 L2 梯度裁剪。
"""

from __future__ import annotations
import math
from collections.abc import Iterable
import torch
from torch.optim import Optimizer


def _number(name,value,positive=False):
    if type(value) not in (int,float):
        raise TypeError(f"{name} must be a finite number")
    try:
        finite=math.isfinite(value)
    except OverflowError:
        finite=False
    if not finite or (value<=0 if positive else value<0):
        bound="positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")


def _check_group(group):
    _number("lr",group["lr"])
    _number("eps",group["eps"],positive=True)
    _number("weight_decay",group["weight_decay"])
    betas=group["betas"]
    if not isinstance(betas,(tuple,list)) or len(betas)!=2:
        raise ValueError("betas must contain exactly two numbers")
    for i,beta in enumerate(betas):
        _number(f"betas[{i}]",beta)
        if beta>=1:
            raise ValueError(f"betas[{i}] must be less than 1")


def _check_tensor(value,name):
    if not isinstance(value,torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout!=torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise TypeError(f"{name} must have dtype float16, bfloat16, float32 or float64")


def _compute_dtype(p):
    return torch.float32 if p.dtype in (torch.float16,torch.bfloat16) else p.dtype


class AdamW(Optimizer):
    """
    保留作业公式：Adam 更新后再衰减，eps 加在未校正的 sqrt(v) 上。
    FP16/BF16 参数使用 FP32 矩状态与更新计算，写回时转换到参数精度；
    FP32/FP64 保留原精度。不维护额外的 FP32 主参数。
    参数与梯度须为有限、稠密的实数浮点张量；不逐步扫描 NaN/Inf。
    """

    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        _check_group(defaults)
        super().__init__(params,defaults)

    def add_param_group(self,param_group):
        group=dict(param_group)
        _check_group({**self.defaults,**group})
        params=group.get("params",[])
        if isinstance(params,torch.Tensor):
            params=[params]
        elif isinstance(params,set):
            raise TypeError("params must be an ordered iterable, not a set")
        else:
            params=list(params)
        for p in params:
            _check_tensor(p,"parameter")
        group["params"]=params
        super().add_param_group(group)

    def load_state_dict(self,state_dict):
        # 基类默认把矩状态转为参数精度；从原始状态恢复，避免低精度往返丢失。
        for group in state_dict["param_groups"]:
            _check_group(group)
        super().load_state_dict(state_dict)
        for saved_group,group in zip(state_dict["param_groups"],self.param_groups,strict=True):
            for key,p in zip(saved_group["params"],group["params"],strict=True):
                saved=state_dict["state"].get(key,{})
                for name in ("exp_avg","exp_avg_sq"):
                    if name in saved:
                        self.state[p][name]=saved[name].to(device=p.device,dtype=_compute_dtype(p),copy=True)

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        # 先检查全部参数组，避免遇到非法梯度时已更新前面的参数。
        for group in self.param_groups:
            _check_group(group)
            for p in group["params"]:
                _check_tensor(p,"parameter")
                if p.grad is not None:
                    _check_tensor(p.grad,"gradient")
        for group in self.param_groups:
            beta1,beta2=group["betas"]
            eps=group["eps"]
            lr=group["lr"]
            wd=group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad.to(dtype=_compute_dtype(p))
                state=self.state[p]
                if len(state)==0:
                    state["step"]=0
                    state["exp_avg"]=torch.zeros_like(p,dtype=_compute_dtype(p),memory_format=torch.preserve_format)
                    state["exp_avg_sq"]=torch.zeros_like(p,dtype=_compute_dtype(p),memory_format=torch.preserve_format)
                exp_avg,exp_avg_sq=state["exp_avg"],state["exp_avg_sq"]
                state["step"]+=1
                t=state["step"]
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                bias_correction1=1-beta1**t
                bias_correction2=1-beta2**t
                step_size=lr*(math.sqrt(bias_correction2)/bias_correction1)
                denom=exp_avg_sq.sqrt().add_(eps)
                updated=p.to(dtype=_compute_dtype(p))
                updated.addcdiv_(exp_avg,denom,value=-step_size)
                if wd!=0:
                    updated.add_(updated,alpha=-lr*wd)
                if updated is not p:
                    p.copy_(updated)
        return loss


@torch.no_grad()
def clip_gradient_norm(parameters:Iterable[torch.nn.Parameter],max_norm:float):
    """
    按全局 L2 范数原地裁剪；保留作业的阈值判断、1e-6 与 None 返回值。
    FP16/BF16 梯度的范数和缩放使用 FP32，FP32/FP64 保留原精度。
    非有限总范数报错，避免将全部梯度污染。
    """
    _number("max_norm",max_norm)
    params_with_grad=[]
    for p in parameters:
        _check_tensor(p,"parameter")
        if p.grad is not None:
            _check_tensor(p.grad,"gradient")
            params_with_grad.append(p)
    if not params_with_grad:
        return
    total_norm=0.0
    for p in params_with_grad:
        grad=p.grad.detach().to(dtype=_compute_dtype(p))
        param_norm=torch.norm(grad,p=2)
        total_norm+=param_norm.item()**2
    total_norm=total_norm**0.5
    if not math.isfinite(total_norm):
        raise ValueError("total gradient norm must be finite")
    eps=1e-6
    if total_norm>max_norm:
        clip_coef=max_norm/(total_norm+eps)
        for p in params_with_grad:
            grad=p.grad.detach().to(dtype=_compute_dtype(p))
            grad.mul_(clip_coef)
            p.grad.copy_(grad)
