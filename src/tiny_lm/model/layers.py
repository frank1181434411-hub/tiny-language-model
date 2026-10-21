"""
从 Assignment 1 迁移的 Linear、Embedding、RMSNorm、SiLU 和 SwiGLU。
"""

from __future__ import annotations
import math
import torch
from torch import nn


def _check_size(name,value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value<=0:
        raise ValueError(f"{name} must be positive")


def _check_dtype(dtype):
    if dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise TypeError("dtype must be float16, bfloat16, float32 or float64")


def _check_input(x,size=None):
    if not isinstance(x,torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if x.layout!=torch.strided:
        raise ValueError("input must be a dense strided tensor")
    _check_dtype(x.dtype)
    if size is not None and (x.ndim==0 or x.shape[-1]!=size):
        raise ValueError(f"input last dimension must be {size}")


class Linear(nn.Module):
    """
    无 bias 的线性映射，权重采用三倍标准差截断正态初始化。
    """

    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        _check_size("in_features",in_features)
        _check_size("out_features",out_features)
        _check_dtype(torch.get_default_dtype() if dtype is None else dtype)
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        std_d=(2.0/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std_d,a=-3*std_d,b=3*std_d)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        _check_input(x,self.weight.shape[1])
        return torch.einsum('...i,oi->...o',x,self.weight)


class Embedding(nn.Module):
    """
    整数 token 查表，权重按标准差为 1 的截断正态初始化。
    """

    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        _check_size("num_embeddings",num_embeddings)
        _check_size("embedding_dim",embedding_dim)
        _check_dtype(torch.get_default_dtype() if dtype is None else dtype)
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))
        std_d=1.0
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std_d,a=-3*std_d,b=3*std_d)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        if not isinstance(token_ids,torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.layout!=torch.strided:
            raise ValueError("token_ids must be a dense strided tensor")
        if token_ids.dtype not in (torch.int32,torch.int64):
            raise TypeError("token_ids must have dtype int32 or int64")
        if token_ids.device!=self.weight.device:
            raise ValueError("token_ids and embedding weights must be on the same device")
        if torch.any(token_ids<0).item() or torch.any(token_ids>=self.weight.shape[0]).item():
            raise ValueError("token_ids must be between 0 and num_embeddings - 1")
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    在最后一维做 RMS 归一化；FP16/BF16 使用 FP32 计算，FP64 保留精度。
    乘以可学习增益后转回输入 dtype。
    """

    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        _check_size("d_model",d_model)
        _check_dtype(torch.get_default_dtype() if dtype is None else dtype)
        if type(eps) not in (int,float):
            raise TypeError("eps must be a finite positive number")
        try:
            finite=math.isfinite(eps)
        except OverflowError:
            finite=False
        if not finite or eps<=0:
            raise ValueError("eps must be finite and positive")
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.ones(d_model,**factory_kwargs))
        self.eps=eps

    def forward(self,x:torch.Tensor)->torch.Tensor:
        _check_input(x,self.weight.shape[0])
        in_dtype=x.dtype
        x_float=x.to(torch.float32) if in_dtype in (torch.float16,torch.bfloat16) else x
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)
        ret=(x_float/rms)*self.weight
        return ret.to(in_dtype)


def silu_fn(in_features):
    """
    保留手写 SiLU：x*sigmoid(x)。
    """
    _check_input(in_features)
    return in_features*torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """
    两路升维后做 SiLU 门控，再降回 d_model；保留 w1、w3、w2 参数名。
    """

    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        _check_size("d_model",d_model)
        _check_size("d_ff",d_ff)
        self.d_model=d_model
        self.d_ff=d_ff
        self.w1=Linear(d_model,d_ff,device,dtype)
        self.w3=Linear(d_model,d_ff,device,dtype)
        self.w2=Linear(d_ff,d_model,device,dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        gate=silu_fn(self.w1(x))
        sign=self.w3(x)
        return self.w2(gate*sign)
