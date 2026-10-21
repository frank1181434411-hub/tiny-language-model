"""
从 Assignment 1 迁移的 RoPE、手写 softmax 和因果多头注意力。
"""

from __future__ import annotations
import math
import torch
from torch import nn
from tiny_lm.model.layers import Linear


def _check_tensor(name,value,min_ndim):
    if not isinstance(value,torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout!=torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise TypeError(f"{name} must have dtype float16, bfloat16, float32 or float64")
    if value.ndim<min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")


def _check_size(name,value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value<=0:
        raise ValueError(f"{name} must be positive")


class RotaryPositionalEmbedding(nn.Module):
    """
    按相邻奇偶维度旋转；位置必须在固定 max_len 范围内。
    FP16/BF16 用 FP32 计算，FP64 使用独立重建的 FP64 缓存。
    """

    def __init__(self,theta:float,d_k:int,max_len:int,device=None):
        super().__init__()
        _check_size("d_k",d_k)
        _check_size("max_len",max_len)
        if d_k%2!=0:
            raise ValueError("d_k must be even for RoPE")
        if type(theta) not in (int,float):
            raise TypeError("theta must be a finite positive number")
        try:
            finite=math.isfinite(theta)
        except OverflowError:
            finite=False
        if not finite or theta<=0:
            raise ValueError("theta must be finite and positive")
        self.dim=d_k
        self.theta=float(theta)
        self.max_len=max_len
        self.register_buffer("cos_cached",torch.empty(0,device=device),persistent=False)
        self.register_buffer("sin_cached",torch.empty(0,device=device),persistent=False)
        self._cache_dtype=None
        self._build_cache(torch.float32)

    def _build_cache(self,dtype):
        device=self.cos_cached.device
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            pow=torch.arange(0,self.dim,2,device=device,dtype=dtype)/self.dim
            freq=torch.exp(-math.log(self.theta)*pow)
            temp=torch.arange(self.max_len,device=device,dtype=dtype)
            freq_matrix=torch.outer(temp,freq)
            if not torch.isfinite(freq_matrix).all().item():
                raise ValueError("RoPE angles must be finite at the computation dtype")
            self.cos_cached=freq_matrix.cos()
            self.sin_cached=freq_matrix.sin()
        self._cache_dtype=dtype

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse=recurse)
        # dtype/device 转换后从原参数重建，避免缓存精度往返损失。
        self._cache_dtype=None
        return result

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        _check_tensor("x",x,2)
        if x.shape[-1]!=self.dim:
            raise ValueError("x last dimension must equal d_k")
        if x.shape[-2]==0:
            raise ValueError("x sequence length must be positive")
        if x.device!=self.cos_cached.device:
            raise ValueError("x and RoPE buffers must be on the same device")
        if not isinstance(token_positions,torch.Tensor):
            raise TypeError("token_positions must be a torch.Tensor")
        if token_positions.layout!=torch.strided:
            raise ValueError("token_positions must be a dense strided tensor")
        if token_positions.dtype not in (torch.int32,torch.int64):
            raise TypeError("token_positions must have dtype int32 or int64")
        if token_positions.device!=x.device:
            raise ValueError("token_positions and x must be on the same device")
        seq_len=x.shape[-2]
        if token_positions.shape in ((seq_len,),x.shape[:-1]):
            pass
        elif x.ndim>=4 and token_positions.shape==(*x.shape[:-3],seq_len):
            token_positions=token_positions.unsqueeze(-2)
        else:
            raise ValueError("token_positions must match the sequence, all leading dimensions, or batch dimensions without heads")
        if torch.any(token_positions<0).item() or torch.any(token_positions>=self.max_len).item():
            raise ValueError("token_positions must be between 0 and max_len-1")
        compute_dtype=torch.float32 if x.dtype in (torch.float16,torch.bfloat16) else x.dtype
        if self._cache_dtype!=compute_dtype or self.cos_cached.dtype!=compute_dtype:
            self._build_cache(compute_dtype)
        with torch.autocast(device_type=x.device.type,enabled=False):
            cos=self.cos_cached[token_positions]
            sin=self.sin_cached[token_positions]
            x_float=x.to(compute_dtype)
            x_even=x_float[...,0::2]
            x_odd=x_float[...,1::2]
            ret=torch.empty_like(x_float)
            ret[...,0::2]=x_even*cos-x_odd*sin
            ret[...,1::2]=x_even*sin+x_odd*cos
        return ret.to(x.dtype)


def softmax(x:torch.Tensor,dim:int=-1)->torch.Tensor:
    """
    减去最大值后归一化；低精度输入用 FP32 计算，结果转回输入 dtype。
    允许部分位置为负无穷；拒绝 NaN、正无穷与整行负无穷。
    """
    _check_tensor("x",x,1)
    if type(dim) is not int:
        raise TypeError("dim must be an integer")
    if not -x.ndim<=dim<x.ndim:
        raise ValueError("dim must index an existing axis")
    if x.shape[dim]==0:
        raise ValueError("softmax axis must be nonempty")
    with torch.autocast(device_type=x.device.type,enabled=False):
        x_float=x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
        max_x=torch.max(x_float,dim=dim,keepdim=True).values
        if not torch.isfinite(max_x).all().item():
            raise ValueError("softmax requires a finite maximum in every row; NaN, positive infinity and fully masked rows are invalid")
        stb_x=x_float-max_x
        exp_x=torch.exp(stb_x)
        sum_exp=torch.sum(exp_x,dim=dim,keepdim=True)
        ret=exp_x/sum_exp
    return ret.to(x.dtype)


def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor|None=None)->torch.Tensor:
    """
    手写缩放点积注意力；布尔 mask 中 True 表示允许关注。
    FP16/BF16 的打分、softmax 和加权求和使用 FP32，输出转回输入 dtype。
    """
    for name,value in (("Q",Q),("K",K),("V",V)):
        _check_tensor(name,value,2)
        if value.shape[-2]==0 or value.shape[-1]==0:
            raise ValueError(f"{name} sequence and feature dimensions must be positive")
    if Q.dtype!=K.dtype or Q.dtype!=V.dtype:
        raise ValueError("Q, K and V must have the same dtype")
    if Q.device!=K.device or Q.device!=V.device:
        raise ValueError("Q, K and V must be on the same device")
    if Q.shape[-1]!=K.shape[-1]:
        raise ValueError("Q and K feature dimensions must match")
    if K.shape[-2]!=V.shape[-2]:
        raise ValueError("K and V sequence lengths must match")
    try:
        batch_shape=torch.broadcast_shapes(Q.shape[:-2],K.shape[:-2],V.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("Q, K and V batch dimensions must be broadcastable") from exc
    score_shape=(*batch_shape,Q.shape[-2],K.shape[-2])
    if mask is not None:
        if not isinstance(mask,torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        if mask.dtype!=torch.bool:
            raise TypeError("mask must have dtype bool; True means allowed")
        if mask.layout!=torch.strided:
            raise ValueError("mask must be a dense strided tensor")
        if mask.device!=Q.device:
            raise ValueError("mask and Q must be on the same device")
        try:
            mask=torch.broadcast_to(mask,score_shape)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to the attention score shape") from exc
        if not mask.any(dim=-1).all().item():
            raise ValueError("mask must allow at least one key for every query")
    compute_dtype=torch.float32 if Q.dtype in (torch.float16,torch.bfloat16) else Q.dtype
    with torch.autocast(device_type=Q.device.type,enabled=False):
        q_float=Q.to(compute_dtype)
        k_float=K.to(compute_dtype)
        v_float=V.to(compute_dtype)
        d_k=Q.size(-1)
        scores=torch.einsum('...nk,...mk->...nm',q_float,k_float)/math.sqrt(d_k)
        scores=scores.expand(score_shape)
        if mask is not None:
            scores=scores.masked_fill(~mask,float('-inf'))
        probs=softmax(scores,dim=-1)
        ret=torch.einsum('...nm,...mk->...nk',probs,v_float)
    return ret.to(Q.dtype)


class CausalSelfAttention(nn.Module):
    """
    输入形状为 (batch,seq_len,d_model)，保留四个无 bias 投影及因果对角线。
    theta 与 max_seq_len 同时提供时启用 RoPE；暂不支持 KV cache。
    """

    def __init__(self,d_model:int,num_heads:int,bias:bool=False,max_seq_len=None,theta=None,device=None,dtype=None):
        super().__init__()
        _check_size("d_model",d_model)
        _check_size("num_heads",num_heads)
        if d_model%num_heads!=0:
            raise ValueError("d_model must be divisible by num_heads")
        if type(bias) is not bool:
            raise TypeError("bias must be a bool")
        if bias:
            raise ValueError("bias=True is not supported by the migrated Linear layers")
        if (theta is None)!=(max_seq_len is None):
            raise ValueError("theta and max_seq_len must be provided together")
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        if theta is not None and self.d_k%2!=0:
            raise ValueError("head dimension must be even when RoPE is enabled")
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.rope=RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device=device) if theta is not None else None

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        _check_tensor("x",x,3)
        if x.ndim!=3 or x.shape[-1]!=self.d_model:
            raise ValueError("x must have shape (batch,seq_len,d_model)")
        batch_size,seq_len,dim=x.size()
        if seq_len==0:
            raise ValueError("sequence length must be positive")
        if x.device!=self.q_proj.weight.device:
            raise ValueError("x and attention weights must be on the same device")
        if self.rope is None and token_positions is not None:
            raise ValueError("token_positions require RoPE to be enabled")
        if self.rope is not None and seq_len>self.rope.max_len:
            raise ValueError("sequence length must not exceed max_seq_len")
        q_proj=self.q_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        k_proj=self.k_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        v_proj=self.v_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arange(seq_len,device=x.device)
            q_proj=self.rope(q_proj,token_positions)
            k_proj=self.rope(k_proj,token_positions)
        mask=torch.tril(torch.ones((seq_len,seq_len),device=x.device,dtype=torch.bool))
        att_out=scaled_dot_product_attention(q_proj,k_proj,v_proj,mask=mask)
        att_out=att_out.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        return self.output_proj(att_out)
