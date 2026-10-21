"""
从 Assignment 1 迁移的 TransformerBlock 和输出 token logits 的 TransformerLM。
"""

from __future__ import annotations
import torch
from torch import nn
from tiny_lm.config import ModelConfig
from tiny_lm.model.attention import CausalSelfAttention
from tiny_lm.model.layers import Embedding,Linear,RMSNorm,SwiGLU,_check_input,_check_size


class TransformerBlock(nn.Module):
    """
    保留双残差连接、可选 RMSNorm 和 SwiGLU/SiLU 前馈网络。
    输入和输出形状均为 (batch,seq_len,d_model)，支持 pre_norm/post_norm。
    归一化与注意力核心沿用低精度提升规则，投影、前馈与残差沿用张量 dtype/autocast。
    """

    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int|None,theta:float|None,device=None,dtype=None,use_rms_norm:bool=True,norm_mode:str="pre_norm",ffn_type:str="swiglu"):
        super().__init__()
        _check_size("d_model",d_model)
        _check_size("d_ff",d_ff)
        if type(use_rms_norm) is not bool:
            raise TypeError("use_rms_norm must be a bool")
        if not isinstance(norm_mode,str):
            raise TypeError("norm_mode must be a string")
        if norm_mode not in ("pre_norm","post_norm","pre","post"):
            raise ValueError("norm_mode must be pre_norm or post_norm (legacy aliases: pre, post)")
        if not isinstance(ffn_type,str):
            raise TypeError("ffn_type must be a string")
        if ffn_type not in ("swiglu","silu","SwiGLU","SiLu"):
            raise ValueError("ffn_type must be swiglu or silu (legacy aliases: SwiGLU, SiLu); GELU is not implemented")
        self.d_model=d_model
        self.use_rms_norm=use_rms_norm
        self.norm_mode={"pre":"pre_norm","post":"post_norm"}.get(norm_mode,norm_mode)
        self.ffn_type=ffn_type.lower()
        self.attn=CausalSelfAttention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,theta=theta,device=device,dtype=dtype)
        if use_rms_norm:
            self.ln1=RMSNorm(d_model,device=device,dtype=dtype)
            self.ln2=RMSNorm(d_model,device=device,dtype=dtype)
        else:
            self.ln1=nn.Identity()
            self.ln2=nn.Identity()
        if self.ffn_type=="swiglu":
            self.ffn=SwiGLU(d_model,d_ff,device=device,dtype=dtype)
        else:
            self.ffn=nn.Sequential(Linear(d_model,d_ff,device=device,dtype=dtype),nn.SiLU(),Linear(d_ff,d_model,device=device,dtype=dtype))

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        _check_input(x,self.d_model)
        if x.ndim!=3:
            raise ValueError("x must have shape (batch,seq_len,d_model)")
        if x.shape[1]==0:
            raise ValueError("sequence length must be positive")
        if x.device!=self.attn.q_proj.weight.device:
            raise ValueError("x and block weights must be on the same device")
        if self.norm_mode=="pre_norm":
            x=x+self.attn(self.ln1(x),token_positions=token_positions)
            x=x+self.ffn(self.ln2(x))
        else:
            x=self.ln1(x+self.attn(x,token_positions=token_positions))
            x=self.ln2(x+self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """
    Embedding、多层 TransformerBlock、最终 RMSNorm 与独立词表投影。
    输入 (batch,seq_len) 的整数 token IDs，返回 (batch,seq_len,vocab_size) logits。
    保留旧参数名称与未绑定的输入/输出权重；生成和训练入口另行迁移。
    """

    def __init__(self,vocab_size:int,context_length:int,d_model:int,num_layers:int,num_heads:int,d_ff:int,rope_theta:float,device=None,dtype=None,use_rms_norm:bool=True,norm_mode:str="pre_norm",ffn_type:str="swiglu"):
        super().__init__()
        for name,value in (("vocab_size",vocab_size),("context_length",context_length),("num_layers",num_layers)):
            _check_size(name,value)
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.token_embeddings=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.layers=nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,device=device,dtype=dtype,use_rms_norm=use_rms_norm,norm_mode=norm_mode,ffn_type=ffn_type) for _ in range(num_layers)])
        self.ln_final=RMSNorm(d_model,device=device,dtype=dtype) if use_rms_norm else nn.Identity()
        self.lm_head=Linear(d_model,vocab_size,device=device,dtype=dtype)
        self._model_config=dict(vocab_size=vocab_size,context_length=context_length,d_model=d_model,num_layers=num_layers,num_heads=num_heads,d_ff=d_ff,rope_theta=rope_theta,use_rms_norm=use_rms_norm,norm_mode=self.layers[0].norm_mode,ffn_type=self.layers[0].ffn_type)

    def get_config(self)->dict:
        """
        返回独立的构造配置副本，供 checkpoint 检查并重建非持久化 RoPE 缓存。
        dtype 与设备不在配置内；权重 dtype 随 state_dict 保存。
        """
        return dict(self._model_config)

    @classmethod
    def from_config(cls,config:ModelConfig,*,device=None,dtype=None)->TransformerLM:
        """
        从已解析的 ModelConfig 构造模型；设备和参数 dtype 由调用方显式提供。
        不读取配置或权重文件，不自动选择 GPU，不改变全局随机种子。
        """
        if not isinstance(config,ModelConfig):
            raise TypeError("config must be a ModelConfig instance")
        return cls(vocab_size=config.vocab_size,context_length=config.context_length,d_model=config.d_model,num_layers=config.num_layers,num_heads=config.num_heads,d_ff=config.d_ff,rope_theta=config.rope_theta,device=device,dtype=dtype,norm_mode=config.norm_mode,ffn_type=config.ffn_type)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        if not isinstance(token_ids,torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim!=2:
            raise ValueError("token_ids must have shape (batch,seq_len)")
        batch_dim,seq_len=token_ids.shape
        if seq_len==0:
            raise ValueError("sequence length must be positive")
        if seq_len>self.context_length:
            raise ValueError("sequence length must not exceed context_length")
        x=self.token_embeddings(token_ids)
        token_positions=torch.arange(seq_len,device=token_ids.device).unsqueeze(0).expand(batch_dim,seq_len)
        for layer in self.layers:
            x=layer(x,token_positions=token_positions)
        x=self.ln_final(x)
        return self.lm_head(x)
