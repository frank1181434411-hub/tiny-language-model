"""
从作业迁移单提示词自回归采样，保留温度与 top-p，增加 top-k 和独立随机状态。
"""

from __future__ import annotations
import math
from dataclasses import asdict
from typing import TYPE_CHECKING
from tiny_lm.config import ModelConfig

if TYPE_CHECKING:
    import torch
    from tiny_lm.model.transformer import TransformerLM


def _check_sampling(vocab_size,max_new_tokens,temperature,top_p,top_k,eos_token_id):
    if type(max_new_tokens) is not int or max_new_tokens<0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    for name,value in (("temperature",temperature),("top_p",top_p)):
        if type(value) not in (int,float):
            raise TypeError(f"{name} must be a finite number")
        try:
            finite=math.isfinite(value)
        except OverflowError:
            finite=False
        if not finite or value<=0:
            raise ValueError(f"{name} must be finite and positive")
    if top_p>1:
        raise ValueError("top_p must be at most 1")
    if top_k is not None and (type(top_k) is not int or not 1<=top_k<=vocab_size):
        raise ValueError("top_k must be an integer between 1 and vocab_size or None")
    if eos_token_id is not None and (type(eos_token_id) is not int or not 0<=eos_token_id<vocab_size):
        raise ValueError("eos_token_id must be an integer within the vocabulary or None")


def _sampling_probabilities(logits,temperature,top_p,top_k):
    import torch
    from tiny_lm.model.attention import softmax
    if logits.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise TypeError("generation logits must have a floating point dtype")
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("generation logits must be finite")
    with torch.autocast(device_type=logits.device.type,enabled=False):
        logits=logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
        if temperature!=1.0:
            logits=logits/(temperature+1e-8)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("temperature-scaled logits must be finite")
        if top_k is not None:
            threshold=torch.topk(logits,top_k,dim=-1).values[...,-1:]
            logits=logits.masked_fill(logits<threshold,float("-inf"))
        if top_p<1.0:
            sorted_logits,sorted_indices=torch.sort(logits,descending=True,dim=-1)
            sum_probs=torch.cumsum(softmax(sorted_logits,dim=-1),dim=-1)
            indices_remove_flg=sum_probs>top_p
            indices_remove_flg[...,1:]=indices_remove_flg[...,:-1].clone()
            indices_remove_flg[...,0]=False
            indices_remove_pos=indices_remove_flg.scatter(1,sorted_indices,indices_remove_flg)
            logits=logits.masked_fill(indices_remove_pos,float("-inf"))
        return softmax(logits,dim=-1)


def generate(model:TransformerLM,prompt_ids:torch.Tensor,max_new_tokens:int,*,eos_token_id:int|None=None,temperature:float=1.0,top_p:float=1.0,top_k:int|None=None,generator:torch.Generator)->torch.Tensor:
    """
    输入单条非空提示 (1,seq_len)，返回包含完整提示和新增 token 的 int64 张量。
    每步重算最后 context_length 个 token；新生成 EOS 后结束并保留 EOS。
    使用独立 CPU generator 采样；不修改提示、参数、梯度或调用方的模型模式。
    """
    import torch
    from tiny_lm.model.transformer import TransformerLM
    if not isinstance(model,TransformerLM):
        raise TypeError("model must be a TransformerLM")
    _check_sampling(model.vocab_size,max_new_tokens,temperature,top_p,top_k,eos_token_id)
    if not isinstance(prompt_ids,torch.Tensor):
        raise TypeError("prompt_ids must be a torch.Tensor")
    if prompt_ids.ndim!=2 or prompt_ids.shape[0]!=1 or prompt_ids.shape[1]==0:
        raise ValueError("prompt_ids must have shape (1,seq_len) with a nonempty prompt")
    if prompt_ids.dtype not in (torch.int32,torch.int64):
        raise TypeError("prompt_ids must have dtype int32 or int64")
    if prompt_ids.device!=model.token_embeddings.weight.device:
        raise ValueError("prompt_ids and model weights must be on the same device")
    if prompt_ids.min().item()<0 or prompt_ids.max().item()>=model.vocab_size:
        raise ValueError("prompt_ids must contain IDs within the model vocabulary")
    if not isinstance(generator,torch.Generator) or generator.device.type!="cpu":
        raise TypeError("generator must be an independent CPU torch.Generator")
    if generator is torch.default_generator:
        raise ValueError("generator must not be the global default generator")
    ret=prompt_ids.to(dtype=torch.int64).clone()
    modes=[(module,module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond=ret[:,-model.context_length:]
                logits=model(idx_cond)
                if logits.shape!=(1,idx_cond.shape[1],model.vocab_size):
                    raise ValueError("model logits must have shape (1,seq_len,vocab_size)")
                probs=_sampling_probabilities(logits[:,-1,:],temperature,top_p,top_k)
                next_token=torch.multinomial(probs.cpu(),num_samples=1,generator=generator).to(ret.device)
                ret=torch.cat((ret,next_token),dim=1)
                if eos_token_id is not None and next_token.item()==eos_token_id:
                    break
    finally:
        for module,training in modes:
            module.training=training
    return ret


def load_generation_model(src,*,device="cpu",config:ModelConfig|None=None)->TransformerLM:
    """
    从 v1 checkpoint 重建模型并推断权重精度；不构造 optimizer，不恢复训练随机状态。
    可用 ModelConfig 校验模型字段；加载时仍需在 CPU 读取完整 checkpoint。
    """
    import torch
    from tiny_lm.checkpoint import read_checkpoint,_validate_model_state
    from tiny_lm.model.transformer import TransformerLM
    if config is not None and not isinstance(config,ModelConfig):
        raise TypeError("config must be a ModelConfig or None")
    device=torch.device(device)
    if device.type not in ("cpu","cuda"):
        raise ValueError("generation currently supports CPU and CUDA devices only")
    if device.type=="cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index>=torch.cuda.device_count())):
        raise ValueError("the requested CUDA device is not available")
    payload=read_checkpoint(src)
    model_config=payload["model_config"]
    fields={"vocab_size","context_length","d_model","num_layers","num_heads","d_ff","rope_theta","use_rms_norm","norm_mode","ffn_type"}
    if not isinstance(model_config,dict) or set(model_config)!=fields:
        raise ValueError("checkpoint must contain a complete TransformerLM construction configuration")
    if config is not None and any(model_config[key]!=value for key,value in asdict(config).items()):
        raise ValueError("config.model does not match the checkpoint model configuration")
    weights=payload["model_state_dict"].get("token_embeddings.weight")
    if not isinstance(weights,torch.Tensor) or weights.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        raise ValueError("checkpoint token embedding weights must have a supported floating point dtype")
    # 在 CPU 构造并恢复初始化消耗的 RNG；移动到目标设备不重新随机初始化。
    with torch.random.fork_rng(devices=[]):
        model=TransformerLM(**model_config,device="cpu",dtype=weights.dtype)
    _validate_model_state(payload,model)
    for value in payload["model_state_dict"].values():
        if not torch.isfinite(value).all().item():
            raise ValueError("checkpoint model weights must be finite")
    model.load_state_dict(payload["model_state_dict"],strict=True)
    model.to(device)
    model.eval()
    return model
