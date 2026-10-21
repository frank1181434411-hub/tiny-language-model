"""
TransformerBlock 与 TransformerLM 的独立 FP64 对照、组合梯度与接口回归。
"""

import importlib
import io
import pytest

torch=pytest.importorskip("torch",reason="TransformerBlock tests require compute dependencies")
TransformerBlock=importlib.import_module("tiny_lm.model.transformer").TransformerBlock
TransformerLM=importlib.import_module("tiny_lm.model.transformer").TransformerLM
F=torch.nn.functional
DTYPES=(torch.float16,torch.bfloat16,torch.float32,torch.float64)


def _block(**kwargs):
    options=dict(d_model=4,num_heads=2,d_ff=7,max_seq_len=8,theta=10000.0,dtype=torch.float64)
    options.update(kwargs)
    block=TransformerBlock(**options)
    generator=torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name,p in block.named_parameters():
            values=torch.randn(p.shape,generator=generator,dtype=torch.float64)*0.15
            if name.startswith("ln"):
                values+=1
            p.copy_(values)
    return block


def _reference(x,weights,norm_mode,ffn_type,use_rms_norm,positions=None):
    def norm(value,name):
        if not use_rms_norm:
            return value
        return value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+1e-5)*weights[name+".weight"]

    def rotate(value):
        angles=positions.double()[...,None]*10000.0**(-torch.arange(0,2,2,dtype=torch.float64)/2)
        if angles.ndim==2:
            angles=angles[None,None]
        else:
            angles=angles[:,None]
        even,odd=value[...,0::2],value[...,1::2]
        return torch.stack((even*angles.cos()-odd*angles.sin(),even*angles.sin()+odd*angles.cos()),dim=-1).flatten(-2)

    def attention(value):
        batch,seq,_=value.shape
        q,k,v=[F.linear(value,weights["attn."+name+".weight"]).reshape(batch,seq,2,2).transpose(1,2) for name in ("q_proj","k_proj","v_proj")]
        if positions is not None:
            q,k=rotate(q),rotate(k)
        result=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return F.linear(result.transpose(1,2).reshape(batch,seq,4),weights["attn.output_proj.weight"])

    def ffn(value):
        if ffn_type=="swiglu":
            return F.linear(F.silu(F.linear(value,weights["ffn.w1.weight"]))*F.linear(value,weights["ffn.w3.weight"]),weights["ffn.w2.weight"])
        return F.linear(F.silu(F.linear(value,weights["ffn.0.weight"])),weights["ffn.2.weight"])

    if norm_mode=="pre_norm":
        hidden=x+attention(norm(x,"ln1"))
        return hidden+ffn(norm(hidden,"ln2"))
    hidden=norm(x+attention(x),"ln1")
    return norm(hidden+ffn(hidden),"ln2")


def _close(actual,expected,dtype):
    rtol,atol={torch.float16:(1e-2,1e-3),torch.bfloat16:(7e-2,7e-3),torch.float32:(3e-5,3e-6),torch.float64:(1e-10,1e-12)}[dtype]
    torch.testing.assert_close(actual.double(),expected.double(),rtol=rtol,atol=atol)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
@pytest.mark.parametrize("ffn_type",["swiglu","silu"])
@pytest.mark.parametrize("use_rms_norm",[False,True])
def test_forward_and_all_gradients_against_float64(dtype,norm_mode,ffn_type,use_rms_norm):
    block=_block(dtype=dtype,norm_mode=norm_mode,ffn_type=ffn_type,use_rms_norm=use_rms_norm)
    x=torch.linspace(-0.9,1.1,24,dtype=dtype).reshape(2,3,4).requires_grad_()
    positions=torch.tensor([[0,2,4],[1,3,5]])
    reference_x=x.detach().double().requires_grad_()
    weights={name:p.detach().double().requires_grad_() for name,p in block.named_parameters()}
    expected=_reference(reference_x,weights,norm_mode,ffn_type,use_rms_norm,positions)
    actual=block(x,positions)
    probe=torch.linspace(-0.7,0.9,24,dtype=dtype).reshape(x.shape)
    actual_gradients=torch.autograd.grad(actual,(x,*block.parameters()),probe)
    reference_gradients=torch.autograd.grad(expected,(reference_x,*weights.values()),probe.double())
    assert actual.shape==x.shape
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for actual_gradient,reference_gradient in zip(actual_gradients,reference_gradients,strict=True):
        _close(actual_gradient,reference_gradient,dtype)


@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
@pytest.mark.parametrize("ffn_type",["swiglu","silu"])
def test_input_and_parameter_finite_differences(norm_mode,ffn_type):
    block=_block(norm_mode=norm_mode,ffn_type=ffn_type)
    names,parameters=zip(*block.named_parameters(),strict=True)
    x=torch.linspace(-0.4,0.6,8,dtype=torch.float64).reshape(1,2,4).requires_grad_()
    def forward(value,*weights):
        return torch.func.functional_call(block,dict(zip(names,weights,strict=True)),(value,))
    assert torch.autograd.gradcheck(forward,(x,*parameters),fast_mode=True)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
def test_causality_positions_and_input_unchanged(dtype,norm_mode):
    block=_block(dtype=dtype,norm_mode=norm_mode)
    x=torch.linspace(-0.5,0.7,16,dtype=dtype).reshape(1,4,4).requires_grad_()
    original=x.detach().clone()
    output=block(x)
    modified=x.detach().clone()
    modified[:,2:]+=3
    torch.testing.assert_close(output[:,:2],block(modified)[:,:2],rtol=0,atol=0)
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    torch.testing.assert_close(output,block(x,torch.arange(4)),rtol=0,atol=0)
    gradient=torch.autograd.grad(output[:,:2].sum(),x)[0]
    assert torch.count_nonzero(gradient[:,2:])==0


@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
@pytest.mark.parametrize("ffn_type",["swiglu","silu"])
@pytest.mark.parametrize("use_rms_norm",[False,True])
def test_parameter_names_and_serialized_roundtrip(norm_mode,ffn_type,use_rms_norm):
    block=_block(norm_mode=norm_mode,ffn_type=ffn_type,use_rms_norm=use_rms_norm)
    names={"attn."+name+".weight" for name in ("q_proj","k_proj","v_proj","output_proj")}
    names|={"ffn."+name+".weight" for name in (("w1","w3","w2") if ffn_type=="swiglu" else ("0","2"))}
    if use_rms_norm:
        names|={"ln1.weight","ln2.weight"}
    assert set(block.state_dict())==names
    buffer=io.BytesIO()
    torch.save(block.state_dict(),buffer)
    buffer.seek(0)
    restored=_block(norm_mode=norm_mode,ffn_type=ffn_type,use_rms_norm=use_rms_norm)
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    x=torch.linspace(-0.4,0.7,12,dtype=torch.float64).reshape(1,3,4)
    torch.testing.assert_close(block(x),restored(x),rtol=0,atol=0)


@pytest.mark.parametrize(("legacy","canonical"),[("pre","pre_norm"),("post","post_norm")])
@pytest.mark.parametrize(("old_ffn","new_ffn"),[("SwiGLU","swiglu"),("SiLu","silu")])
def test_legacy_names(legacy,canonical,old_ffn,new_ffn):
    old=_block(norm_mode=legacy,ffn_type=old_ffn)
    new=_block(norm_mode=canonical,ffn_type=new_ffn)
    assert old.norm_mode==canonical
    assert old.ffn_type==new_ffn
    x=torch.ones(1,2,4,dtype=torch.float64)
    torch.testing.assert_close(old(x),new(x),rtol=0,atol=0)


def test_no_rope_noncontiguous_and_empty_batch():
    block=_block(max_seq_len=None,theta=None)
    x=torch.linspace(-0.4,0.6,24,dtype=torch.float64).reshape(2,4,3).transpose(1,2)
    assert not x.is_contiguous()
    torch.testing.assert_close(block(x),block(x.contiguous()),rtol=0,atol=0)
    expected=_reference(x,dict(block.named_parameters()),"pre_norm","swiglu",True)
    torch.testing.assert_close(block(x),expected)
    assert block(torch.empty(0,3,4,dtype=torch.float64)).shape==(0,3,4)
    with pytest.raises(ValueError,match="require RoPE"):
        block(x,torch.arange(3))


@pytest.mark.parametrize(("options","error","message"),[({"norm_mode":"invalid"},ValueError,"norm_mode"),({"norm_mode":None},TypeError,"norm_mode"),({"ffn_type":"gelu"},ValueError,"GELU"),({"ffn_type":1},TypeError,"ffn_type"),({"use_rms_norm":1},TypeError,"use_rms_norm"),({"d_model":0},ValueError,"d_model"),({"d_ff":0},ValueError,"d_ff"),({"d_ff":True},TypeError,"d_ff"),({"num_heads":3},ValueError,"divisible"),({"num_heads":4},ValueError,"even"),({"max_seq_len":0},ValueError,"max_len"),({"theta":0},ValueError,"theta"),({"theta":None},ValueError,"provided together"),])
def test_invalid_configuration(options,error,message):
    with pytest.raises(error,match=message):
        _block(**options)


@pytest.mark.parametrize(("x","error","message"),[([[1,2,3,4]],TypeError,"Tensor"),(torch.zeros(2,4,dtype=torch.float64),ValueError,"shape"),(torch.zeros(1,2,3,dtype=torch.float64),ValueError,"last dimension"),(torch.zeros(1,0,4,dtype=torch.float64),ValueError,"positive"),(torch.zeros(1,9,4,dtype=torch.float64),ValueError,"max_seq_len"),(torch.ones(1,2,4,dtype=torch.int64),TypeError,"dtype"),(torch.zeros(1,2,4,device="meta",dtype=torch.float64),ValueError,"same device"),])
def test_invalid_inputs(x,error,message):
    with pytest.raises(error,match=message):
        _block()(x)


@pytest.mark.parametrize("positions",[torch.tensor([-1,0]),torch.tensor([0,8]),torch.tensor([0.0,1.0]),torch.zeros(2,2,dtype=torch.int64)])
def test_invalid_positions(positions):
    with pytest.raises((TypeError,ValueError),match="token_positions"):
        _block()(torch.ones(1,2,4,dtype=torch.float64),positions)


def _model(**kwargs):
    options=dict(vocab_size=11,context_length=8,d_model=4,num_layers=2,num_heads=2,d_ff=7,rope_theta=10000.0,dtype=torch.float64)
    options.update(kwargs)
    model=TransformerLM(**options)
    generator=torch.Generator().manual_seed(43)
    with torch.no_grad():
        for name,p in model.named_parameters():
            values=torch.randn(p.shape,generator=generator,dtype=torch.float64)*0.15
            if name.split(".")[-2] in ("ln1","ln2","ln_final"):
                values+=1
            p.copy_(values)
    return model


def _lm_reference(token_ids,weights,norm_mode,ffn_type,use_rms_norm,num_layers=2):
    x=F.embedding(token_ids,weights["token_embeddings.weight"])
    positions=torch.arange(token_ids.shape[1])
    for i in range(num_layers):
        prefix=f"layers.{i}."
        block_weights={name.removeprefix(prefix):value for name,value in weights.items() if name.startswith(prefix)}
        x=_reference(x,block_weights,norm_mode,ffn_type,use_rms_norm,positions)
    if use_rms_norm:
        x=x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)*weights["ln_final.weight"]
    return F.linear(x,weights["lm_head.weight"])


def _close_lm_gradient(actual,expected,dtype):
    if dtype not in (torch.float16,torch.bfloat16):
        _close(actual,expected,dtype)
        return
    # 多层和重复 token 的梯度累加存在抵消；逐张量约束整体与最大绝对误差。
    error=actual.double()-expected
    tolerance=8*torch.finfo(dtype).eps
    # 反传中间量和参数梯度均会舍入，非正规数允许一个最小量化步长。
    roundoff=torch.finfo(dtype).tiny*torch.finfo(dtype).eps
    assert torch.isfinite(error).all()
    assert error.norm()<=tolerance*expected.norm()+roundoff*expected.numel()**0.5
    assert error.abs().max()<=tolerance*expected.abs().max()+roundoff


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
@pytest.mark.parametrize("ffn_type",["swiglu","silu"])
@pytest.mark.parametrize("use_rms_norm",[False,True])
def test_lm_logits_and_all_parameter_gradients(dtype,norm_mode,ffn_type,use_rms_norm):
    model=_model(dtype=dtype,norm_mode=norm_mode,ffn_type=ffn_type,use_rms_norm=use_rms_norm)
    token_ids=torch.tensor([[1,2,1],[3,1,4]])
    weights={name:p.detach().double().requires_grad_() for name,p in model.named_parameters()}
    expected=_lm_reference(token_ids,weights,norm_mode,ffn_type,use_rms_norm)
    actual=model(token_ids)
    probe=torch.linspace(-0.5,0.7,actual.numel(),dtype=dtype).reshape(actual.shape)
    actual_gradients=torch.autograd.grad(actual,tuple(model.parameters()),probe)
    reference_gradients=torch.autograd.grad(expected,tuple(weights.values()),probe.double())
    assert actual.shape==(2,3,11)
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for actual_gradient,reference_gradient in zip(actual_gradients,reference_gradients,strict=True):
        _close_lm_gradient(actual_gradient,reference_gradient,dtype)


@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
def test_lm_parameter_finite_differences(norm_mode):
    model=_model(num_layers=1,norm_mode=norm_mode)
    names,parameters=zip(*model.named_parameters(),strict=True)
    token_ids=torch.tensor([[1,2]])
    def forward(*weights):
        return torch.func.functional_call(model,dict(zip(names,weights,strict=True)),(token_ids,))
    assert torch.autograd.gradcheck(forward,parameters,fast_mode=True)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("norm_mode",["pre_norm","post_norm"])
def test_lm_causality_untied_weights_and_independent_layers(dtype,norm_mode):
    model=_model(dtype=dtype,norm_mode=norm_mode)
    token_ids=torch.tensor([[1,2,3,4]])
    original=token_ids.clone()
    output=model(token_ids)
    torch.testing.assert_close(output[:,:2],model(torch.tensor([[1,2,5,6]]))[:,:2],rtol=0,atol=0)
    gradient=torch.autograd.grad(output[:,:2].sum(),model.token_embeddings.weight)[0]
    assert torch.count_nonzero(gradient[3:5])==0
    torch.testing.assert_close(token_ids,original)
    assert model.token_embeddings.weight.data_ptr()!=model.lm_head.weight.data_ptr()
    for left,right in zip(model.layers[0].parameters(),model.layers[1].parameters(),strict=True):
        assert left.data_ptr()!=right.data_ptr()


@pytest.mark.parametrize("dtype",DTYPES)
def test_lm_serialized_state_roundtrip(dtype):
    model=_model(dtype=dtype)
    expected_names={"token_embeddings.weight","ln_final.weight","lm_head.weight"}
    for i in range(2):
        expected_names|={f"layers.{i}."+name for name in _block().state_dict()}
    assert set(model.state_dict())==expected_names
    buffer=io.BytesIO()
    torch.save(model.state_dict(),buffer)
    buffer.seek(0)
    restored=TransformerLM(11,8,4,2,2,7,10000.0,dtype=dtype)
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    tokens=torch.tensor([[1,2,1,3]])
    torch.testing.assert_close(model(tokens),restored(tokens),rtol=0,atol=0)


def test_lm_token_layout_int32_and_sequence_boundaries():
    model=_model()
    tokens=torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.int32).t()
    assert not tokens.is_contiguous()
    torch.testing.assert_close(model(tokens),model(tokens.contiguous().long()),rtol=0,atol=0)
    assert model(torch.zeros(0,3,dtype=torch.int64)).shape==(0,3,11)
    assert model(torch.zeros(1,1,dtype=torch.int64)).shape==(1,1,11)
    assert model(torch.zeros(1,8,dtype=torch.int64)).shape==(1,8,11)


@pytest.mark.parametrize(("field","value","error"),[("vocab_size",0,ValueError),("context_length",0,ValueError),("context_length",True,TypeError),("num_layers",0,ValueError),("num_layers",-1,ValueError),("num_layers",1.5,TypeError),("num_layers",True,TypeError)])
def test_lm_invalid_dimensions(field,value,error):
    with pytest.raises(error,match=field):
        _model(**{field:value})


@pytest.mark.parametrize(("tokens","error","message"),[([[1,2]],TypeError,"Tensor"),(torch.tensor([1,2]),ValueError,"shape"),(torch.zeros(1,0,dtype=torch.int64),ValueError,"positive"),(torch.zeros(1,9,dtype=torch.int64),ValueError,"context_length"),(torch.zeros(1,2),TypeError,"dtype"),(torch.ones(1,2,dtype=torch.bool),TypeError,"dtype"),(torch.tensor([[-1,1]]),ValueError,"between"),(torch.tensor([[11,1]]),ValueError,"between"),(torch.zeros(1,2,dtype=torch.int64,device="meta"),ValueError,"same device")])
def test_lm_invalid_token_inputs(tokens,error,message):
    with pytest.raises(error,match=message):
        _model()(tokens)


def test_lm_from_config_and_unsupported_ffn():
    from dataclasses import replace
    from pathlib import Path
    from tiny_lm.config import load_config
    config=load_config(Path(__file__).resolve().parents[1]/"configs"/"smoke.yaml").model
    model=TransformerLM.from_config(config,device="cpu",dtype=torch.float64)
    assert len(model.layers)==config.num_layers
    assert model.context_length==config.context_length
    assert model.token_embeddings.weight.shape==(config.vocab_size,config.d_model)
    assert model.layers[0].attn.rope.theta==config.rope_theta
    assert model.layers[0].ffn.d_ff==config.d_ff
    assert model.layers[0].norm_mode==config.norm_mode
    assert model.lm_head.weight.dtype==torch.float64
    assert model(torch.tensor([[1,2,3]])).shape==(1,3,config.vocab_size)
    post=TransformerLM.from_config(replace(config,norm_mode="post_norm"))
    assert post.layers[0].norm_mode=="post_norm"
    with pytest.raises(ValueError,match="GELU"):
        TransformerLM.from_config(replace(config,ffn_type="gelu"))
    with pytest.raises(TypeError,match="ModelConfig"):
        TransformerLM.from_config({})


def test_lm_loss_optimizer_and_sampling_integration():
    import numpy as np
    from tiny_lm.data import get_batch
    from tiny_lm.losses import cross_entropy
    from tiny_lm.optim import AdamW,clip_gradient_norm
    from tiny_lm.schedule import get_lr_cosine_schedule
    model=_model(dtype=torch.float32)
    optimizer=AdamW(model.parameters(),lr=0.01)
    x,y=get_batch(np.arange(33,dtype=np.int64)%11,2,4,"cpu",generator=torch.Generator().manual_seed(5))
    before=model.lm_head.weight.detach().clone()
    for step in range(2):
        optimizer.param_groups[0]["lr"]=get_lr_cosine_schedule(step,0.01,0.001,0,10)
        optimizer.zero_grad()
        loss=cross_entropy(model(x),y)
        assert torch.isfinite(loss)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
        clip_gradient_norm(model.parameters(),1.0)
        optimizer.step()
    assert not torch.equal(before,model.lm_head.weight)
    assert torch.isfinite(model(x)).all()
    assert all(state["step"]==2 for state in optimizer.state.values())
