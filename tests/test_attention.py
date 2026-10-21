"""
Check attention numerics, gradients, masking, positions and cache precision on CPU.
"""

import importlib
import math
import pytest

torch=pytest.importorskip("torch",reason="attention tests require the optional compute dependencies")
attention=importlib.import_module("tiny_lm.model.attention")
RotaryPositionalEmbedding=attention.RotaryPositionalEmbedding
CausalSelfAttention=attention.CausalSelfAttention
softmax=attention.softmax
scaled_dot_product_attention=attention.scaled_dot_product_attention
F=torch.nn.functional
DTYPES=(torch.float16,torch.bfloat16,torch.float32,torch.float64)


def _close(actual,expected,dtype):
    rtol,atol={torch.float16:(5e-3,5e-4),torch.bfloat16:(4e-2,4e-3),torch.float32:(2e-5,2e-6),torch.float64:(1e-10,1e-12)}[dtype]
    torch.testing.assert_close(actual.double(),expected.double(),rtol=rtol,atol=atol)


def _rope_reference(x,positions,theta):
    freq=torch.tensor([theta**(-2*i/x.shape[-1]) for i in range(x.shape[-1]//2)],dtype=torch.float64)
    phase=positions.double().unsqueeze(-1)*freq
    if x.ndim>=4 and positions.shape==(*x.shape[:-3],x.shape[-2]):
        phase=phase.unsqueeze(-3)
    rotation=torch.polar(torch.ones_like(phase),phase)
    pairs=x.double().reshape(*x.shape[:-1],x.shape[-1]//2,2).contiguous()
    return torch.view_as_real(torch.view_as_complex(pairs)*rotation).flatten(-2)


def _sdpa_reference(q,k,v,mask=None):
    scores=(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1])
    if mask is not None:
        scores=scores.masked_fill(~mask,float('-inf'))
    return F.softmax(scores,dim=-1)@v


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("dim",[-1,0,1])
def test_softmax_values_and_gradients(dtype,dim):
    x=torch.linspace(-4,3,24,dtype=dtype).reshape(2,3,4).requires_grad_()
    reference=x.detach().double().requires_grad_()
    actual=softmax(x,dim)
    expected=F.softmax(reference,dim=dim)
    probe=torch.linspace(-0.7,0.9,24,dtype=dtype).reshape(x.shape)
    gradient=torch.autograd.grad(actual,x,probe)[0]
    reference_gradient=torch.autograd.grad(expected,reference,probe.double())[0]
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    _close(gradient,reference_gradient,dtype)


def test_softmax_float16_large_axis_does_not_sum_to_zero():
    x=torch.zeros(1,100000,dtype=torch.float16,requires_grad=True)
    actual=softmax(x)
    assert actual.dtype==torch.float16
    torch.testing.assert_close(actual,F.softmax(x,dim=-1),rtol=0,atol=0)
    assert actual.float().sum().item()==pytest.approx(1.0,abs=0.002)
    assert torch.isfinite(torch.autograd.grad(actual[:,0].sum(),x)[0]).all()


def test_softmax_allows_partial_negative_infinity():
    x=torch.tensor([[1.0,float('-inf'),3.0]],requires_grad=True)
    actual=softmax(x)
    torch.testing.assert_close(actual,F.softmax(x,dim=-1))
    assert actual[0,1].item()==0
    assert torch.autograd.grad(actual[0,0],x)[0][0,1].item()==0


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("masked",[False,True])
def test_sdpa_forward_and_gradients_with_batch_broadcast(dtype,masked):
    q=torch.linspace(-0.8,1.0,24,dtype=dtype).reshape(2,1,3,4).requires_grad_()
    k=torch.linspace(-0.5,0.7,40,dtype=dtype).reshape(1,2,5,4).requires_grad_()
    v=torch.linspace(-0.9,0.6,60,dtype=dtype).reshape(1,2,5,6).requires_grad_()
    references=[value.detach().double().requires_grad_() for value in (q,k,v)]
    mask=torch.tensor([[True,False,True,False,True],[True,True,False,False,True],[False,True,True,True,False]]) if masked else None
    actual=scaled_dot_product_attention(q,k,v,mask)
    expected=_sdpa_reference(*references,mask)
    probe=torch.linspace(-0.7,0.9,actual.numel(),dtype=dtype).reshape(actual.shape)
    gradients=torch.autograd.grad(actual,(q,k,v),probe)
    reference_gradients=torch.autograd.grad(expected,references,probe.double())
    assert actual.shape==(2,2,3,6)
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for left,right in zip(gradients,reference_gradients,strict=True):
        _close(left,right,dtype)


def test_v_can_expand_the_batch_dimensions():
    q=torch.arange(6,dtype=torch.float64).reshape(2,3)
    k=torch.arange(12,dtype=torch.float64).reshape(4,3)
    v=torch.arange(40,dtype=torch.float64).reshape(5,4,2)
    mask=torch.ones(5,2,4,dtype=torch.bool)
    mask[0,:,3]=False
    actual=scaled_dot_product_attention(q,k,v,mask)
    expected=torch.stack([_sdpa_reference(q,k,v[i],mask[i]) for i in range(5)])
    torch.testing.assert_close(actual,expected)


def test_fp16_dot_product_overflow_regression():
    q=torch.full((1,2,4),300.0,dtype=torch.float16,requires_grad=True)
    k=q.detach().clone().requires_grad_()
    v=torch.tensor([[[1.0,2.0],[3.0,4.0]]],dtype=torch.float16,requires_grad=True)
    actual=scaled_dot_product_attention(q,k,v)
    expected=_sdpa_reference(q.double(),k.double(),v.double())
    assert torch.isfinite(actual).all()
    _close(actual,expected,q.dtype)
    for gradient in torch.autograd.grad(actual.sum(),(q,k,v)):
        assert torch.isfinite(gradient).all()


def test_cpu_autocast_does_not_downcast_attention_core():
    q=torch.linspace(-0.9,1.0,24).reshape(2,3,4)
    expected=scaled_dot_product_attention(q,q,q)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        actual=scaled_dot_product_attention(q,q,q)
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_masked_values_do_not_affect_output_or_receive_gradients():
    q=torch.ones(2,3,dtype=torch.float64,requires_grad=True)
    k=torch.ones(4,3,dtype=torch.float64,requires_grad=True)
    v=torch.arange(8,dtype=torch.float64).reshape(4,2).requires_grad_()
    mask=torch.tensor([True,False,False,False])
    result=scaled_dot_product_attention(q,k,v,mask)
    torch.testing.assert_close(result,v[0].expand(2,2))
    gradients=torch.autograd.grad(result.sum(),(q,k,v))
    assert torch.count_nonzero(gradients[1][1:])==0
    assert torch.count_nonzero(gradients[2][1:])==0


@pytest.mark.parametrize("shape",[(3,4),(2,3,4),(2,2,3,4),(2,3,2,3,4)])
@pytest.mark.parametrize("dtype",DTYPES)
def test_rope_shared_positions_forward_and_gradients(shape,dtype):
    x=torch.linspace(-0.8,0.9,math.prod(shape),dtype=dtype).reshape(shape).requires_grad_()
    positions=torch.tensor([0,2,7])
    rope=RotaryPositionalEmbedding(10000.0,4,8)
    reference_x=x.detach().double().requires_grad_()
    actual=rope(x,positions)
    expected=_rope_reference(reference_x,positions,10000.0)
    probe=torch.linspace(-0.7,0.9,x.numel(),dtype=dtype).reshape(x.shape)
    gradient=torch.autograd.grad(actual,x,probe)[0]
    reference_gradient=torch.autograd.grad(expected,reference_x,probe.double())[0]
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    _close(gradient,reference_gradient,dtype)
    _close(actual.double().square().sum(-1),x.double().square().sum(-1),dtype)


@pytest.mark.parametrize("with_extra_batch",[False,True])
def test_rope_per_batch_positions_across_heads(with_extra_batch):
    shape=(2,3,2,3,4) if with_extra_batch else (2,2,3,4)
    x=torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)/100
    positions=torch.arange(math.prod(shape[:-3])*3,dtype=torch.int32).reshape(*shape[:-3],3)
    rope=RotaryPositionalEmbedding(12345.0,4,32)
    actual=rope(x,positions)
    expected=torch.empty_like(x)
    flat_x=x.reshape(-1,2,3,4)
    flat_positions=positions.reshape(-1,3)
    for batch_index in range(flat_x.shape[0]):
        for head_index in range(2):
            expected.reshape(-1,2,3,4)[batch_index,head_index]=_rope_reference(flat_x[batch_index,head_index],flat_positions[batch_index],12345.0)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def test_rope_full_position_shape():
    x=torch.arange(48,dtype=torch.float64).reshape(2,2,3,4)/100
    positions=torch.arange(12).reshape(2,2,3)
    rope=RotaryPositionalEmbedding(10000.0,4,12)
    torch.testing.assert_close(rope(x,positions),_rope_reference(x,positions,10000.0))


@pytest.mark.parametrize("conversion",["direct","half_double","bfloat_double","half_float_double"])
def test_rope_cache_rebuilds_without_losing_float64_precision(conversion):
    rope=RotaryPositionalEmbedding(12345.0,8,4096)
    if conversion=="half_double":
        rope.half().double()
    elif conversion=="bfloat_double":
        rope.bfloat16().double()
    elif conversion=="half_float_double":
        rope.half().float()
        rope(torch.ones(3,8),torch.tensor([0,2049,4095]))
        rope.double()
    x=torch.linspace(-1.1,0.8,24,dtype=torch.float64).reshape(3,8)
    positions=torch.tensor([0,2049,4095])
    expected=_rope_reference(x,positions,12345.0)
    torch.testing.assert_close(rope(x,positions),expected,rtol=1e-11,atol=1e-12)
    assert rope.cos_cached.dtype==rope.sin_cached.dtype==torch.float64
    assert rope.state_dict()=={}


def test_rope_cache_switches_between_float64_and_float32():
    rope=RotaryPositionalEmbedding(10000.0,4,8)
    positions=torch.tensor([1,3])
    x=torch.linspace(-1,1,8,dtype=torch.float64).reshape(2,4)
    first=rope(x,positions)
    rope(x.float(),positions)
    assert rope.cos_cached.dtype==torch.float32
    torch.testing.assert_close(rope(x,positions),first,rtol=0,atol=0)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("use_rope",[False,True])
def test_causal_attention_values_and_all_gradients(use_rope,dtype):
    layer=CausalSelfAttention(8,2,max_seq_len=12 if use_rope else None,theta=10000.0 if use_rope else None,dtype=dtype)
    with torch.no_grad():
        for index,p in enumerate(layer.parameters()):
            p.copy_(torch.linspace(-0.2,0.3,p.numel(),dtype=torch.float64).reshape(p.shape)+index*0.01)
    x=torch.linspace(-0.7,0.8,48,dtype=dtype).reshape(2,3,8).requires_grad_()
    positions=torch.tensor([[0,2,4],[3,5,7]]) if use_rope else None
    reference_x=x.detach().double().requires_grad_()
    weights={name:p.detach().double().requires_grad_() for name,p in layer.named_parameters()}
    q,k,v=[F.linear(reference_x,weights[name+".weight"]).reshape(2,3,2,4).transpose(1,2) for name in ("q_proj","k_proj","v_proj")]
    if use_rope:
        q=_rope_reference(q,positions,10000.0)
        k=_rope_reference(k,positions,10000.0)
    mask=torch.ones(3,3,dtype=torch.bool).tril()
    merged=_sdpa_reference(q,k,v,mask).transpose(1,2).reshape(2,3,8)
    expected=F.linear(merged,weights["output_proj.weight"])
    actual=layer(x,positions)
    probe=torch.linspace(-0.7,0.9,48,dtype=dtype).reshape(x.shape)
    gradients=torch.autograd.grad(actual,(x,*layer.parameters()),probe)
    reference_gradients=torch.autograd.grad(expected,(reference_x,*weights.values()),probe.double())
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for left,right in zip(gradients,reference_gradients,strict=True):
        _close(left,right,dtype)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("use_rope",[False,True])
def test_causality_and_default_positions(dtype,use_rope):
    layer=CausalSelfAttention(8,2,max_seq_len=8 if use_rope else None,theta=10000.0 if use_rope else None,dtype=dtype)
    x=torch.linspace(-0.5,0.5,64,dtype=dtype).reshape(2,4,8)
    actual=layer(x)
    changed=x.clone()
    changed[:,2:]+=2
    modified=layer(changed)
    assert actual.dtype==dtype
    torch.testing.assert_close(actual[:,:2],modified[:,:2],rtol=0,atol=0)
    assert torch.isfinite(actual).all()
    if use_rope:
        torch.testing.assert_close(actual,layer(x,torch.arange(4)),rtol=0,atol=0)


def test_attention_state_roundtrip_and_odd_head_dimension_without_rope():
    original=CausalSelfAttention(6,2,dtype=torch.float64)
    restored=CausalSelfAttention(6,2,dtype=torch.float64)
    assert set(original.state_dict())=={"q_proj.weight","k_proj.weight","v_proj.weight","output_proj.weight"}
    restored.load_state_dict(original.state_dict())
    x=torch.arange(18,dtype=torch.float64).reshape(1,3,6)/10
    torch.testing.assert_close(restored(x),original(x),rtol=0,atol=0)


def test_noncontiguous_inputs_and_empty_batch():
    q=torch.arange(24,dtype=torch.float64).reshape(2,4,3).transpose(1,2)/10
    assert not q.is_contiguous()
    torch.testing.assert_close(scaled_dot_product_attention(q,q,q),_sdpa_reference(q,q,q))
    torch.testing.assert_close(softmax(q),F.softmax(q,dim=-1))
    layer=CausalSelfAttention(4,2,dtype=torch.float64,max_seq_len=4,theta=10000.0)
    torch.testing.assert_close(layer(q),layer(q.contiguous()))
    assert layer(torch.empty(0,3,4,dtype=torch.float64)).shape==(0,3,4)


def test_float64_finite_difference_gradients():
    x=torch.tensor([[0.2,-0.3,0.7,-0.4]],dtype=torch.float64,requires_grad=True)
    rope=RotaryPositionalEmbedding(10000.0,4,4)
    assert torch.autograd.gradcheck(lambda value:rope(value,torch.tensor([2])),(x,))
    assert torch.autograd.gradcheck(softmax,(x,))
    q=x.detach().reshape(2,2).requires_grad_()
    k=(q.detach()+0.1).requires_grad_()
    v=(q.detach()-0.2).requires_grad_()
    mask=torch.tensor([[True,False],[True,True]])
    assert torch.autograd.gradcheck(lambda a,b,c:scaled_dot_product_attention(a,b,c,mask),(q,k,v))


@pytest.mark.parametrize("use_rope",[False,True])
def test_causal_input_and_parameter_finite_difference_gradients(use_rope):
    layer=CausalSelfAttention(4,2,max_seq_len=4 if use_rope else None,theta=10000.0 if use_rope else None,dtype=torch.float64)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.linspace(-0.2,0.3,p.numel(),dtype=torch.float64).reshape(p.shape))
    names,parameters=zip(*layer.named_parameters(),strict=True)
    x=torch.linspace(-0.4,0.6,8,dtype=torch.float64).reshape(1,2,4).requires_grad_()

    def forward(value,*weights):
        return torch.func.functional_call(layer,dict(zip(names,weights,strict=True)),(value,))

    assert torch.autograd.gradcheck(forward,(x,*parameters))


@pytest.mark.parametrize("values",[[float('-inf'),float('-inf')],[0.0,float('nan')],[0.0,float('inf')]])
def test_softmax_rejects_undefined_rows(values):
    with pytest.raises(ValueError,match="finite maximum"):
        softmax(torch.tensor([values]))


@pytest.mark.parametrize("dim,error",[(True,TypeError),(1.5,TypeError),(2,ValueError),(-3,ValueError)])
def test_softmax_invalid_dim(dim,error):
    with pytest.raises(error,match="dim"):
        softmax(torch.zeros(2,3),dim)


def test_softmax_rejects_empty_axis():
    with pytest.raises(ValueError,match="nonempty"):
        softmax(torch.empty(2,0))


@pytest.mark.parametrize("value,error",[([1.0,2.0],TypeError),(torch.ones(2,dtype=torch.int64),TypeError),(torch.ones(2,dtype=torch.complex64),TypeError),(torch.ones(2).to_sparse(),ValueError),(torch.tensor(1.0),ValueError)])
def test_invalid_softmax_inputs(value,error):
    with pytest.raises(error):
        softmax(value)


@pytest.mark.parametrize("mask,error,message",[(False,TypeError,"mask"),(torch.ones(2,2),TypeError,"bool"),(torch.zeros(2,2,dtype=torch.bool),ValueError,"at least one key"),(torch.tensor([[True,False],[False,False]]),ValueError,"at least one key"),(torch.ones(3,3,dtype=torch.bool),ValueError,"broadcast"),(torch.ones(2,2,dtype=torch.bool,device="meta"),ValueError,"same device"),(torch.ones(2,2,dtype=torch.bool).to_sparse(),ValueError,"dense")])
def test_invalid_masks(mask,error,message):
    q=torch.ones(2,4)
    with pytest.raises(error,match=message):
        scaled_dot_product_attention(q,q,q,mask)


@pytest.mark.parametrize("shapes,message",[(((2,3),(2,4),(2,4)),"feature"),(((2,4),(3,4),(2,4)),"sequence"),(((2,2,4),(3,2,4),(3,2,4)),"broadcastable"),(((0,4),(2,4),(2,4)),"positive"),(((2,0),(2,0),(2,4)),"positive"),(((2,4),(0,4),(0,4)),"positive")])
def test_sdpa_invalid_shapes(shapes,message):
    with pytest.raises(ValueError,match=message):
        scaled_dot_product_attention(*(torch.ones(shape) for shape in shapes))


def test_sdpa_invalid_dtype_and_device():
    q=torch.ones(2,4)
    with pytest.raises(ValueError,match="same dtype"):
        scaled_dot_product_attention(q,q.double(),q)
    with pytest.raises(ValueError,match="same device"):
        scaled_dot_product_attention(q,q,q.to("meta"))
    with pytest.raises(TypeError,match="Q"):
        scaled_dot_product_attention(None,q,q)


@pytest.mark.parametrize("kwargs,error,message",[({"theta":True},TypeError,"theta"),({"theta":0},ValueError,"theta"),({"theta":float('nan')},ValueError,"theta"),({"theta":float('inf')},ValueError,"theta"),({"d_k":0},ValueError,"d_k"),({"d_k":3},ValueError,"even"),({"d_k":True},TypeError,"d_k"),({"max_len":0},ValueError,"max_len"),({"max_len":2.5},TypeError,"max_len")])
def test_invalid_rope_configuration(kwargs,error,message):
    args=dict(theta=10000.0,d_k=4,max_len=8)
    args.update(kwargs)
    with pytest.raises(error,match=message):
        RotaryPositionalEmbedding(**args)


@pytest.mark.parametrize("positions,error,message",[([-1,0,1],TypeError,"torch.Tensor"),(torch.tensor([-1,0,1]),ValueError,"between"),(torch.tensor([0,1,8]),ValueError,"between"),(torch.ones(3),TypeError,"dtype"),(torch.ones(3,dtype=torch.bool),TypeError,"dtype"),(torch.zeros(2,dtype=torch.int64),ValueError,"match"),(torch.zeros(1,3,dtype=torch.int64),ValueError,"match"),(torch.zeros(3,dtype=torch.int64,device="meta"),ValueError,"same device")])
def test_invalid_rope_positions(positions,error,message):
    rope=RotaryPositionalEmbedding(10000.0,4,8)
    with pytest.raises(error,match=message):
        rope(torch.ones(2,2,3,4),positions)


def test_invalid_rope_input_shape_and_device():
    rope=RotaryPositionalEmbedding(10000.0,4,8)
    with pytest.raises(ValueError,match="last dimension"):
        rope(torch.ones(3,6),torch.arange(3))
    with pytest.raises(ValueError,match="positive"):
        rope(torch.ones(0,4),torch.empty(0,dtype=torch.int64))
    with pytest.raises(ValueError,match="same device"):
        rope(torch.ones(3,4,device="meta"),torch.arange(3))


@pytest.mark.parametrize("kwargs,error,message",[({"d_model":0},ValueError,"d_model"),({"num_heads":0},ValueError,"num_heads"),({"d_model":True},TypeError,"d_model"),({"num_heads":3},ValueError,"divisible"),({"bias":True},ValueError,"bias=True"),({"bias":1},TypeError,"bool"),({"theta":10000.0},ValueError,"together"),({"max_seq_len":8},ValueError,"together"),({"d_model":6,"theta":10000.0,"max_seq_len":8},ValueError,"even")])
def test_invalid_causal_configuration(kwargs,error,message):
    args=dict(d_model=8,num_heads=2)
    args.update(kwargs)
    with pytest.raises(error,match=message):
        CausalSelfAttention(**args)


def test_invalid_causal_inputs():
    plain=CausalSelfAttention(4,2)
    rope=CausalSelfAttention(4,2,max_seq_len=4,theta=10000.0)
    with pytest.raises(ValueError,match="shape"):
        plain(torch.ones(2,3,5))
    with pytest.raises(ValueError,match="shape"):
        plain(torch.ones(1,2,3,4))
    with pytest.raises(ValueError,match="positive"):
        plain(torch.ones(2,0,4))
    with pytest.raises(ValueError,match="require RoPE"):
        plain(torch.ones(2,3,4),torch.arange(3))
    with pytest.raises(ValueError,match="max_seq_len"):
        rope(torch.ones(2,5,4))
    with pytest.raises(ValueError,match="same device"):
        plain(torch.ones(2,3,4,device="meta"))
