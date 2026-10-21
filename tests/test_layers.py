"""
Check migrated layer initialization, forward values and gradients on CPU.
"""

import importlib
import math
import pytest

torch=pytest.importorskip("torch",reason="layer tests require the optional compute dependencies")

layers=importlib.import_module("tiny_lm.model.layers")
Embedding,Linear,RMSNorm,SwiGLU,silu_fn=layers.Embedding,layers.Linear,layers.RMSNorm,layers.SwiGLU,layers.silu_fn

F=torch.nn.functional
DTYPES=(torch.float16,torch.bfloat16,torch.float32,torch.float64)


def _close(actual,expected,dtype):
    rtol,atol={torch.float16:(5e-3,5e-4),torch.bfloat16:(4e-2,4e-3),torch.float32:(2e-5,2e-6),torch.float64:(1e-10,1e-12)}[dtype]
    torch.testing.assert_close(actual.double(),expected.double(),rtol=rtol,atol=atol)


def _fill_parameters(module):
    with torch.no_grad():
        for index,p in enumerate(module.parameters()):
            p.copy_(torch.linspace(-0.4,0.5,p.numel(),dtype=torch.float64).reshape(p.shape)+index*0.03)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("shape",[(4,),(3,4),(2,3,4)])
def test_linear_forward_and_gradients(dtype,shape):
    layer=Linear(4,5,dtype=dtype)
    _fill_parameters(layer)
    x=torch.linspace(-1.2,1.1,math.prod(shape),dtype=dtype).reshape(shape).requires_grad_()
    reference_x=x.detach().double().requires_grad_()
    reference_weight=layer.weight.detach().double().requires_grad_()
    actual=layer(x)
    expected=F.linear(reference_x,reference_weight)
    probe=torch.linspace(-0.7,0.9,actual.numel(),dtype=dtype).reshape(actual.shape)
    gradients=torch.autograd.grad(actual,(x,layer.weight),probe)
    reference_gradients=torch.autograd.grad(expected,(reference_x,reference_weight),probe.double())
    assert actual.shape==(*shape[:-1],5)
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for gradient,reference in zip(gradients,reference_gradients,strict=True):
        _close(gradient,reference,dtype)


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("index_dtype",[torch.int32,torch.int64])
def test_embedding_lookup_and_repeated_token_gradients(dtype,index_dtype):
    layer=Embedding(5,3,dtype=dtype)
    _fill_parameters(layer)
    ids=torch.tensor([[0,2,2],[4,0,2]],dtype=index_dtype)
    reference_weight=layer.weight.detach().double().requires_grad_()
    actual=layer(ids)
    expected=F.embedding(ids,reference_weight)
    probe=torch.linspace(-0.7,0.9,actual.numel(),dtype=dtype).reshape(actual.shape)
    gradient=torch.autograd.grad(actual,layer.weight,probe)[0]
    reference=torch.autograd.grad(expected,reference_weight,probe.double())[0]
    assert actual.shape==(2,3,3)
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    _close(gradient,reference,dtype)
    assert torch.count_nonzero(gradient[[1,3]])==0


@pytest.mark.parametrize("dtype",DTYPES)
@pytest.mark.parametrize("shape",[(4,),(3,4),(2,3,4)])
def test_rmsnorm_forward_and_gradients(dtype,shape):
    layer=RMSNorm(4,eps=0.03,dtype=dtype)
    _fill_parameters(layer)
    x=torch.linspace(-1.2,1.1,math.prod(shape),dtype=dtype).reshape(shape).requires_grad_()
    reference_x=x.detach().double().requires_grad_()
    reference_weight=layer.weight.detach().double().requires_grad_()
    actual=layer(x)
    expected=F.rms_norm(reference_x,(4,),reference_weight,eps=0.03)
    probe=torch.linspace(-0.7,0.9,actual.numel(),dtype=dtype).reshape(actual.shape)
    gradients=torch.autograd.grad(actual,(x,layer.weight),probe)
    reference_gradients=torch.autograd.grad(expected,(reference_x,reference_weight),probe.double())
    assert actual.shape==shape
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for gradient,reference in zip(gradients,reference_gradients,strict=True):
        _close(gradient,reference,dtype)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_rmsnorm_large_low_precision_input_is_finite(dtype):
    x=torch.tensor([[30000.0,-30000.0,10000.0,-10000.0]],dtype=dtype,requires_grad=True)
    layer=RMSNorm(4,dtype=torch.float32)
    reference_x=x.detach().double().requires_grad_()
    reference_weight=layer.weight.detach().double().requires_grad_()
    actual=layer(x)
    expected=F.rms_norm(reference_x,(4,),reference_weight,eps=layer.eps)
    probe=torch.tensor([[0.5,1.0,-1.0,0.25]],dtype=dtype)
    gradients=torch.autograd.grad(actual,(x,layer.weight),probe)
    references=torch.autograd.grad(expected,(reference_x,reference_weight),probe.double())
    assert actual.dtype==dtype
    assert torch.isfinite(actual).all()
    _close(actual,expected,dtype)
    for gradient,reference in zip(gradients,references,strict=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient.double(),reference.double(),rtol=4*torch.finfo(dtype).eps,atol=6e-8)


def test_rmsnorm_float64_input_is_not_downcast():
    x=torch.tensor([[1e100,-2e100,3e100,-4e100]],dtype=torch.float64,requires_grad=True)
    layer=RMSNorm(4,dtype=torch.float64)
    actual=layer(x)
    expected=F.rms_norm(x,(4,),layer.weight,eps=layer.eps)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def test_rmsnorm_zero_input():
    layer=RMSNorm(4,dtype=torch.float64)
    x=torch.zeros((2,4),dtype=torch.float64,requires_grad=True)
    result=layer(x)
    gradient=torch.autograd.grad(result.sum(),x)[0]
    assert torch.count_nonzero(result)==0
    torch.testing.assert_close(gradient,torch.full_like(x,1/math.sqrt(layer.eps)))


@pytest.mark.parametrize("dtype",DTYPES)
def test_silu_forward_and_gradients(dtype):
    x=torch.tensor([-100.0,-3.0,-0.5,0.0,0.7,4.0,100.0],dtype=dtype,requires_grad=True)
    reference_x=x.detach().double().requires_grad_()
    actual=silu_fn(x)
    expected=F.silu(reference_x)
    gradient=torch.autograd.grad(actual.sum(),x)[0]
    reference=torch.autograd.grad(expected.sum(),reference_x)[0]
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    _close(gradient,reference,dtype)


@pytest.mark.parametrize("dtype",DTYPES)
def test_swiglu_forward_and_all_parameter_gradients(dtype):
    layer=SwiGLU(4,6,dtype=dtype)
    _fill_parameters(layer)
    x=torch.linspace(-0.8,0.9,24,dtype=dtype).reshape(2,3,4).requires_grad_()
    reference_x=x.detach().double().requires_grad_()
    parameters=tuple(layer.parameters())
    reference_weights={name:p.detach().double().requires_grad_() for name,p in layer.named_parameters()}
    actual=layer(x)
    gate=F.silu(F.linear(reference_x,reference_weights["w1.weight"]))
    sign=F.linear(reference_x,reference_weights["w3.weight"])
    expected=F.linear(gate*sign,reference_weights["w2.weight"])
    probe=torch.linspace(-0.7,0.9,actual.numel(),dtype=dtype).reshape(actual.shape)
    gradients=torch.autograd.grad(actual,(x,*parameters),probe)
    references=torch.autograd.grad(expected,(reference_x,*reference_weights.values()),probe.double())
    assert actual.shape==x.shape
    assert actual.dtype==dtype
    _close(actual,expected,dtype)
    for gradient,reference in zip(gradients,references,strict=True):
        _close(gradient,reference,dtype)


@pytest.mark.parametrize("factory",[lambda:Linear(3,2,dtype=torch.float64),lambda:RMSNorm(3,dtype=torch.float64),lambda:SwiGLU(3,4,dtype=torch.float64)])
def test_float64_parameter_and_input_gradcheck(factory):
    layer=factory()
    _fill_parameters(layer)
    x=torch.tensor([[0.2,-0.7,1.1]],dtype=torch.float64,requires_grad=True)
    names,parameters=zip(*layer.named_parameters(),strict=True)

    def forward(value,*weights):
        return torch.func.functional_call(layer,dict(zip(names,weights,strict=True)),(value,))

    assert torch.autograd.gradcheck(forward,(x,*parameters))


def test_silu_float64_gradcheck():
    x=torch.tensor([-3.0,-0.2,0.0,0.8,4.0],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(silu_fn,(x,))


@pytest.mark.parametrize("dtype",DTYPES)
def test_initialization_shapes_bounds_and_parameter_names(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        linear=Linear(4,6,device="cpu",dtype=dtype)
        embedding=Embedding(7,4,device="cpu",dtype=dtype)
        norm=RMSNorm(4,device="cpu",dtype=dtype)
        ffn=SwiGLU(4,6,device="cpu",dtype=dtype)
    assert linear.weight.shape==(6,4)
    assert embedding.weight.shape==(7,4)
    assert set(linear.state_dict())==set(embedding.state_dict())==set(norm.state_dict())=={"weight"}
    assert set(ffn.state_dict())=={"w1.weight","w3.weight","w2.weight"}
    bound=torch.tensor(3*math.sqrt(2/10),dtype=dtype).item()
    assert linear.weight.abs().max().item()<=bound
    assert embedding.weight.abs().max().item()<=3
    torch.testing.assert_close(norm.weight,torch.ones(4,dtype=dtype))
    for layer in (linear,embedding,norm,ffn):
        for parameter in layer.parameters():
            assert parameter.dtype==dtype
            assert parameter.device.type=="cpu"
            assert parameter.requires_grad
            assert torch.isfinite(parameter).all()


def test_state_dict_roundtrip():
    original=SwiGLU(4,6,dtype=torch.float64)
    restored=SwiGLU(4,6,dtype=torch.float64)
    restored.load_state_dict(original.state_dict())
    x=torch.arange(8,dtype=torch.float64).reshape(2,4)
    torch.testing.assert_close(restored(x),original(x),rtol=0,atol=0)


@pytest.mark.parametrize("kind",["linear","norm","ffn","silu"])
def test_noncontiguous_and_empty_inputs(kind):
    layer={"linear":lambda:Linear(4,4),"norm":lambda:RMSNorm(4),"ffn":lambda:SwiGLU(4,6),"silu":lambda:silu_fn}[kind]()
    x=torch.arange(24,dtype=torch.float32).reshape(2,4,3).transpose(1,2)
    assert not x.is_contiguous()
    torch.testing.assert_close(layer(x),layer(x.contiguous()))
    assert layer(torch.empty((0,4))).shape==(0,4)


def test_embedding_scalar_empty_and_noncontiguous_ids():
    layer=Embedding(5,3)
    torch.testing.assert_close(layer(torch.tensor(2)),layer.weight[2])
    assert layer(torch.empty((0,2),dtype=torch.int64)).shape==(0,2,3)
    ids=torch.tensor([[1,2,3],[3,2,1]]).T
    assert not ids.is_contiguous()
    torch.testing.assert_close(layer(ids),layer(ids.contiguous()))


@pytest.mark.parametrize("factory",[lambda value:Linear(value,3),lambda value:Linear(3,value),lambda value:Embedding(value,3),lambda value:Embedding(3,value),lambda value:RMSNorm(value),lambda value:SwiGLU(value,3),lambda value:SwiGLU(3,value)])
@pytest.mark.parametrize("value,error",[(0,ValueError),(-1,ValueError),(True,TypeError),(2.5,TypeError)])
def test_invalid_dimensions(factory,value,error):
    with pytest.raises(error,match="must be"):
        factory(value)


@pytest.mark.parametrize("factory",[lambda dtype:Linear(2,3,dtype=dtype),lambda dtype:Embedding(3,2,dtype=dtype),lambda dtype:RMSNorm(2,dtype=dtype),lambda dtype:SwiGLU(2,3,dtype=dtype)])
@pytest.mark.parametrize("dtype",[torch.int64,torch.bool,torch.complex64])
def test_invalid_parameter_dtype(factory,dtype):
    with pytest.raises(TypeError,match="dtype"):
        factory(dtype)


@pytest.mark.parametrize("eps,error",[(0,ValueError),(-1,ValueError),(math.nan,ValueError),(math.inf,ValueError),(10**400,ValueError),(True,TypeError),("0.1",TypeError)])
def test_invalid_rmsnorm_eps(eps,error):
    with pytest.raises(error,match="eps"):
        RMSNorm(4,eps=eps)


@pytest.mark.parametrize("factory",[lambda:Linear(4,3),lambda:RMSNorm(4),lambda:SwiGLU(4,6)])
@pytest.mark.parametrize("shape",[(),(3,),(2,3)])
def test_incorrect_last_dimension(factory,shape):
    with pytest.raises(ValueError,match="last dimension"):
        factory()(torch.zeros(shape))


@pytest.mark.parametrize("factory",[lambda:Linear(4,3),lambda:RMSNorm(4),lambda:SwiGLU(4,6),lambda:silu_fn])
@pytest.mark.parametrize("value,error",[([1.0]*4,TypeError),(torch.ones(4,dtype=torch.int64),TypeError),(torch.ones(4,dtype=torch.complex64),TypeError),(torch.ones(4).to_sparse(),ValueError)])
def test_invalid_float_inputs(factory,value,error):
    with pytest.raises(error):
        factory()(value)


@pytest.mark.parametrize("value,error",[([-1],TypeError),(torch.tensor([-1]),ValueError),(torch.tensor([5]),ValueError),(torch.tensor([0.0]),TypeError),(torch.tensor([True]),TypeError),(torch.tensor([1],dtype=torch.int16),TypeError),(torch.tensor([1]).to_sparse(),ValueError)])
def test_invalid_embedding_ids(value,error):
    with pytest.raises(error,match="token_ids"):
        Embedding(5,3)(value)


def test_embedding_device_mismatch():
    with pytest.raises(ValueError,match="same device"):
        Embedding(5,3)(torch.empty(1,dtype=torch.int64,device="meta"))
