"""Numerical and state regressions for the original Assignment 1 update rule."""

import copy
import io
import math

import pytest

torch=pytest.importorskip("torch",reason="optimizer tests require compute dependencies")

from tiny_lm.optim import AdamW,clip_gradient_norm  # noqa: E402,I001


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64,torch.float16,torch.bfloat16])
def test_multistep_matches_scalar_float64_reference(dtype):
    p=torch.nn.Parameter(torch.tensor([0.25,-1.5,2.0],dtype=dtype))
    opt=AdamW([p],lr=0.03,betas=(0.7,0.8),eps=0.05,weight_decay=0.2)
    means=[0.0]*3
    variances=[0.0]*3
    for t in range(1,9):
        p.grad=torch.tensor([0.125*t,-0.25,0.001],dtype=dtype)
        before=p.detach().double().tolist()
        expected=[]
        for i,g in enumerate(p.grad.double().tolist()):
            means[i]=0.7*means[i]+0.3*g
            variances[i]=0.8*variances[i]+0.2*g*g
            alpha=0.03*math.sqrt(1-0.8**t)/(1-0.7**t)
            expected.append(
                (before[i]-alpha*means[i]/(math.sqrt(variances[i])+0.05))*(1-0.03*0.2)
            )
        opt.step()
        torch.testing.assert_close(p,torch.tensor(expected,dtype=dtype))
        state=opt.state[p]
        expected_dtype=torch.float64 if dtype==torch.float64 else torch.float32
        assert state["exp_avg"].dtype==expected_dtype
        assert state["exp_avg_sq"].dtype==expected_dtype
        torch.testing.assert_close(
            state["exp_avg"],torch.tensor(means,dtype=expected_dtype)
        )
        torch.testing.assert_close(
            state["exp_avg_sq"],torch.tensor(variances,dtype=expected_dtype)
        )


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_low_precision_extreme_gradients_remain_finite(dtype):
    p=torch.nn.Parameter(torch.tensor([1.0,1.0],dtype=dtype))
    opt=AdamW([p],weight_decay=0)
    p.grad=torch.tensor([60000.0,1e-7],dtype=dtype)
    opt.step()
    assert torch.isfinite(p).all()
    assert torch.isfinite(opt.state[p]["exp_avg_sq"]).all()
    assert opt.state[p]["exp_avg_sq"][1]>0


def test_none_gradient_zero_gradient_and_groups():
    skipped=torch.nn.Parameter(torch.tensor([2.0],dtype=torch.float64))
    decayed=torch.nn.Parameter(skipped.detach().clone())
    frozen=torch.nn.Parameter(skipped.detach().clone())
    opt=AdamW([
        {"params":[skipped,decayed],"lr":0.1,"weight_decay":0.5},
        {"params":[frozen],"lr":0.0},
    ])
    decayed.grad=torch.zeros_like(decayed)
    frozen.grad=torch.ones_like(frozen)
    opt.step()
    assert skipped.item()==2.0
    assert skipped not in opt.state
    assert decayed.item()==pytest.approx(1.9)
    assert frozen.item()==2.0
    assert opt.state[decayed]["step"]==1


def test_closure_enables_grad_and_returns_loss():
    p=torch.nn.Parameter(torch.tensor([2.0]))
    opt=AdamW([p])
    def closure():
        opt.zero_grad()
        loss=p.square().sum()
        loss.backward()
        return loss
    loss=opt.step(closure)
    assert loss.item()==4.0
    assert p.item()<2.0
    assert p.is_leaf


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64,torch.float16,torch.bfloat16])
def test_serialized_resume_is_identical(dtype):
    p=torch.nn.Parameter(torch.tensor([0.25,-1.5],dtype=dtype))
    opt=AdamW([p],betas=(0.8,0.9))
    for i in range(3):
        p.grad=torch.tensor([0.12345+i,-0.4567],dtype=dtype)
        opt.step()
    buffer=io.BytesIO()
    torch.save({"parameter":p.detach(),"optimizer":opt.state_dict()},buffer)
    buffer.seek(0)
    saved=torch.load(buffer,weights_only=True)
    restored=torch.nn.Parameter(saved["parameter"])
    resumed=AdamW([restored])
    resumed.load_state_dict(saved["optimizer"])
    snapshot=copy.deepcopy(saved["optimizer"])
    for i in range(4):
        p.grad=torch.tensor([0.23456+i,-0.34567],dtype=dtype)
        restored.grad=p.grad.clone()
        opt.step()
        resumed.step()
        torch.testing.assert_close(p,restored,rtol=0,atol=0)
        for name in ("exp_avg","exp_avg_sq"):
            torch.testing.assert_close(opt.state[p][name],resumed.state[restored][name],
                                       rtol=0,atol=0)
        assert opt.state[p]["step"]==resumed.state[restored]["step"]
    for name in ("exp_avg","exp_avg_sq"):
        torch.testing.assert_close(
            saved["optimizer"]["state"][0][name],snapshot["state"][0][name],rtol=0,atol=0
        )


@pytest.mark.parametrize(("options","message"),[
    ({"lr":-1},"lr"),({"lr":float("nan")},"lr"),({"lr":float("inf")},"lr"),
    ({"lr":True},"lr"),({"weight_decay":-0.1},"weight_decay"),
    ({"weight_decay":float("inf")},"weight_decay"),({"eps":0},"eps"),
    ({"eps":float("nan")},"eps"),({"betas":(0.9,)},"betas"),
    ({"betas":(1,0.9)},"betas"),({"betas":(0.9,-1)},"betas"),
    ({"betas":(0.9,float("nan"))},"betas"),
])
def test_invalid_hyperparameters_and_group_overrides(options,message):
    p=torch.nn.Parameter(torch.ones(1))
    with pytest.raises((TypeError,ValueError),match=message):
        AdamW([p],**options)
    with pytest.raises((TypeError,ValueError),match=message):
        AdamW([{"params":[p],**options}])


def test_later_invalid_group_fails_before_any_update():
    p=torch.nn.Parameter(torch.ones(1))
    q=torch.nn.Parameter(torch.ones(1))
    opt=AdamW([{"params":[p]},{"params":[q]}])
    p.grad=torch.ones_like(p)
    q.grad=torch.ones_like(q)
    opt.param_groups[1]["lr"]=float("nan")
    with pytest.raises(ValueError,match="lr"):
        opt.step()
    assert p.item()==1
    assert not opt.state
    with pytest.raises(ValueError,match="weight_decay"):
        opt.add_param_group({"params":[torch.nn.Parameter(torch.ones(1))],"weight_decay":-1})


def test_sparse_gradient_and_complex_parameter_rejected():
    p=torch.nn.Parameter(torch.ones(2))
    with torch.sparse.check_sparse_tensor_invariants():
        p.grad=torch.sparse_coo_tensor([[0]],[1.0],(2,),check_invariants=True)
    opt=AdamW([p])
    with pytest.raises(ValueError,match="dense"):
        opt.step()
    with pytest.raises(ValueError,match="dense"):
        clip_gradient_norm([p],1)
    with pytest.raises(TypeError,match="dtype"):
        AdamW([torch.nn.Parameter(torch.ones(1,dtype=torch.complex64))])


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("max_norm",[0.0,2.0,5.0,10.0])
def test_clipping_matches_global_float64_reference(dtype,max_norm):
    p=torch.nn.Parameter(torch.zeros(2,dtype=dtype))
    q=torch.nn.Parameter(torch.zeros(1,dtype=dtype))
    p.grad=torch.tensor([3.0,0.0],dtype=dtype)
    q.grad=torch.tensor([4.0],dtype=dtype)
    result=clip_gradient_norm(iter([p,q]),max_norm)
    scale=max_norm/(5+1e-6) if max_norm<5 else 1
    torch.testing.assert_close(p.grad,torch.tensor([3*scale,0],dtype=dtype))
    torch.testing.assert_close(q.grad,torch.tensor([4*scale],dtype=dtype))
    assert result is None


def test_float16_clipping_avoids_norm_overflow():
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float16))
    p.grad=torch.full_like(p,60000)
    clip_gradient_norm([p],1)
    torch.testing.assert_close(p.grad,torch.full_like(p,1/math.sqrt(2)))


def test_clipping_noncontiguous_and_no_gradient():
    p=torch.nn.Parameter(torch.zeros(2,3).t())
    p.grad=torch.full_like(p,2)
    assert not p.grad.is_contiguous()
    clip_gradient_norm([p],1)
    assert torch.linalg.vector_norm(p.grad).item()==pytest.approx(1,abs=1e-6)
    assert clip_gradient_norm([],1) is None
    assert clip_gradient_norm([torch.nn.Parameter(torch.ones(1))],1) is None


@pytest.mark.parametrize("max_norm",[-1,float("nan"),float("inf"),True])
def test_clipping_invalid_limit(max_norm):
    with pytest.raises((TypeError,ValueError),match="max_norm"):
        clip_gradient_norm([],max_norm)


@pytest.mark.parametrize("value",[float("nan"),float("inf")])
def test_clipping_nonfinite_norm_rejected_before_mutation(value):
    p=torch.nn.Parameter(torch.zeros(2))
    p.grad=torch.tensor([1,value])
    with pytest.raises(ValueError,match="finite"):
        clip_gradient_norm([p],1)
    assert p.grad[0].item()==1
