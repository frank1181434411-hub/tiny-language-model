"""
Check schedule boundaries, invalid inputs and reproducible step lookup.
"""

import math

import pytest

from tiny_lm.schedule import get_lr_cosine_schedule


@pytest.mark.parametrize(
    ("it","expected"),
    [(0,0.0),(1,0.25),(3,0.75),(4,1.0),(6,0.6),(8,0.2),(9,0.2),(1000,0.2)],
)
def test_schedule_landmarks(it,expected):
    actual=get_lr_cosine_schedule(it,1.0,0.2,4,8)
    assert actual==pytest.approx(expected)


@pytest.mark.parametrize("warmup",[0,1,4])
def test_curve_bounds_and_monotonicity(warmup):
    values=[get_lr_cosine_schedule(it,0.01,0.001,warmup,12) for it in range(16)]
    assert all(0<=value<=0.01 for value in values)
    assert values[warmup]==0.01
    assert values[12:]==[0.001]*4
    assert values[:warmup+1]==sorted(values[:warmup+1])
    assert values[warmup:]==sorted(values[warmup:],reverse=True)


def test_zero_warmup():
    assert get_lr_cosine_schedule(0,1.0,0.2,0,8)==1.0
    assert get_lr_cosine_schedule(4,1.0,0.2,0,8)==pytest.approx(0.6)
    assert get_lr_cosine_schedule(8,1.0,0.2,0,8)==0.2


def test_shortest_decay():
    assert get_lr_cosine_schedule(0,1.0,0.0,0,1)==1.0
    assert get_lr_cosine_schedule(1,1.0,0.0,0,1)==0.0


def test_equal_learning_rates_still_warm_up():
    values=[get_lr_cosine_schedule(it,0.5,0.5,2,4) for it in range(6)]
    assert values==[0.0,0.25,0.5,0.5,0.5,0.5]


def test_minimum_applies_to_decay_not_warmup():
    assert get_lr_cosine_schedule(0,1.0,0.5,4,8)==0.0
    assert get_lr_cosine_schedule(1,1.0,0.5,4,8)==0.25


def test_resumed_step_lookup_matches_uninterrupted_schedule():
    expected=[get_lr_cosine_schedule(it,0.01,0.001,4,12) for it in range(16)]
    for it in [12,0,8,4,15,8]:
        assert get_lr_cosine_schedule(it,0.01,0.001,4,12)==expected[it]
    resumed=[get_lr_cosine_schedule(it,0.01,0.001,4,12) for it in range(7,16)]
    assert resumed==expected[7:]


def test_large_finite_learning_rate_does_not_overflow_during_warmup():
    actual=get_lr_cosine_schedule(2,1e308,0.0,4,8)
    assert math.isfinite(actual)
    assert actual==pytest.approx(5e307)


@pytest.mark.parametrize("it",[0,1,4,8,9])
def test_integer_learning_rates_return_float(it):
    assert isinstance(get_lr_cosine_schedule(it,1,0,4,8),float)


@pytest.mark.parametrize("name",["it","warmup_iters","cosine_cycle_iters"])
@pytest.mark.parametrize("value",[True,False,1.5,"2",None])
def test_invalid_step_types(name,value):
    args=dict(it=1,max_learning_rate=1.0,min_learning_rate=0.2,
              warmup_iters=4,cosine_cycle_iters=8)
    args[name]=value
    with pytest.raises(TypeError,match=name):
        get_lr_cosine_schedule(**args)


@pytest.mark.parametrize("name",["it","warmup_iters","cosine_cycle_iters"])
def test_negative_steps(name):
    args=dict(it=1,max_learning_rate=1.0,min_learning_rate=0.2,
              warmup_iters=4,cosine_cycle_iters=8)
    args[name]=-1
    with pytest.raises(ValueError,match=name):
        get_lr_cosine_schedule(**args)


@pytest.mark.parametrize(("warmup","end"),[(0,0),(4,4),(5,4)])
@pytest.mark.parametrize("it",[0,4,20])
def test_invalid_stage_order_is_rejected_before_branching(warmup,end,it):
    with pytest.raises(ValueError,match="cosine_cycle_iters"):
        get_lr_cosine_schedule(it,1.0,0.2,warmup,end)


@pytest.mark.parametrize("name",["max_learning_rate","min_learning_rate"])
@pytest.mark.parametrize("value",[True,False,"0.1",None])
def test_invalid_learning_rate_types(name,value):
    args=dict(it=1,max_learning_rate=1.0,min_learning_rate=0.2,
              warmup_iters=4,cosine_cycle_iters=8)
    args[name]=value
    with pytest.raises(TypeError,match=name):
        get_lr_cosine_schedule(**args)


@pytest.mark.parametrize("name",["max_learning_rate","min_learning_rate"])
@pytest.mark.parametrize("value",[math.nan,math.inf,-math.inf,10**400])
def test_nonfinite_learning_rates(name,value):
    args=dict(it=1,max_learning_rate=1.0,min_learning_rate=0.2,
              warmup_iters=4,cosine_cycle_iters=8)
    args[name]=value
    with pytest.raises(ValueError,match=name):
        get_lr_cosine_schedule(**args)


@pytest.mark.parametrize(
    ("maximum","minimum","message"),
    [(0.0,0.0,"max_learning_rate"),(-1.0,0.0,"max_learning_rate"),
     (1.0,-0.1,"min_learning_rate"),(1.0,1.1,"min_learning_rate")],
)
def test_invalid_learning_rate_ranges(maximum,minimum,message):
    with pytest.raises(ValueError,match=message):
        get_lr_cosine_schedule(1,maximum,minimum,4,8)
