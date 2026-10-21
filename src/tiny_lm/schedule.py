"""
从 Assignment 1 迁移的线性预热与余弦学习率衰减。
"""

import math


def get_lr_cosine_schedule(it:int,max_learning_rate:float,min_learning_rate:float,warmup_iters:int,cosine_cycle_iters:int)->float:
    """
    返回本次更新的学习率；it 为更新前已完成的更新次数。
    cosine_cycle_iters 是衰减结束的绝对步数，必须大于 warmup_iters。
    零预热从最大学习率开始；到达结束步数后保持最小值。
    """
    for name,value in (("it",it),("warmup_iters",warmup_iters),("cosine_cycle_iters",cosine_cycle_iters)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value<0:
            raise ValueError(f"{name} must be nonnegative")
    if cosine_cycle_iters<=warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters")
    for name,value in (("max_learning_rate",max_learning_rate),("min_learning_rate",min_learning_rate)):
        if type(value) not in (int,float):
            raise TypeError(f"{name} must be a finite number")
        try:
            finite=math.isfinite(value)
        except OverflowError:
            finite=False
        if not finite:
            raise ValueError(f"{name} must be finite")
    if max_learning_rate<=0:
        raise ValueError("max_learning_rate must be positive")
    if min_learning_rate<0 or min_learning_rate>max_learning_rate:
        raise ValueError("min_learning_rate must be between 0 and max_learning_rate")

    if it<warmup_iters:
        # 先算比例，避免有限学习率乘步数时溢出。
        return max_learning_rate*(it/warmup_iters)
    if it==warmup_iters:
        return float(max_learning_rate)
    if it>=cosine_cycle_iters:
        return float(min_learning_rate)
    decay_ratio=(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_learning_rate+coeff*(max_learning_rate-min_learning_rate)
