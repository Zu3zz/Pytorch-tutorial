# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019-08-07 19:42
# @File   : 01-basic-data-type.py
"""
python vs pytorch
Int     IntTensor
float   FloatTensor
Int array   IntTensor of size
pytorch don't support string
"""
import torch
import numpy as np
torch.tensor([1.1])
# tensor([1.1000])
torch.tensor([1.1,2.2])
# tensor([1.100,2,200)]

# 接收的是维度 random初始化
torch.FloatTensor(1)
# tensor([-0.4337])
torch.FloatTensor(2)
# tensor([4.4842e-44, 1.6255e-43])

data = np.ones(2)
# array([1., 1.])
torch.from_numpy(data)
# tensor([1., 1.], dtype=torch.float64)


# for a 2 dim vector
a = torch.randn(2,3)
# tensor([[-0.1,0.5,1,1],
#         [-2,0.2,1.2]])
a.shape
# torch.Size([2,3])
list(a.shape)
# [2,3]
a.size()
# 2
a.size(1)
# 3
a.shape[1]
# 3

a = torch.randn(2,3,28,28)
a.numel()
# 4780
a.dim()
# 4