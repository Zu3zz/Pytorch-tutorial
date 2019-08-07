# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019-08-07 00:55
# @File   : 01_Compare_Gpu_and_Cpu.py
import torch
import time

# to make sure your gpu is avaliable
print(torch.cuda.is_available())

# test difference between gpu & cpu
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

# second time in gpu is much more faster
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
"""output
cpu 0.5823187828063965 tensor(138473.7656)
cuda:0 0.7159512042999268 tensor(141391.2500, device='cuda:0')
cuda:0 0.0005238056182861328 tensor(141391.2500, device='cuda:0')
"""