# import torch
# from torch.distributions import Normal
# from gymnasium import spaces
# import numpy as np

# a=np.array([1,2,3,4])
# b=np.array([1,2,3])
# print(b+a[0:3])
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("\logs")
for i in range(100):
    writer.add_scalar("y=x",i,i)
for i in range(100):
    writer.add_scalar("y=x",2*i,i)
writer.close()