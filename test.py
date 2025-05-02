import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

dummy_input = torch.rand(16, 4, 4)
conv = nn.Conv2d(16, 256, (2,2))
conv2 = nn.Conv2d(256, 512, (2,2))
x = conv(dummy_input)
x = conv2(x)
x = x.flatten()
print(x.shape)

