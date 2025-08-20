import torch 
import numpy as np


#dummy funtion
def f(x):
    return torch.sin(x)


x = torch.tensor([np.pi],  requires_grad=True)
y = torch.sin(x)


print(x)
y.backward()
a = x.grad
a.backward()
print(a.grad)

