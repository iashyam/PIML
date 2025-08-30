import torch 
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import os

debug = os.getenv("DEBUG", 0)

#dummy funtion
def f(x):
    return torch.sin(x)  

N = 100

x = torch.linspace(-2*pi, 2*pi, N, requires_grad=True)
# x = torch.tensor([np.pi, np.pi/2],  requires_grad=True)
y = f(x)

y_x = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
y_xx = torch.autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0]

if debug:
    print(f"a: {a} \n x: {x} \n y:  {y}")
# print(torch.autograd.grad(y, x))

plt.plot(x.detach().numpy(), y.detach().numpy(), label="f(x)")
plt.plot(x.detach().numpy(), y_x.detach().numpy(), label="f'(x)")
plt.plot(x.detach().numpy(), y_xx.detach().numpy(), label="f''(x)")
plt.legend()
plt.show()