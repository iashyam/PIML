import torch 
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import os

debug = os.getenv("DEBUG", 0)

#dummy funtion
def f(x):
    return x**2

N = 100

x = torch.linspace(-2*pi, 2*pi, N, requires_grad=True)
# x = torch.tensor([np.pi, np.pi/2],  requires_grad=True)
y = f(x)

external_grad = torch.ones(x.shape[0])
y.backward(gradient=external_grad)
a = x.grad

if debug:
    print(f"a: {a} \n x: {x} \n y:  {y}")
# print(torch.autograd.grad(y, x))

plt.plot(x.detach().numpy(), y.detach().numpy(), label="f(x)")
plt.plot(x.detach().numpy(), a.detach().numpy(), label="f'(x)")
plt.legend()
plt.show()