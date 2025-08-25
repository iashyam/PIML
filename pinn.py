## this is a simple implemention of the PINNs. 
import torch 
from torch import nn
import os
from tqdm import tqdm
import auto_diffrentiation 
## setting command line variables
gpu = os.getenv("GPU", 0)
debug = os.getenv("DEBUG", 0)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(), 
            nn.Linear(32, 1)
        ) 

    def forwad(self, x, t):
        return self.linear(x, t)
    
## variables
model = Model()
EPOCHS = 12

def train(us, xs, ts, model=model, optimizer=None, EPOCHS=EPOCHS):
    for i in (bar := tqdm(range(EPOCHS))):
        for x,u, t in zip(xs, us, ts):
            u_hat = model(x, t)

            u_hat.backward()
            u_x = x.grad
            u_t = t.grad
            u_xx = u_x.grad

            loss = (u - u_hat)**2 + 0.1* (u_t + u_xx)**2

            loss.backward()
            optimizer.step()

            bar.set_description(f"Epoch: {i}, loss: {loss.item}") 


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        print(f"Using device {device}")