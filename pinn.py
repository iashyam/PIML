## this is a simple implemention of the PINNs. 
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
import os
from tqdm import tqdm
## setting command line variables
gpu = os.getenv("GPU", 0)
debug = os.getenv("DEBUG", 0)

class Model(nn.Module):

	def __init__(self):
	    super().__init__()
	    self.net = nn.Sequential(
	        nn.Linear(2,20),
	        nn.Tanh(), 
			nn.Linear(20,20),
			nn.Tanh(), 
			nn.Linear(20,20),
			nn.Tanh(), 
			nn.Linear(20,20),
			nn.Tanh(), 
			nn.Linear(20,20),
			nn.Tanh(), 
			nn.Linear(20,20),
			nn.Tanh(), 
			nn.Linear(20,20),
			nn.Tanh(), 
			nn.Linear(20,32),
			nn.Tanh(), 
	        nn.Linear(32, 1)
	    ) 

	def forward(self, input):

	    return self.net(input)
	
model = Model()
def u(t, x):
	input = torch.cat([x, t], dim=-1)
	return model(input)

def f(t, x):
	print(t.requires_grad)
	print(x.requires_grad)
	u_ = u(t, x)	
	u_t = grad(u_, t, torch.ones_like(u_), create_graph=True)[0]
	u_x = grad(u_, x, torch.ones_like(u_), create_graph=True)[0]
	u_xx = grad(u_x, x, torch.ones_like(u_), create_graph=True)[0]
	return u_t + u_ * u_x - (0.1/torch.pi) * u_xx


class myDataset(Dataset):
	def __init__(self, t_u, x_u, u_, t_f, x_f):
	    self.t_u = t_u
	    self.x_u = x_u
	    self.u_ = u_
	    self.t_f = t_f
	    self.x_f = x_f

	def __len__(self):
	    return len(self.t_u)

	def __getitem__(self, idx):
	    return self.t_u[idx].unsqueeze(0), self.x_u[idx].unsqueeze(0), self.u_[idx].unsqueeze(0), self.t_f[idx].unsqueeze(0), self.x_f[idx].unsqueeze(0)

def train(dataloader, model=model, lamda: float=0.35,  optimizer=None, EPOCHS=20):
	criterion = nn.MSELoss()
	for i in (bar := tqdm(range(EPOCHS))):
		running_loss = 0.0
		for tu, xu, u_, tf, xf in dataloader:
			model.train()
			u_pred = u(tu, xu)
			f_pred = f(tf, xf)
			assert u_pred.shape == u_.shape, f"u_pred shape {u_pred.shape}, u shape {u_.shape}!"
			u_loss = criterion(u_pred, u_)
			f_loss = criterion(f_pred, torch.zeros_like(f_pred))
			loss = (lamda) * u_loss +(1-lamda) *  f_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			running_loss /= len(dataloader)

		bar.set_description(f"Epoch {i}, loss: {loss.item():.6f}")


def predict(t, x):
	model.eval()
	with torch.no_grad():
	    return u(t, x)
	
def plot_results():
	import matplotlib.pyplot as plt
	import numpy as np
	x = np.linspace(-1, 1, 256)
	t = 0.25
	x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
	t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(-1).repeat(1, x_tensor.shape[0]).T
	u_pred = predict(t_tensor, x_tensor).squeeze().numpy()
	u_exact = -np.sin(np.pi * x) * np.exp(-0.25 * np.pi**2) # exact solution at t=0.25  
	plt.plot(x, u_pred, label='Predicted')
	plt.plot(x, u_exact, label='Exact', linestyle='dashed')
	plt.xlabel('x')
	plt.ylabel('u(t,x)')
	plt.title('PINN Prediction vs Exact Solution at t=0.25')
	plt.legend()
	plt.show()

if __name__=="__main__":
	
	#put the intials and boundary conditions here
	Nu = 1000
	Nf = 10000
	x_u_b = torch.floor(torch.rand(Nu, requires_grad=True)*2)*2-1
	t_u_b = torch.rand(Nu, requires_grad=True)
	u_b = torch.zeros(Nu, requires_grad=True)
	x_u_i = torch.floor(torch.rand(Nu, requires_grad=True)*2)*2-1
	t_u_i = torch.zeros(Nu, requires_grad=True)
	u_i = -torch.sin(torch.pi*x_u_i)
	x_u = torch.cat([x_u_b, x_u_i], dim=0)
	t_u = torch.cat([t_u_b, t_u_i], dim=0)
	u_ = torch.cat([u_b, u_i], dim=0)

	x_f = torch.rand(Nf, requires_grad=True)*2-1
	t_f = torch.rand(Nf, requires_grad=True)

	print(f"x_u: {x_u.shape}, t_u: {t_u.shape}, u: {u_.shape}, x_f: {x_f.shape}, t_f: {t_f.shape}")

	dataset = myDataset(t_u, x_u, u_, t_f, x_f)
	dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	train(dataloader, model,0.25,optimizer, EPOCHS=40)
	torch.save(model.state_dict(), "pinn.pth")

	plot_results()

