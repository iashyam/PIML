import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
import os
from tqdm import tqdm
## setting command line variables
gpu = os.getenv("GPU", 0)
debug = os.getenv("DEBUG", 0)
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

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
	t.requires_grad=True
	x.requires_grad=True
	u_ = u(t, x)	
	u_t = grad(u_, t, torch.ones_like(u_), create_graph=True, allow_unused=True)[0]
	u_x = grad(u_, x, torch.ones_like(u_), create_graph=True, allow_unused=True)[0]
	u_xx = grad(u_x, x, torch.ones_like(u_), create_graph=True, allow_unused=True)[0]
	return u_t + u_ * u_x - (0.01/torch.pi) * u_xx


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
		return (
			self.t_u[idx].unsqueeze(0),
			self.x_u[idx].unsqueeze(0),
			self.u_[idx].unsqueeze(0),
			self.t_f[idx].unsqueeze(0),
			self.x_f[idx].unsqueeze(0)
			)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train(dataloader, lamda: float=0.35 , EPOCHS=20):
	criterion = nn.MSELoss()
	history = {}
	history['loss'] = []
	history['f_loss'] = []
	history['u_loss'] = []
	model.train()
	k = 0 
	for i in (bar := tqdm(range(EPOCHS))):
		running_loss = 0 
		running_f_loss = 0
		running_u_loss = 0 
		for tu, xu, u_, tf, xf in dataloader:
			optimizer.zero_grad()
			if debug==3: print(xu.shape, tu.shape)
			input_tensor = torch.cat([tu, xu], dim=-1)
			u_pred = model(input_tensor)
			u_loss = criterion(u_pred, u_)
			f_pred = f(tf, xf)
			assert u_pred.shape == u_.shape, f"u_pred shape {u_pred.shape}, u shape {u_.shape}!"
			f_loss = criterion(f_pred, torch.zeros_like(f_pred))
			# loss = (lamda)*u_loss + (1-lamda)*f_loss
			loss = u_loss + f_loss
			loss.backward()
			optimizer.step()
			# running_loss /= len(dataloader
			running_loss += loss.item()
			running_f_loss += u_loss.item()
			running_u_loss += f_loss.item()
		bar.set_description(f"epoch {i}, loss: {running_loss:.4f}, f_loss: {running_f_loss:.4f} u_loss: {running_u_loss:.4f}")
		history['loss'].append(running_loss)
		history['f_loss'].append(running_f_loss)
		history['u_loss'].append(running_u_loss)

	return history

def predict(t, x):
	model.eval()
	with torch.no_grad():
	    return u(t, x)
	
def plot_results(t: float):
	x = np.linspace(-1, 1, 256)
	x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
	t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(-1).repeat(1, x_tensor.shape[0]).T
	u_pred = predict(t_tensor, x_tensor).squeeze().numpy()
	u_exact = -np.sin(np.pi * x) * np.exp(-t * np.pi**2) # exact solution at t=0.25  
	plt.plot(x, u_pred, label='Predicted')
	plt.plot(x, u_exact, label='Exact', linestyle='dashed')
	plt.xlabel('x')
	plt.ylabel('u(t,x)')
	plt.title(f'PINN Prediction vs Exact Solution at t={t}')
	plt.legend()
	plt.savefig("comprison.png", dpi=250)
	plt.show()

def plot_history(history):
	
	plt.plot(history['loss'], label='loss')
	plt.plot(history['f_loss'], label='f_loss')
	plt.plot(history['u_loss'], label='u_loss')
	plt.legend()
	plt.savefig("losses.png", dpi=250)
	plt.show()

if __name__=="__main__":
	
	#put the intials and boundary conditions here
	Nu = 100
	Nf = 10000 
	x_u_b = torch.floor(torch.rand(Nu)*2)*2-1
	t_u_b = torch.rand(Nu)
	u_b = torch.zeros(Nu)
	x_u_i = torch.rand(Nu)*2-1
	t_u_i = torch.zeros(Nu)
	u_i = -torch.sin(torch.pi*x_u_i)
	x_u = torch.cat([x_u_b, x_u_i], dim=0)
	t_u = torch.cat([t_u_b, t_u_i], dim=0)
	u_ = torch.cat([u_b, u_i], dim=0)

	x_f = torch.rand(Nf)*2-1
	t_f = torch.rand(Nf)

	dataset = myDataset(t_u, x_u, u_, t_f, x_f)
	dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
	history = train(dataloader,0.5, EPOCHS=1000)
	torch.save(model.state_dict(), "pinn.pth")

	plot_results(0.5)
	plot_results(0.75)
	plot_history(history)
