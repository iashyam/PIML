#!/home/taylor/anaconda3/bin/python
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
			nn.Dropout(p=0.34),
			nn.Tanh(), 
			nn.Linear(20,32),
			nn.Tanh(), 
	        nn.Linear(32, 1)
	    ) 

	def forward(self, input):
		# assert input.shape[-1]==2, f"input tensor shape {input.shape} isn't correct!"
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


def curate_data(Ni, Nb, Nf, BF):

	xb = (torch.floor(torch.rand(Ni)*2)*2-1).unsqueeze(1)
	tb = torch.rand(Ni).unsqueeze(1)
	ub = torch.zeros(Ni).unsqueeze(1)
	xi = (torch.rand(Nb)*2-1).unsqueeze(1)
	ti = torch.zeros(Nb).unsqueeze(1)
	ui = BF(xi)
	
	xf = (torch.rand(Nf)*2-1).unsqueeze(1)
	tf = torch.rand(Nf).unsqueeze(1)
	
	return [xb, tb, ub, xi, ti, ui, xf, tf]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(Ni, Nb, Nf, BF, lamda: float=0.35 , EPOCHS=20):

	criterion = nn.MSELoss()
	history = {}
	history['loss'] = []
	history['f_loss'] = []
	history['u_loss'] = []
	model.train()

	for i in (bar := tqdm(range(EPOCHS))):
		running_loss = 0 
		running_f_loss = 0
		running_u_loss = 0 
		xb, tb, ub, xi, ti, ui, xf, tf =  curate_data(Ni, Nb, Nf, BF)
		# print(f"xb: {xb.shape}, tb: {tb.shape},ub: {ub.shape},xi: {xi.shape},ui: {ui.shape},xf: {xf.shape}, tf: {tf.shape}")
		optimizer.zero_grad()

		ub_hat = u(tb, xb)
		b_loss = criterion(ub_hat, ub)	
		ui_hat = u(ti, xi)
		i_loss = criterion(ui_hat, ui)	
		u_loss = b_loss + i_loss
		
		uf = f(xf, tf)
		f_loss = criterion(uf, torch.zeros_like(uf))
		loss = 2* lamda * u_loss + 2 * (1-lamda) * f_loss
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.item() 
		running_f_loss += f_loss.item()
		running_u_loss += u_loss.item()

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

	N = 250
	x = np.linspace(-1, 1, N)
	x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
	t_tensor = torch.tensor([t]*N, dtype=torch.float32).unsqueeze(-1)
	u_pred = predict(t_tensor, x_tensor).squeeze().numpy()
	u_exact = -np.sin(np.pi * x) * np.exp(-t * np.pi**2) # exact solution at t=0.25  
	plt.plot(x, u_pred, label='Predicted')
	# plt.plot(x, u_exact, label='Exact', linestyle='dashed')
	plt.xlabel('x')
	plt.ylabel('u(t,x)')
	plt.title(f'PINN Prediction vs Exact Solution at t={t}')
	plt.legend()
	plt.savefig(f"static/images/comprison at{t}.png", dpi=250)
	plt.show()

def plot_cmap(xu, tu):
	Nu = 1000
	xs = (torch.rand(Nu)*2-1)
	tu = torch.rand(Nu)

def plot_history(history):
	
	plt.plot(history['loss'], label='loss')
	plt.plot(history['f_loss'], label='f_loss')
	plt.plot(history['u_loss'], label='u_loss')
	plt.legend()
	plt.savefig("static/images/losses.png", dpi=250)
	plt.show()

if __name__=="__main__":
	
	BF = lambda x: -torch.sin(torch.pi*x)
	history = train(Ni=100, Nb=100, Nf=10000, BF=BF, lamda=0.50 , EPOCHS=5000)
	plot_history(history)
	plot_results(0.25)
	plot_results(0.75)
	plot_results(0.50)
