import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(2, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 1),
                                 )
        
    def forward(self, input):
        assert input.shape[-1] == 2, "Input's last dimension must be 2"
        return self.net(input)

class myDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.cat([self.x[idx], self.y[idx]], dim=0), self.z[idx]



def training_loop(model, dataloader, epochs=100, optimizer=None, device="cpu"):
    history = {}
    history['loss'] = []

    for epoch in (bar := tqdm(range(epochs))):
        running_loss = 0.0
        model.train()
        model.to(device)
        for batch_idx, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        running_loss /= (batch_idx + 1)
        bar.set_description(f"Epoch {epoch}, loss: {running_loss:.6f}")
        history['loss'].append(running_loss)
    
    return history

def compare_plots(model, orginal_function, ndots: int=100):
    x = torch.linspace(0, 1, ndots).unsqueeze(1)
    t = torch.linspace(0, 1, ndots).unsqueeze(1)
    figure = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X = X.view(100, 100, 1)
    T = T.view(100, 100, 1)
    out = model(torch.cat([X, T], dim=2))
    plt.pcolormesh(out.view(ndots, ndots).detach().numpy(), cmap='rainbow')
    plt.axis('off')
    plt.axis('equal')
    plt.title("PINN Prediction")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(orginal_function(X, T).view(ndots, ndots).detach().numpy(), cmap='rainbow')
    plt.axis('off')
    plt.axis('equal')
    plt.title("Original Function")
    plt.show()

if __name__=="__main__":

    def original_function(x, y):
        return  torch.exp(-((x - 0.5)**2 + (y - 0.5)**2)/0.1)

    x = torch.rand(10000).unsqueeze(1)
    t = torch.rand(10000).unsqueeze(1)
    z = original_function(x, t)

    dataset = myDataset(x, t, z)
    chunksize = 100
    dataloader = DataLoader(dataset, batch_size=chunksize, shuffle=True)
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = training_loop(model, dataloader, epochs=50, optimizer=optimizer)
    compare_plots(model, original_function)

