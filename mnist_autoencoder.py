import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

torch.manual_seed(0)

class StudentAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),  
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)
        out_flat = self.decoder(z)
        out = out_flat.view(-1, 1, 28, 28)
        return out, z

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # data
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = StudentAE(latent_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = F.mse_loss(recon, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        avg_loss = running / len(train_loader)
        print(f"epoch {epoch} avg train mse: {avg_loss:.6f}")

        # quick test loss 
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, _ = model(x)
                test_loss += F.mse_loss(recon, x).item()
        print(f"         avg test mse: {test_loss / len(test_loader):.6f}")

    # save reconstructions
    os.makedirs("outputs", exist_ok=True)
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        recon, _ = model(x)

        # put originals + recons in one grid
        n = 16
        originals = x[:n].cpu()
        recons = recon[:n].cpu()
        both = torch.cat([originals, recons], dim=0)

        grid = utils.make_grid(both, nrow=n, pad_value=1.0)
        out_path = "outputs/mnist_recon_grid.png"
        utils.save_image(grid, out_path)
        print("saved:", out_path)
        print("top row = originals, bottom row = reconstructions")

if __name__ == "__main__":
    main()
