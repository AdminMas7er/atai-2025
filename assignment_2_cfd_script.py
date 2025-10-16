
# # Assignment 2 for 2AMM40


import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import animation

# for cleaner look (set your theme to dark mode)
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1E1E1E'
plt.rcParams['axes.facecolor'] = '#1E1E1E'
plt.rcParams['savefig.facecolor'] = '#1E1E1E'




################# quickfix for Snellius GPU MIG usage
import os 
import subprocess
if "MIG" in subprocess.check_output(["nvidia-smi", "-L"], text=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# this box fuses the loose data files such that they can be read by the dataset object
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "CFD" / "grid"
base_in = DATA_ROOT / "loose"
base_out2 = DATA_ROOT / "concat"
base_out2.mkdir(parents=True, exist_ok=True)
# Fuse loose files -> concat
for Re in [100, 150, 200, 250, 300, 350, 400]:
    u = np.load(base_in / f"u_grid_Re{Re}.npy")
    v = np.load(base_in / f"v_grid_Re{Re}.npy")
    p = np.load(base_in / f"p_grid_Re{Re}.npy")
    concat = np.stack([u, v, p], axis=1)
    np.save(base_out2 / f"uvp_grid_Re{Re}.npy", concat)


class FlowDataset(Dataset):
    def __init__(self, filenames, flip_augmentation=False, timesample=1, use_coords=True, norm_mean=None, norm_std=None):
        
        
        self.sequences = []
        self.index_map = []
        self.flip_augmentation = flip_augmentation
        self.use_coords = use_coords

        # coordinates
        self.coordsy = np.linspace(-5, 5, 64, endpoint=True)
        self.coordsx = np.linspace(-10, 10, 128, endpoint=True)
        self.coords = np.array(np.meshgrid(self.coordsx, self.coordsy)).T.reshape(128, 64, 2)
        self.coords = torch.tensor(self.coords, dtype=torch.float32).permute(2, 1, 0)

        # use coordinates to make obstacle mask
        center = torch.tensor([-5.0, 0.0]).view(2, 1, 1)
        radius = 0.5
        squared_distance = ((self.coords - center) ** 2).sum(dim=0) 
        self.mask = squared_distance < radius**2  # shape [64, 128]
        self.mask = self.mask.unsqueeze(0).float()
        self.norm_mean = None if norm_mean is None else torch.tensor(norm_mean, dtype=torch.float32)
        self.norm_std  = None if norm_std  is None else torch.tensor(norm_std,  dtype=torch.float32)

        # sample/read the data
        for seq_idx, filename in enumerate(filenames):
            data = np.load(filename)  
            data = data[::timesample] 
            self.sequences.append(data)
            T = data.shape[0]
            self.index_map.extend([(seq_idx, t) for t in range(T - 1)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, t = self.index_map[idx]
        seq = self.sequences[seq_idx]
        input = seq[t]    
        target_frame=seq[t + 1]
        # if flip augmentation is true then flip the data horizontally 50% of the time
        if self.flip_augmentation and np.random.rand() > 0.5:
            input = self.flip(input)
            target_frame = self.flip(target_frame)
        input_tensor=torch.tensor(input, dtype=torch.float32)
        target_tensor = torch.tensor(target_frame, dtype=torch.float32)
        if self.use_coords:
            input_tensor=torch.cat([input_tensor, self.coords], axis=0)
        if self.norm_mean is not None:
            m = self.norm_mean[:, None, None]
            s = self.norm_std[:, None, None].clamp_min(1e-6)
            input_tensor[:3] = (input_tensor[:3] - m) / s
            target_tensor = (target_tensor - m) / s

        return (self.mask, input_tensor, target_tensor)
        
    def get_trajectory(self, seq_idx):
        # returns full trajectory
        seq = self.sequences[seq_idx]
        return (
            self.mask.unsqueeze(0), 
            self.coords.unsqueeze(0), 
            torch.tensor(seq, dtype=torch.float32)
        )

    def flip(self, x):
        x = np.flip(x, axis=2).copy()
        x[1] *= -1
        return x

datafolder = base_out2
train_files = [datafolder / f for f in [
    'uvp_grid_Re100.npy', 'uvp_grid_Re200.npy', 'uvp_grid_Re300.npy', 'uvp_grid_Re400.npy'
]]
val_files = [datafolder / f for f in [
    'uvp_grid_Re150.npy', 'uvp_grid_Re250.npy', 'uvp_grid_Re350.npy'
]]

def _compute_uvp_stats(files, timesample=1):
    ms, vs = [], []
    for f in files:
        x = np.load(f)[::timesample].astype(np.float32)  # (T,3,H,W)
        ms.append(x.mean(axis=(0, 2, 3)))
        vs.append(x.var(axis=(0, 2, 3)))
    mean = np.mean(ms, axis=0)
    std = np.sqrt(np.mean(vs, axis=0))
    std = np.maximum(std, 1e-6)
    return mean, std


dt = 10 # only sample every dt timesteps
train_mean, train_std = _compute_uvp_stats([str(p) for p in train_files], timesample=dt)
batch_size = 64
use_coordinates=True
train_dataset = FlowDataset(train_files, flip_augmentation=False, timesample=dt, use_coords=use_coordinates, norm_mean=train_mean, norm_std=train_std)
val_dataset = FlowDataset(val_files, flip_augmentation=False, timesample=dt, use_coords=use_coordinates, norm_mean=train_mean, norm_std=train_std)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)


class ELBO_Loss(torch.nn.Module):
    def __init__(self, L,beta=1.0):
        """
        Helper class to compute the ELBO loss for the AR-LVM model

        Args:
            L: size of the periodic boundary conditions
        """
        super(ELBO_Loss, self).__init__()
        self.L = L
        self.beta = beta

    def forward(self, y_pred, y_true, mask=None):
        enc_dist, prior_dist, predicted_frame = y_pred
        # Properly scaled KL (average over batch, channels, H, W)
        kl = torch.distributions.kl.kl_divergence(enc_dist, prior_dist).mean()
        # Masked reconstruction (exclude obstacle; mask==1 inside obstacle)
        if mask is not None:
            fluid = (1.0 - mask)  # (B,1,H,W)
            diff2 = (predicted_frame - y_true).pow(2)  # (B,C,H,W)
            # Broadcast fluid to channels
            recon = (diff2 * fluid).sum() / (fluid.sum() * diff2.shape[1] + 1e-6)
        else:
            recon = torch.nn.functional.mse_loss(predicted_frame, y_true, reduction='mean')

        return self.beta * kl + recon
        

class Encoder(torch.nn.Module):
    def __init__(self, num_layers=2, processor=None, emb_dim=64, latent_dim=16, nonlinearity = torch.nn.functional.relu):
        """
        Encoder for the AR-LVM model, takes in the node embeddings and returns a distribution over the latent space

        Args:
            num_layers: number of message passing layers
            processor: Processor model
            emb_dim: dimension of the node embeddings
            latent_dim: dimension of the latent space
            nonlinearity: activation function
        """
        super(Encoder, self).__init__()
        self.processor = processor
        self.mp_layers = torch.nn.ModuleList([torch.nn.Conv2d(emb_dim*2, emb_dim*2, kernel_size=3, padding='same') for _ in range(num_layers)])
        self.fc_mu = torch.nn.Conv2d(emb_dim*2, latent_dim, kernel_size=1)
        self.fc_sigma = torch.nn.Conv2d(emb_dim*2, latent_dim, kernel_size=1)
        self.nonlinearity = nonlinearity

    def forward(self, h, h_next): 
        """
        Args:
            h: node embeddings
            x_initial: initial state
            x_next: next state
            coords: coordinates of the nodes
            mask: mask for the obstacle

        Returns:
            torch.Distribution: Normal distribution over the latent space
        """

        h = torch.cat([h, h_next], dim=1)
        # Apply message passing layers
        for mp in self.mp_layers:
            h = mp(h)
            h = self.nonlinearity(h)
        
        # Compute the mean and sigma
        mu = self.fc_mu(h)          # Create one z per node
        sigma = self.fc_sigma(h)    # Create one z per node
        sigma = torch.nn.functional.softplus(sigma) + 1e-6
        # Create a Normal distribution with mu and sigma
        enc_dist = torch.distributions.Normal(mu, sigma)
        return enc_dist
        
    
class Processor(torch.nn.Module):
    def __init__(self, num_layers=2, in_dim=4, emb_dim=64, nonlinearity = torch.nn.functional.relu):
        """
        Forward model, takes in the input data and returns the node embeddings

        Args:
            num_layers: number of message passing layers
            in_dim: input dimension
            emb_dim: dimension of the node embeddings
            nonlinearity: activation function
        """
        super(Processor, self).__init__()
        self.in_emb = torch.nn.Conv2d(in_dim, emb_dim, kernel_size=1)
        self.mp_layers = torch.nn.ModuleList([torch.nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding='same') for _ in range(num_layers)])
        self.nonlinearity = nonlinearity


    def forward(self, x_grid): #update this
        """
        Forward pass of the Processor

        Args:
            x: input data
            edge_index: adjacency list
            edge_attr: edge attributes

        Returns:
            torch.Tensor: node embeddings
        """
        h = self.in_emb(x_grid)
        h = self.nonlinearity(h)

        for mp in self.mp_layers:
            h = mp(h)
            h = self.nonlinearity(h)

        return h

    
class Decoder(torch.nn.Module):
    def __init__(self, num_layers=2, latent_dim=16, emb_dim=64, out_dim=4, nonlinearity = torch.nn.functional.relu, sigma = 0.1):
        """
        Decoder for the AR-LVM model, takes in the node embeddings and the latent variable z and returns a distribution over the output space,
        in this case the next timestep, with fixed variance.

        Args:
            num_layers: number of message passing layers
            emb_dim: dimension of the node embeddings
            out_dim: dimension of the output space
            nonlinearity: activation function
            sigma: fixed variance
        """
        super(Decoder, self).__init__()
        self.mp_layers = torch.nn.ModuleList([torch.nn.Conv2d(emb_dim+latent_dim, emb_dim+latent_dim, kernel_size=3, padding='same') for _ in range(num_layers)])
        self.fc = torch.nn.Conv2d(emb_dim+latent_dim, out_dim, kernel_size=1)
        self.nonlinearity = nonlinearity
        self.sigma = sigma

    def forward(self, h, z): #update this
        """
        Forward pass of the Decoder. Concatenates the node embeddings and the latent variable z, then applies message passing layers
        and returns a Normal distribution over the output space

        Args:
            h: node embeddings
            z: latent variable
            edge_index: adjacency list
            edge_attr: edge attributes
        """
        # Concatenate h and z
        h_cat = torch.cat([h, z], dim=1)
        # Apply message passing layers
        for mp in self.mp_layers:
            h_cat = mp(h_cat)
            h_cat = self.nonlinearity(h_cat)
        # Apply the final linear layer
        out = self.fc(h_cat)
        # Create a Normal distribution with the output and a fixed variance
        dec_dist = torch.distributions.Normal(out, self.sigma)
        return dec_dist
        

class Prior(torch.nn.Module):
    def __init__(self, num_layers=2, emb_dim=64, latent_dim=16, nonlinearity = torch.nn.functional.relu):
        """
        Prior class for the AR-LVM model, takes in the node embeddings and returns a distribution over the latent space

        Args:
            num_layers: number of message passing layers
            emb_dim: dimension of the node embeddings
            latent_dim: dimension of the latent space
            nonlinearity: activation function
        """
        super(Prior, self).__init__()
        self.mp_layers = torch.nn.ModuleList([torch.nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding='same') for _ in range(num_layers)])
        self.fc_mu = torch.nn.Conv2d(emb_dim, latent_dim, kernel_size=1)
        self.fc_sigma = torch.nn.Conv2d(emb_dim, latent_dim, kernel_size=1)
        self.nonlinearity = nonlinearity
    
    def forward(self, h):
        """
        Args:
            h: node embeddings
            edge_index: adjacency list
            edge_attr: edge attributes

        Returns:
            torch.Distribution: Normal distribution over the latent space
        """
        # Apply message passing layers
        for mp in self.mp_layers:
            h = mp(h)
            h = self.nonlinearity(h)
        
        # Compute the mean and sigma
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)
        sigma = torch.nn.functional.softplus(sigma) + 1e-6
        # Create a Normal distribution with mu and sigma
        prior_dist = torch.distributions.Normal(mu, sigma)
        return prior_dist


class AR_LVM_Model(torch.nn.Module):
    def __init__(self, encoder=None, processor=None, decoder=None, prior=None,use_coordinates=False):
        
        super(AR_LVM_Model, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.prior = prior
        self.use_coordinates=use_coordinates

    def forward(self, x_t, x_next, mask):
        """
        Forward pass of the AR-LVM model for grid data.
        """
        if self.use_coordinates and x_t.shape[1]>=5:
            coords=x_t[:, -2:, :, :]
            input_ar = torch.cat([x_t, mask], dim=1)
            in_next_ar = torch.cat([x_next, mask, coords], dim=1)
        else:
            input_ar = torch.cat([x_t, mask], dim=1)
            in_next_ar = torch.cat([x_next, mask], dim=1)

        h_t = self.processor(input_ar)
        h_next = self.processor(in_next_ar)
     
        enc_dist = self.encoder(h_t, h_next)

        z_inf = enc_dist.rsample() 
        
        prior_dist = self.prior(h_t)

        out_distance = self.decoder(h_t, z_inf)
        predicted_frame = out_distance.loc  # Mean prediction

        return (enc_dist, prior_dist, predicted_frame)

    def sample(self, mask, x_t):
        """
        Sample from the AR-LVM model with grid data.
        """
        # The dataloader already created the batch, so we can directly concatenate.
        processor_temp = torch.cat([x_t, mask], dim=1)

        # Get the node embeddings
        h = self.processor(processor_temp)
        # Get the Prior distribution p(z|h)
        prior_dist = self.prior(h)
        z = prior_dist.rsample()
        # Get the Decoder distribution p(x^t+1|z, h)
        out = self.decoder(h, z)
        return out.loc


class Trainer:
    def __init__(self, model, train_loader, validation_loader, batch_size=1, lr=0.0001, epochs=100, loss_fn=torch.nn.MSELoss(), model_name= "02-LV-FBF.pt"):
        """
        Simple Trainer class to train a PyTorch (geometric) model on a dataset.

        Args:
            model: PyTorch model to train
            train_dataset: PyTorch dataset to train on
            validation_dataset: PyTorch dataset to validate on
            batch_size: Batch size for training
            lr: Learning rate
            epochs: Number of epochs to train for
            loss_fn: Loss function to use
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train_loop(self):
        """
        Train loop for the model
        """
        best_model_loss = np.inf
        for epoch in range(self.epochs):
            # Train the model
            self.model.train()
            mean_train_loss = 0
            for i, data in enumerate(self.train_loader):
               mask, x_t, x_next = data
               mask = mask.to(self.device)
               x_t = x_t.to(self.device)
               x_next = x_next.to(self.device)
               self.optimizer.zero_grad()
               y_pred = self.model(x_t, x_next, mask)
               loss = self.loss_fn(y_pred, x_next)
               loss.backward()
               self.optimizer.step()
               mean_train_loss += loss.item() # Also fix the training loss calculation
            mean_train_loss /= len(self.train_loader)

            # Validate the model
            self.model.eval()
            mean_val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(self.validation_loader):
                    mask, x_t, x_next = data
                    mask = mask.to(self.device)
                    x_t = x_t.to(self.device)
                    x_next = x_next.to(self.device)
                    out = self.model(x_t, x_next, mask)  # FIX: Correct the argument order
                    loss = self.loss_fn(out, x_next,mask)
                    mean_val_loss += loss.item()
                mean_val_loss /= len(self.validation_loader)

            if mean_val_loss < best_model_loss:
                best_model_loss = mean_val_loss
                torch.save(self.model.state_dict(), f"{self.model_name}")

            print(f"Epoch {epoch}, Mean Train Loss: {mean_train_loss}, Mean Validation Loss: {mean_val_loss}")


input_dim=6 if use_coordinates else 4
processor=Processor(num_layers=4, in_dim=input_dim, emb_dim=64, nonlinearity=torch.nn.functional.relu)
lvm_cfd=AR_LVM_Model(
    encoder=Encoder(num_layers=2, emb_dim=64, latent_dim=16, nonlinearity=torch.nn.functional.relu),
    processor=processor,
    decoder=Decoder(num_layers=2, latent_dim=16, emb_dim=64, out_dim=3, nonlinearity=torch.nn.functional.relu, sigma=0.1),
    prior=Prior(num_layers=2, emb_dim=64, latent_dim=16, nonlinearity=torch.nn.functional.relu),
    use_coordinates=use_coordinates
)
loss=ELBO_Loss(L=20.0)
p = Trainer(model=lvm_cfd, train_loader=train_loader, validation_loader=val_loader, batch_size=batch_size, lr=0.0001, epochs=200, loss_fn=loss, model_name="02-LV-FBF-1.pt")
p.train_loop()


