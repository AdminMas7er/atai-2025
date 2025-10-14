# %%
import torch 
from torch_geometric.data import  DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import animation

# for cleaner look (set your theme to dark mode)
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1E1E1E'
plt.rcParams['axes.facecolor'] = '#1E1E1E'
plt.rcParams['savefig.facecolor'] = '#1E1E1E'

from tqdm import tqdm
from IPython.display import Image


# %%
################# quickfix for Snellius GPU MIG usage
import os 
import subprocess
if "MIG" in subprocess.check_output(["nvidia-smi", "-L"], text=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# this box fuses the loose data files such that they can be read by the dataset object
base_in = 'atai-2025/data/CFD/grid/loose/'
base_out2 = 'atai-2025/data/CFD/grid/concat/'
os.makedirs(base_out2, exist_ok=True)
for Re in [100,150,200,250,300,350,400]:
    u = np.load(f"{base_in}u_grid_Re{Re}.npy")
    v = np.load(f"{base_in}v_grid_Re{Re}.npy")
    p = np.load(f"{base_in}p_grid_Re{Re}.npy")
    concat = np.stack([u, v, p], axis=1)
    filename_save = f"{base_out2}uvp_grid_Re{Re}.npy"
    np.save(filename_save, concat)

# %%
class FlowDataset(Dataset):
    def __init__(self, filenames, flip_augmentation=False, timesample=1,bundle_size=6):
        self.sequences = []
        self.index_map = []
        self.flip_augmentation = flip_augmentation
        self.bundle_size = bundle_size

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

        # sample/read the data
        for seq_idx, filename in enumerate(filenames):
            data = np.load(filename)  
            data = data[::timesample] 
            self.sequences.append(data)
            T = data.shape[0]
            self.index_map.extend([(seq_idx, t) for t in range(T - self.bundle_size)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, t = self.index_map[idx]
        seq = self.sequences[seq_idx]
        input = seq[t]    
        target_bundle = seq[t+1:t+1+self.bundle_size]
        # if flip augmentation is true then flip the data horizontally 50% of the time
        if self.flip_augmentation and np.random.rand() > 0.5:
            input = self.flip(input)
            target_bundle = np.stack([self.flip(f) for f in target_bundle], axis=0)
        return (
                self.mask, 
                torch.tensor(input, dtype=torch.float32), 
                torch.tensor(target_bundle, dtype=torch.float32)
                )
        
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

datafolder = 'atai-2025/data/CFD/grid/concat/'
train_files = [
    'uvp_grid_Re100.npy',
    'uvp_grid_Re200.npy',
    'uvp_grid_Re300.npy',
    'uvp_grid_Re400.npy'
]
val_files = [
    'uvp_grid_Re150.npy',
    'uvp_grid_Re250.npy',
    'uvp_grid_Re350.npy'
]
train_files = [datafolder + f for f in train_files]
val_files = [datafolder + f for f in val_files]

dt = 20 # only sample every dt timesteps
batch_size = 64
bundle_size=6
train_dataset = FlowDataset(train_files, flip_augmentation=False, timesample=dt, bundle_size=bundle_size)
val_dataset = FlowDataset(val_files, flip_augmentation=False, timesample=dt, bundle_size=bundle_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%
class ELBO_Loss(torch.nn.Module):
    def __init__(self, L):
        """
        Helper class to compute the ELBO loss for the AR-LVM model

        Args:
            L: size of the periodic boundary conditions
        """
        super(ELBO_Loss, self).__init__()
        self.L = L
    
    def forward(self, y_pred, y_true_bundle):
        """
        Forward pass of the ELBO loss.

        Args:
            y_pred: (q(z|h, x^t+1), p(z|h), p(x^t+1|z, h))
                each is a torch.Distribution object
            y_true: (x^t+1)

        ELBO is the sum of the KL divergence between the approximate posterior and the prior
        and the reconstruction loss
        = KL(q(z|h, x^t+1) || p(z|h)) - E_q(z|h, x^t+1)[log p(x^t+1|z, h)]

        Returns:
            torch.Tensor: ELBO loss

        """
        enc_dist, prior_dist, predicted_bundle = y_pred
        
        # KL divergence is calculated once per sequence
        kl = torch.distributions.kl.kl_divergence(enc_dist, prior_dist).mean(0).sum()
        
        # Reconstruction loss is the MSE over the entire predicted sequence
        recon = torch.nn.functional.mse_loss(predicted_bundle, y_true_bundle, reduction='sum') / y_true_bundle.shape[0] # Mean over batch
        
        return kl + recon
        

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
    def __init__(self, encoder=None, processor=None, decoder=None, prior=None):
        super(AR_LVM_Model, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.prior = prior

    def forward(self, x_t, target_bundle, mask):
        """
        Forward pass of the AR-LVM model for grid data.
        """
        bundle_size = target_bundle.shape[1]
        input_ar = torch.cat([x_t, mask], dim=1)
        
        # Filtering posterior: only first future frame (no leakage of full horizon)
        first_future = target_bundle[:, 0]                  # (B,C,H,W)
        h_t = self.processor(input_ar)
        h_next = self.processor(torch.cat([first_future, mask], dim=1))
        enc_dist = self.encoder(h_t, h_next)
        # Use the z from the encoder that sees the whole sequence
        z_inf = enc_dist.rsample() 
        
        prior_dist = self.prior(h_t)
        
        generated_frames=[]
        current_x=x_t
        for _ in range(bundle_size):
            h_step = self.processor(torch.cat([current_x, mask], dim=1))
            out_dist = self.decoder(h_step, z_inf)
            current_x = out_dist.loc
            generated_frames.append(current_x)

        # Stack the generated frames into a bundle
        predicted_bundle = torch.stack(generated_frames, dim=1)
        
        return (enc_dist, prior_dist, predicted_bundle)

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

# %%
class Trainer:
    def __init__(self, model, train_loader, validation_loader, batch_size=1, lr=0.0001, epochs=100, loss_fn=torch.nn.MSELoss(), model_name= "02-LV-TB.pt"):
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
               mask,x_t,x_next_bundle=data
               mask=mask.to(self.device)
               x_t=x_t.to(self.device)
               x_next_bundle=x_next_bundle.to(self.device)
               self.optimizer.zero_grad()
               # FIX: Correct the argument order to match the model's forward method
               out = self.model(x_t, x_next_bundle, mask)
               loss = self.loss_fn(out, x_next_bundle)
               loss.backward()
               self.optimizer.step()
               mean_train_loss += loss.item() # Also fix the training loss calculation
            mean_train_loss /= len(self.train_loader)

            # Validate the model
            self.model.eval()
            mean_val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(self.validation_loader):
                    mask, x_t, x_next_bundle = data
                    mask = mask.to(self.device)
                    x_t = x_t.to(self.device)
                    x_next_bundle = x_next_bundle.to(self.device)
                    out = self.model(x_t, x_next_bundle, mask)  # FIX: Correct the argument order
                    loss = self.loss_fn(out, x_next_bundle)
                    mean_val_loss += loss.item()
                mean_val_loss /= len(self.validation_loader)

            if mean_val_loss < best_model_loss:
                best_model_loss = mean_val_loss
                torch.save(self.model.state_dict(), f"atai/2025/models/CFD/{self.model_name}")

            print(f"Epoch {epoch}, Mean Train Loss: {mean_train_loss}, Mean Validation Loss: {mean_val_loss}")

# %%
processor=Processor(num_layers=4, in_dim=4, emb_dim=64, nonlinearity=torch.nn.functional.relu)
lvm_cfd=AR_LVM_Model(
    encoder=Encoder(num_layers=2, emb_dim=64, latent_dim=16, nonlinearity=torch.nn.functional.relu),
    processor=processor,
    decoder=Decoder(num_layers=2, latent_dim=16, emb_dim=64, out_dim=3, nonlinearity=torch.nn.functional.relu, sigma=0.1),
    prior=Prior(num_layers=2, emb_dim=64, latent_dim=16, nonlinearity=torch.nn.functional.relu)
)
loss=ELBO_Loss(L=20.0)
p = Trainer(model=lvm_cfd, train_loader=train_loader, validation_loader=val_loader, batch_size=batch_size, lr=0.0001, epochs=100, loss_fn=loss, model_name="02-LV-FBF.pt")
p.train_loop()


