import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import nn
import copy
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class VNet(nn.Module):
    """
    A UNet implementation adapted to surface generation. 5x5 kernels promote smoothness and only two downsampling
    blocks prevent excessively complex image structures. Multiplicative time-embedding exists at all scales.
    
    input shape: (d,1,16,32)

    output shape: (d,1,16,32)
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.emb_dim_1=128
        self.emb_dim_2=256

        self.act = nn.SiLU()
        self.pre_conv_1=nn.Conv2d(1,64,3,padding=1,padding_mode='reflect')
        self.pre_conv_2=nn.Conv2d(64,64,3,padding=1,padding_mode='reflect')
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.post_conv_1=nn.Conv2d(64,64,3,padding=1,padding_mode='reflect')
        self.post_conv_2=nn.Conv2d(64,1,3,padding=1,padding_mode='reflect')
        
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(64, self.emb_dim_1, kernel_size=5, padding=2,padding_mode='reflect'),
            nn.Conv2d(self.emb_dim_1, self.emb_dim_2, kernel_size=5, padding=2,padding_mode='reflect'),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(self.emb_dim_2, self.emb_dim_1, kernel_size=5, padding=2,padding_mode='reflect'),
            nn.Conv2d(self.emb_dim_1, 64, kernel_size=5, padding=2,padding_mode='reflect'),
        ])

        self.t_emb_1=nn.Sequential(nn.Linear(1,64),nn.SiLU(),nn.Linear(64,self.emb_dim_2))
        self.t_emb_2=nn.Sequential(nn.Linear(1,64),nn.SiLU(),nn.Linear(64, self.emb_dim_1))

        self.t_res_emb_0=nn.Sequential(nn.Linear(1,64),nn.Sigmoid(),nn.Linear(64,64))
        self.t_res_emb_1=nn.Sequential(nn.Linear(1,self.emb_dim_1),nn.Sigmoid(),nn.Linear(self.emb_dim_1, self.emb_dim_1))
        self.t_res_emb_2=nn.Sequential(nn.Linear(1,self.emb_dim_2),nn.Sigmoid(),nn.Linear(self.emb_dim_2, self.emb_dim_2))

    def forward(self, x, t, y, plot=False):
        noisy=x
        batch_size = x.shape[0]

        latents = []

        x=self.pre_conv_1(x)
        x=self.pre_conv_2(x)
        res=x

        t_res_emb_0=self.t_res_emb_0(t).view(batch_size,64,1,1)
        t_res_emb_1=self.t_res_emb_1(t).view(batch_size,self.emb_dim_1,1,1)
        t_res_emb_2=self.t_res_emb_2(t).view(batch_size,self.emb_dim_2,1,1)

        t_emb_1=self.t_emb_1(t).view(batch_size,self.emb_dim_2,1,1)
        t_emb_2=self.t_emb_2(t).view(batch_size,self.emb_dim_1,1,1)

        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            latents.append(x)
            x = self.down(x)
              
        for i, layer in enumerate(self.up_layers):
            x = self.up(x)
            if i==0:
                x=x*t_emb_1+latents.pop()*t_res_emb_2
            if i==1:
                x=x*t_emb_2+latents.pop()*t_res_emb_1
            x = self.act(layer(x))
        
        x+=res*t_res_emb_0
        x=self.post_conv_1(x)
        x=self.act(x)
        x=self.post_conv_2(x)
        
        return x



class VolaDiff():
    """
    Diffusion model for volatility surfaces.

    Technical specs: zero terminal SNR variance-preserving noise schedule, UNet x-prediction denoising, fully stochastic DDIM sampler.

    Train the model on a dataset of n 16x32 volatility surfaces, normalised to approximately [-1,1], reshaped to (n,1,16,32) and saved
    as a pt file. A good final NMSE is 0.04. Load the model from "best_model_checkpoint.pt" to sample from the model. Condition the
    model by providing a data loader of target surfaces to the sampling function. Unconditional generation is also possible by supplying
    'None' to the data loader.
    
    """
    
    def __init__(self):
        self.T_max=1
        self.n_steps=100
        self.dt=self.T_max/self.n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=VNet(in_channels=1).to(self.device)
        self.exp_model=copy.deepcopy(self.model)
        self.best_model=copy.deepcopy(self.model)

    def load(self,path):
        self.best_model.load_state_dict(torch.load(path,weights_only=True))

    def var_inst(self,t):
        return -2*torch.log(1-torch.pow(t,2))

    def alpha(self,t):
        return torch.exp(-1/2*self.var_inst(t))

    def var(self,t):
        return 1-torch.exp(-1/2*self.var_inst(t))

    def corrupt(self,x,t):
        noise=torch.randn_like(x)
        return x*torch.sqrt(self.alpha(t))+torch.sqrt(self.var(t))*noise, noise

    def mse(self,x,y):
        return torch.mean(torch.square(x-y))

    def nmse(self,x,y):
        return torch.mean(torch.square(x-y))/torch.mean(torch.square(y))

    def train(self,data_loader,params,n_epochs=1000,save_model=True,plotting=True,printing=True):
        batch_size=params["batch_size"]
        batches_per_epoch=len(data_loader)//batch_size

        opt=torch.optim.Adam(self.model.parameters(),lr=1e-4)
        losses=[]
        ema_losses=[]
        ema_nmse=1
        best_ema_nmse=1

        for epoch in range(n_epochs):
            for i,batch in enumerate(tqdm(data_loader)) if printing else enumerate(data_loader):
                #ensure that the normalised surfaces range from approximately -1 to 1
                vol_target=batch.to(self.device).view(-1,1,16,32)

                #uniform time distribution
                t=torch.rand(vol_target.shape[0]).to(self.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                noisy_x, noise = self.corrupt(vol_target, t)

                pred = self.model(noisy_x,t,vol_target)
                loss=self.mse(pred,vol_target)
                nmse=self.nmse(pred,vol_target)

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(nmse.item())
                ema_losses.append(0.99*ema_losses[-1]+0.01*losses[-1] if len(ema_losses)>0 else losses[-1])

                gamma=6.94 #decay rate averaging the final 10% of the training
                effective_iter=epoch*batches_per_epoch+i
                decay_rate=np.power(1-1/(effective_iter+1),gamma+1)
                
                #update the exponential moving average model
                for param,exp_param in zip(self.model.parameters(),self.exp_model.parameters()):
                        exp_param.data=decay_rate*exp_param.data+(1-decay_rate)*param.data

                #update the best model
                if 0.99*ema_nmse+0.01*losses[-1]<best_ema_nmse:
                    self.best_model=copy.deepcopy(self.exp_model)
                    best_ema_nmse=0.99*ema_nmse+0.01*losses[-1]
                ema_nmse=0.99*ema_nmse+0.01*losses[-1]
            print(f"Epoch {epoch}, NMSE: {np.round(ema_nmse,4)}, best NMSE: {np.round(best_ema_nmse,4)}") if printing else None

        if plotting:
            plt.plot(np.log(losses),label="Loss")
            plt.plot(np.log(ema_losses),label="Expontentially Weighted Average Loss")
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Log NMSE Loss")
            plt.show()
        
        torch.save(self.best_model.state_dict(), "best_model_checkpoint.pt") if save_model else None

        print(f"Training complete. Best NMSE: {best_ema_nmse}")

    @torch.no_grad()
    def denoise_x(self,z,t,x_pred):
        if t<0.03:
            return x_pred
        return self.var(t-self.dt)/self.var(t)*z+(1-self.var(t-self.dt)/self.var(t))*x_pred+torch.sqrt(self.var(t-self.dt)/self.var(t))*torch.sqrt(self.var(t)-self.var(t-self.dt))*torch.randn_like(z)
    
    @torch.no_grad()
    def sample_uncond(self,batch_size,return_history=False,printing=False):
        z=torch.randn(batch_size,1,16,32)
        sample_history=[]

        for t in tqdm(torch.arange(self.n_steps,1,-1)*1.) if printing else torch.arange(self.n_steps,1,-1)*1.:
            t=t.clone().detach().to(self.device)/self.n_steps*self.T_max
            t_pad=t.repeat(batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            pred=self.best_model(z.to(self.device),t_pad,z.to(self.device))

            if return_history:
                sample_history.append(z)
            z=self.denoise_x(z.to(self.device),t,pred)
        if return_history:
            sample_history.append(z)
        return sample_history if return_history else z

    @torch.no_grad()
    def sample_cond(self,batch_size,target_data_loader,conditioning_time=0.5,return_history=False,printing=False):
        z=torch.randn(batch_size,1,16,32)
        sample_history=[]

        subset=torch.randint(0,target_data_loader.dataset.data.shape[0],(batch_size,))
        pred_con=target_data_loader.dataset.data[subset].to(self.device).view(-1,1,16,32)

        for t in tqdm(torch.arange(self.n_steps,1,-1)*1.) if printing else torch.arange(self.n_steps,1,-1)*1.:
            t=t.clone().detach().to(self.device)/self.n_steps*self.T_max
            t_pad=t.repeat(batch_size).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            pred_uncon=self.best_model(z.to(self.device),t_pad,z.to(self.device))

            #if t>conditioning_time, use the condition targets to denoise, otherwise use the unconditional targets
            if t>conditioning_time:
                pred=pred_con
            else:
                pred=pred_uncon

            if return_history:
                sample_history.append(z)
            z=self.denoise_x(z.to(self.device),t,pred)
        if return_history:
            sample_history.append(z)
        return sample_history if return_history else z

class VolDataset(Dataset):
    def __init__(self,file_path):
        with open(file_path,"rb") as file:
            self.data=torch.load(file,weights_only=True).type(torch.float32)
            if self.data.shape[1:4]!=(1,16,32):
                raise ValueError(f"Data shape {self.data.shape} is not valid. It should be (n,1,16,32)")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        return self.data[idx].float().unsqueeze(0)