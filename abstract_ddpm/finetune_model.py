import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from model import LitDiffusionModel

def expand_alphas(batch, alpha, t):
    """
    If t is not a tensor object than expand alpha[t] to shape of batch
    else get alpha[t] in the shape of x
    """
    if isinstance(t, int):
        return alpha[t].expand(batch.size(0),1)
    else:
        return torch.zeros((batch.size(0),1)) + alpha[t][:,None]


class FineTuneLitDiffusionModel(pl.LightningModule):
    """
    Class for finuetuning the diffusion model. 
    """
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2, scheduler_type="linear",model_chkpt=None,model_hparams=None):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """

        embedding_dim = 8

        # nn.Embedding is used to embed time in a latend dimention that can be passed to the model 
        #self.time_embed = nn.Embedding(n_steps,embedding_dim)

        # The model is a sequential model consisting of linear layers with ReLU as activation function
        self.model = LitDiffusionModel.load_from_checkpoint(model_chkpt,hparams=model_hparams)
        self.frozen_model = LitDiffusionModel.load_from_checkpoint(model_chkpt,hparams=model_hparams)
        self.frozen_model.eval()

        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim
        
        # New hyperparameter
        
        # Scheduler type (liner/sigmoid/cosine) is stored
        self.scheduler_type=scheduler_type           

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
           t = torch.LongTensor([t]).expand(x.size(0))
        #t_embed = self.time_embed(t)
        #return self.model(torch.cat((x, t_embed), dim=1).float())
        return self.model.forward(x,t)

    
    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """

        if self.scheduler_type=="linear":
            self.beta = torch.linspace(lbeta,ubeta,self.n_steps)
        elif self.scheduler_type == "sigmoid":
            self.beta = torch.linspace(-6, 6, self.n_steps)
            self.beta = torch.sigmoid(self.beta) * (ubeta - lbeta) + lbeta
        elif self.scheduler_type == "cosine":
            self.beta = torch.linspace(0, 1, self.n_steps)
            self.beta = (torch.cos((self.beta)*np.pi/2)+0.002)/1.002 * (ubeta - lbeta) + lbeta

        #Store different type of alpha and beta for speedup calculation
        self.alpha = 1-self.beta
        self.alpha_cum = torch.sqrt(torch.cumprod(1 - self.beta, 0))
        self.alpha_cum_sqrt = torch.sqrt(self.alpha_cum)
        self.one_min_alphas_sum_sqrt = torch.sqrt(1-self.alpha_cum)

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        alpha = expand_alphas(x,self.alpha_cum_sqrt, t)
        one_minus_alpha = expand_alphas(x, 1-self.alpha_cum, t)
        _q_sample = (alpha * x) + one_minus_alpha * torch.randn_like(x) 
        return _q_sample

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        epsilon_factor = (expand_alphas(x, self.beta, t) / expand_alphas(x, self.one_min_alphas_sum_sqrt, t))
        epsilon_theta = self.forward(x, t)
        mean = (1 / expand_alphas(x, torch.sqrt(self.alpha), t)) * (x - (epsilon_factor * epsilon_theta))
        sigma = expand_alphas(x, torch.sqrt(self.beta), t)
        sample = mean + sigma * torch.randn_like(x)
        return sample

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        batch_x = batch[:,:3]
        #print(batch_x.shape)
        batch_conditioning = batch[:,3:]

        batch_nc = batch[:,:3].clone()
        batch_nc[:,2] -= batch_conditioning[:,2]


        #print(batch_conditioning.shape)
        t = torch.randint(0, self.n_steps, size=(1,))
        t = t.expand(batch_x.size(0))
        #print(t.shape)
        alpha_sqrt = expand_alphas(batch_x, self.alpha_cum_sqrt, t)
        #print(alpha_sqrt.shape)
        one_min_alpha_sqrt = expand_alphas(batch_x, self.one_min_alphas_sum_sqrt, t)
        #print(one_min_alpha_sqrt.shape)
        noise = torch.randn_like(batch_x)
        #print(noise.shape)
        x = batch_x * alpha_sqrt + noise * one_min_alpha_sqrt
        c = batch_conditioning * alpha_sqrt + noise * one_min_alpha_sqrt
        nc = batch_nc * alpha_sqrt + noise * one_min_alpha_sqrt
        eta = 1.5
        conditioning_input = (x+c)/2.0
        output = self.forward(conditioning_input, t)
        with torch.no_grad():
            frozen_conditioning_score = self.frozen_model(conditioning_input,t)
            frozen_unconditioning_score = self.frozen_model(x,t)
            frozen_nc_score = self.frozen_model(nc,t)
        
        temp = frozen_unconditioning_score - frozen_nc_score
        guidance = (frozen_unconditioning_score + (eta)*(temp))/(eta)
        a = output
        b = guidance - output

        criteria = torch.nn.MSELoss()
        return criteria(output, guidance)

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        if return_intermediate:
            out_samples = []
            out_samples.append(torch.randn((n_samples, self.n_dim)))
            for t in range(self.n_steps-1, -1, -1):
                out_samples.append(self.p_sample(out_samples[-1], t))
            return out_samples[-1], out_samples
        else:
            out_sample = torch.randn((n_samples, self.n_dim))
            for t in range(self.n_steps-1, -1, -1):
                out_sample = self.p_sample(out_sample, t)
            return out_sample

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        # We have experimented with both SGD and Adam optimiser
        #return torch.optim.SGD(self.model.parameters(), lr=0.01)
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)