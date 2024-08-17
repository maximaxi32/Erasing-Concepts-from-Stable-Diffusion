from model import LitDiffusionModel
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset import ThreeDSinDataset
import pytorch_lightning as pl

litmodel = LitDiffusionModel()

litmodel = LitDiffusionModel.load_from_checkpoint(
    "./runs/n_dim=3,n_steps=50,lbeta=1.000e-05,ubeta=1.280e-02,scheduler_type=linear,batch_size=1024,n_epochs=500/last.ckpt",
    hparams_file="./runs/n_dim=3,n_steps=50,lbeta=1.000e-05,ubeta=1.280e-02,scheduler_type=linear,batch_size=1024,n_epochs=500/lightning_logs/version_0/hparams.yaml"
)

print(litmodel.betas)
print(litmodel.alphas_cum)
q_samp = []
x_sam = torch.randn((7781, 3))
q_samp.append(x_sam)
for t in range(49, -1, -1):
    print(t)
    eps_factor, eps_theta, mean, sigma_t, x_sam = litmodel.p_sample(x_sam, t)
    #eps_factor, eps_theta, mean, sigma_t, x_sam = litmodel.p_sample(x_sam, t)
    # print("eps_factor", eps_factor.shape)
    # print("eps_theta", eps_theta.shape)
    # print("mean", mean.shape)
    # print("sigma ",sigma_t.shape)
    print(eps_factor[0])
    print(eps_theta[0])
    print(mean[0])
    print(sigma_t[0])
    q_samp.append(x_sam)

