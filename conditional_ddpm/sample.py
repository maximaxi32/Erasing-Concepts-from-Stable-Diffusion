import os
import argparse
import torch
import numpy as np
from ddpm import DDPM
from eddpm import EDDPM
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-e','--erased', action='store_true', help='model type ddpm/eddpm')
args = parser.parse_args()

if args.erased:
    model_type = "eddpm"
else:
    model_type = "ddpm"

if args.erased:
    litmodel = EDDPM.load_from_checkpoint(
        "run_eddpm/last.ckpt", 
        hparams_file= "./run_eddpm/lightning_logs/version_0/hparams.yaml"
    )
else:
    litmodel = DDPM.load_from_checkpoint(
        "run_ddpm/last.ckpt", 
        hparams_file= "run_ddpm/lightning_logs/version_0/hparams.yaml"
    )
    
    
litmodel.eval()

sample_sz = 10000
labels = [0, 1, None]

for label in labels:
    with torch.no_grad():
        print(f"Sampling for Label {label}")
        if label!= None:
            label = torch.tensor(label)
        gendata = litmodel.sample(sample_sz, label)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(gendata[:, 0], gendata[:, 1], gendata[:, 2], c=gendata[:, 2], alpha=0.1)
        #ax.scatter(testdata[:, 0], testdata[:, 1], testdata[:, 2], c=testdata[:, 2], alpha=0.1)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_zlabel("z axis")
        plt.savefig(f"results/{model_type}_{label}.pdf")
        plt.show()

"""

label = torch.tensor(1)
print(label)
with torch.no_grad():
    gendata = litmodel.sample(10000, label)


#temp_data = np.concatenate((testdata.numpy(), gendata.numpy()), axis=0)
#print(f'gendata.shape = {gendata.shape}')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gendata[:, 0], gendata[:, 1], gendata[:, 2], c=gendata[:, 2], cmap=cm.spring, alpha=0.1)
#ax.scatter(testdata[:, 0], testdata[:, 1], testdata[:, 2], c=testdata[:, 2], alpha=0.1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(0, 1)
plt.show()


label = None
print(label)
with torch.no_grad():
    gendata = litmodel.sample(10000, label)


#temp_data = np.concatenate((testdata.numpy(), gendata.numpy()), axis=0)
#print(f'gendata.shape = {gendata.shape}')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gendata[:, 0], gendata[:, 1], gendata[:, 2], c=gendata[:, 2], cmap=cm.spring, alpha=0.1)
#ax.scatter(testdata[:, 0], testdata[:, 1], testdata[:, 2], c=testdata[:, 2], alpha=0.1)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(0, 1)
plt.show()"""