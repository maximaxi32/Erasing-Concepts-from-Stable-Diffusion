import argparse
import torch
import pytorch_lightning as pl
from model import LitDiffusionModel
from dataset import ThreeDSinDataset

parser = argparse.ArgumentParser()

model_args = parser.add_argument_group('model')
model_args.add_argument('--n_dim', type=int, default=3, help='Number of dimensions')
model_args.add_argument('--n_steps', type=int, default=100, help='Number of diffusion steps')
model_args.add_argument('--lbeta', type=float, default=1e-5, help='Lower bound of beta')
model_args.add_argument('--ubeta', type=float, default=1.28e-2, help='Upper bound of beta')
model_args.add_argument('--scheduler_type', type=str, default="linear", help='Variance Scheduling Function')

training_args = parser.add_argument_group('training')
training_args.add_argument('--seed', type=int, default=1618, help='Random seed for experiments')
training_args.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs')
training_args.add_argument('--batch_size', type=int, default=1024, help='Batch size for training dataloader')
training_args.add_argument('--train_data_path', type=str, default='./data/gaussian_3D_train.npy', help='Path to training data numpy file')
training_args.add_argument('--savedir', type=str, default='./runs/', help='Root directory where all checkpoint and logs will be saved')

args = parser.parse_args()

n_dim = args.n_dim
n_steps = args.n_steps
lbeta = args.lbeta
ubeta = args.ubeta
scheduler_type=args.scheduler_type

pl.seed_everything(args.seed)
batch_size = args.batch_size
n_epochs = args.n_epochs
savedir = args.savedir

litmodel = LitDiffusionModel(
    n_dim=n_dim, 
    n_steps=n_steps, 
    lbeta=lbeta, 
    ubeta=ubeta,
    scheduler_type=scheduler_type
    )

train_dataset = ThreeDSinDataset(args.train_data_path)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

run_name = f'n_dim={n_dim},n_steps={n_steps},lbeta={lbeta:.3e},ubeta={ubeta:.3e},scheduler_type={scheduler_type},batch_size={batch_size},n_epochs={n_epochs}'

trainer = pl.Trainer(
    deterministic=True,
    logger=pl.loggers.TensorBoardLogger(f'{savedir}/{run_name}/'),
    max_epochs=n_epochs,
    log_every_n_steps=1,
    callbacks=[
        # A dummy model checkpoint callback that stores the latest model at the end of every epoch
        pl.callbacks.ModelCheckpoint(
            dirpath=f'{savedir}/{run_name}/',
            filename='{epoch:04d}-{train_loss:.3f}',
            save_top_k=1,
            monitor='epoch',
            mode='max',
            save_last=True,
            every_n_epochs=1,
        ),
    ]
)

trainer.fit(model=litmodel, train_dataloaders=train_dataloader)
