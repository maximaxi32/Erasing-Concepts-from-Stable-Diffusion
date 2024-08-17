from models import SD
from configs import *
import torch
from copy import deepcopy
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

ddpm = SD()
ddpm = ddpm.to(DEVICE)
ddpm.train()

def train(prompt,epochs=100,eta=1.0,path='./saved_models/esd.pt'):
    """
    Method that finetunes the ESD model.
    """
    frozen_ddpm = deepcopy(ddpm)
    frozen_ddpm.eval()

    optimizer = torch.optim.Adam(ddpm.parameters(),lr=1e-5)
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        unconditioned_embeddings = frozen_ddpm.encode_text([''],count=1)
        conditioned_embeddings = frozen_ddpm.encode_text([prompt],count=1)

    del frozen_ddpm.tokenizer
    del frozen_ddpm.text_encoder
    del frozen_ddpm.encoder

    torch.cuda.empty_cache()

    for epoch in range(epochs):
        with torch.no_grad():
            frozen_ddpm.scheduler.set_timesteps(50,DEVICE)
            optimizer.zero_grad()
            t = torch.randint(1,50-1,(1,)).item()

            noise = torch.randn(1, frozen_ddpm.ddpm.in_channels, 512//8, 512//8, device=DEVICE).repeat(1, 1, 1, 1)

            latent = frozen_ddpm.scheduler.init_noise_sigma * noise

            ddpm.scheduler.set_timesteps(50,DEVICE)
            latent_steps = ddpm.reverse_diffusion(latent,conditioned_embeddings,last_itr=t,first_itr=0,original=False)


            frozen_ddpm.scheduler.set_timesteps(1000,DEVICE)
            ddpm.scheduler.set_timesteps(1000,DEVICE)

            t = int(t/50*1000)

            latents_pos = frozen_ddpm.predict_noise(t,latent_steps[0],conditioned_embeddings)
            latents_neutral = frozen_ddpm.predict_noise(t,latent_steps[0],unconditioned_embeddings)


        
        latents_neg = ddpm.predict_noise(t,latent_steps[0],conditioned_embeddings)

        latents_pos.requires_grad = False
        latents_neutral.requires_grad = False

        loss = criterion(latents_neg,latents_neutral-(eta*(latents_pos-latents_neutral)))

        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch} Loss: {loss.item()}')

    torch.save(ddpm.state_dict(),path)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    train('ocean',epochs=100,eta=1e-3,path='./saved_models/esd.pt')


