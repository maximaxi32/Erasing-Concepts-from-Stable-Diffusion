from models import SD 
from configs import *
import torch
from train import train
from PIL import Image
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def inference(prompt,sd_path, esd_path):
    """
    Given a prompt and a path to ESD generate image after loading ESD.
    """
    ddpm = SD()
    ddpm = ddpm.to(DEVICE)
    ddpm.eval()

    gen_images_sd = ddpm(prompt,n_steps=50)

    orig_images = gen_images_sd[0][0]

    del ddpm

    torch.cuda.empty_cache()

    esd = SD()
    esd.load_state_dict(torch.load(esd_path))
    esd = esd.to(DEVICE)
    esd.eval()

    gen_images_esd = esd(prompt,n_steps=50)

    ft_images = gen_images_esd[0][0]

    del esd

    torch.cuda.empty_cache()

    return orig_images, ft_images

if __name__ == '__main__':
    SD_PATH = 'CompVis/stable-diffusion-v1-4'
    ESD_PATH = './saved_models/esd.pt' #ocean
    orig_images, ft_images = inference('beachside sunset',SD_PATH,ESD_PATH)
