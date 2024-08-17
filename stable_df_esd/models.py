import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from configs import *
from diffusers import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from diffusers import UNet2DConditionModel
from tqdm import tqdm
from PIL import Image
from utils import *
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")




class SD(torch.nn.Module):
    """
    Implementation of Stable diffusion model.
    """
    def __init__(self):
        """
        Initialise the SD model with various pretraind models. 
        AutoencoderKL for dimentionality reduction of image, before passing it to 
        the UNet model for getting the score.
        CLIPTokenizer and CLIPTextModel are used for toknization and encoding of text prompt.
        """
        super().__init__()
        self.encoder = AutoencoderKL.from_pretrained(ENCODER_PATH,subfolder=ENCODER_FOLDER)
        self.ddpm = UNet2DConditionModel.from_pretrained(DECODER_PATH,subfolder=DECODER_FOLDER)
        self.tokenizer = CLIPTokenizer.from_pretrained(TEXT_TOKENIZER_PATH)
        self.text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER_PATH)
        self.scheduler = DDIMScheduler.from_pretrained(DDIM_SCHEDULER_PATH,subfolder=DDIM_SCHEDULER_FOLDER)
        self.eval()

    @torch.no_grad()
    def __call__(self,prompts,pixel_size=512,n_steps=100,batch_size=1,last_itr=None):
        """
        Given a prompt get output from reverse diffusion process.
        """
        if type(prompts) != list:
            prompts = [prompts]

        self.scheduler.set_timesteps(n_steps,DEVICE)

        noise = torch.randn(batch_size, self.ddpm.in_channels, pixel_size//8, pixel_size//8, device=DEVICE).repeat(len(prompts), 1, 1, 1)

        latent = self.scheduler.init_noise_sigma * noise

        text_encodings = self.encode_text(prompts=prompts,count=batch_size)
        #print(text_encodings)

        last_itr = last_itr if last_itr is not None else n_steps
        
        latent_steps = self.reverse_diffusion(latent,text_encodings,last_itr=last_itr)

        latent_steps = [self.decode(latent.to(DEVICE)) for latent in latent_steps]

        image_steps = [self.to_image(image) for image in latent_steps]

        image_steps = list(zip(*image_steps))
        return image_steps
        

    @torch.no_grad()
    def reverse_diffusion(self,latents,embeddings,last_itr=1000,first_itr=0,original=False):
        latents_steps = []
        """
        Implementation of reverse diffusion process.
        """
        for itr in tqdm(range(first_itr, last_itr)):

            noise_pred = self.predict_noise(itr, latents, embeddings)

            #calculate xt-1
            output = self.scheduler.step(noise_pred, self.scheduler.timesteps[itr], latents)

            latents = output.prev_sample

            if itr == last_itr - 1:

                output = output.pred_original_sample if original else latents
                latents_steps.append(output)

        return latents_steps


    def encode_text(self,prompts, count):
        """
        Encode the text using the text tokenizer and encoder from CLIP model.
        """
        tokens = self.text_tokenize(prompts)

        text_encodings = self.text_encode(tokens)
        tokens_uncondition = self.text_tokenize([" "] * len(prompts))
        text_encodings_uncondition = self.text_encode(tokens_uncondition)
        #print(text_encodings_uncondition.shape)
        embeddings = torch.cat([text_encodings_uncondition, text_encodings])
        embeddings = embeddings.repeat_interleave(count, 0)
        return embeddings
    

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.ddpm.device))[0]

    def decode(self, latents):

        return self.encoder.decode(1 / self.encoder.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.encoder.encode(tensors).latent_dist.mode() * 0.18215
    
    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images
    
    def predict_noise(self,iteration,latents,text_embeddings,guidance_scale=7.5):
        """
        The function that predicts noise given a latents, text embedding.
        """
        # Doing double forward pass
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(latents, self.scheduler.timesteps[iteration])

        # Noise prediction
        noise_prediction = self.ddpm(latents, self.scheduler.timesteps[iteration], encoder_hidden_states=text_embeddings).sample

        # Classifier free guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction
    


if __name__ == "__main__":
    #model = SD().to(DEVICE).eval()
    model = SD().to(DEVICE).eval()


    generated_images = model(prompts="House",n_steps=20,batch_size=1)

    image_grid(generated_images,outpath='./images/out')


