import gradio as gr
import torch
from models import SD
from configs import *
from PIL import Image
from inference import inference
import numpy as np

"""
Python file to run a demo on the finetuned model.
"""
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

SD_PATH = 'CompVis/stable-diffusion-v1-4'
ESD_PATH = './saved_models/esd.pt' #ocean

def generate_image(prompt):
    orig_images, ft_images = inference(prompt,sd_path=SD_PATH,esd_path=ESD_PATH)
    orig_images = orig_images.resize((200,200))
    ft_images = ft_images.resize((200,200))
    return orig_images, ft_images

inputs = gr.inputs.Textbox(lines=2, placeholder="Type here to generate an image...")
outputs = [
    gr.outputs.Image(type="pil", label="Original Image").style(height="200px", width="200px"),
    gr.outputs.Image(type="pil", label="Edited Image").style(height="200px", width="200px")
]

title = "Erasing Concepts from DDPMs"
description = "CS726 Project by Meet, Saswat, Osim."

gr.Interface(generate_image, inputs, outputs, title=title, description=description, examples=[["beachside sunset."]]).launch()

