# CS726-PROJECT- Erasing Stable Diffusion

### Meet Doshi, 22m0742

### Osim Abes, 22m0825

### Saswat Meher, 22m0804

## File Structure
This folder contains 3 sub folder. "abstract_ddpm" codes related to implementation of erasing concept in ddpm. Folder "conditional_ddpm" contains codes related to implementaion of erasing concept in a conditional ddpm models. The folder "stable_df_esd" contain codes for the applying esd over a Stable Diffusion model to erase concept from images. Following sections describe in details on how to run various models.

## Abstract DDPM
From inside the folder of abstract_ddpm run the following command to train the DDPM and Erased DDPM.
Use the command to generate dataset`python3 generate.py`
 To train the ddpm use `python3 train.py`
 To finetune the ddpm use `python3 finetune.py`
 Run `eval.py` to generate a sample from the finetuned model.
 
 ## Conditional DDPM
From inside the folder of conditional_ddpm run the following command to train the conditional DDPM and then Erased Conditional DDPM.
Use the command to generate dataset`python3 gen_data.py`
 To train the ddpm use `python3 train_ddpm.py`
 To finetune the ddpm use `python3 train_eddpm.py`
 Run `sample.py` to generate a sample from the finetuned model. Use argument `-e` to generate samples from finetune model.
 
 ## Erasing Stable Diffusion
From inside the folder of stable_ef_esd run the following command to train the Erased SD. Modify Config file to change paths for pretrined model.
 To finetune the EDS use `python3 train.py`
 To make inference from the ddpm use `python3 inference.py`
 Use `python3 app.py` to run a demo locally and then make inference on it.
