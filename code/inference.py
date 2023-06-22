
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import json
from PIL import Image
import numpy as np

device="cuda"
# model_name = "nitrosocke/Ghibli-Diffusion"
# model_name = "stabilityai/stable-diffusion-2-1-base"
model_name = "hakurei/waifu-diffusion"

def model_fn(model_dir):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, 
        torch_dtype=torch.float32).to(
            device
    )
    return pipe

# def input_fn(data, content_type):
#     return "data"

def predict_fn(data, model):
    if type(data) == str:
        data = json.loads(data)

    prompt = data["prompt"]
    neg_prompt = data["negative_prompt"]
    image = data["image"]
    strength = data["strength"]
    guidance_scale = data["guidance_scale"]
    num_inference_steps = data["num_inference_steps"]
    seed = data["seed"]
    
    image = Image.fromarray(np.array(image).astype('uint8'), 'RGB') # converts list to np array and then to PIL image
    
    generator = torch.Generator(device=device).manual_seed(seed)
    output_image = model(prompt=prompt, 
                         negative_prompt=neg_prompt,
                         image=image, 
                         strength=strength, 
                         guidance_scale=guidance_scale, 
                         num_inference_steps=num_inference_steps, 
                         generator=generator
                        ).images[0]
    return output_image

def output_fn(prediction, accept):
    return {
        "output_image": np.array(prediction).tolist()
    }
