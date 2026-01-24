from diffusers import StableDiffusionPipeline
import torch
import numpy as np


from PIL import Image

model_id = "riffusion/riffusion-model-v1"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
pipe.load_lora_weights("lora_melodic_house_v1")

prompt = "melodic house with singing"

for i in range(100):
	image = pipe(prompt, 512, 512, num_inference_steps=200).images[0]  
	image.save(f"generated_lora/test{i}.png")