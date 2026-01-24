from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler
import torch
import numpy as np
import random

from PIL import Image

model_id = "riffusion/riffusion-model-v1"

pipe = StableDiffusionPipeline.from_pretrained("edm_model_new_v10")
# scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)

pipe.requires_safety_checker = False
pipe.safety_checker = None

# pipe.scheduler = scheduler
pipe = pipe.to("cuda")

# prompt = "sultan and shepard"

ARTISTS = ["lane 8", "le youth", "sultan and shepard",
		   "kaskade", "fisher", "dom dolla", "jerro", "avicii",
		   "armin van buuren", "yotto", "gorgon city", "mau p",
		   "lane 8", "zedd", "lane 8", "jai wolf", "fred again",
		   "embrz", "nora en pure", "porter robinson", "deadmau5"]

TAGS = ["melodic house", "deep house", "progressive house", "tech house", "pop", "anthemic", "high energy", "sunset", "festival", "dark"]

index = 0
for i in range(100):
	prompt = []

	if random.random() < 0.5:
		prompt.append(random.choice(ARTISTS))

	for tag in TAGS:
		if random.random() < 0.1:
			prompt.append(tag)

	if random.random() < 0.3:
		prompt.append("vocals")
	elif random.random() < 0.3:
		prompt.append("instrumental")

	prompt = ", ".join(prompt)

	if prompt == "":
		prompt = "lane 8, vocals"

	steps = random.choice([999])
	guidance_scale = random.randint(6, 13)

	# prompt = "lane 8, instrumental"
	# prompt = "kaskade, progressive house, vocals"
	print(f"i {index} prompt {prompt} steps {steps} scale {guidance_scale}")

	images = pipe(prompt, 512, 512, num_images_per_prompt=1,
				  num_inference_steps=steps, guidance_scale=guidance_scale)

	
	for image in images.images:
		image.save(f"generated_new_v9/test{index}.png")
		index += 1
