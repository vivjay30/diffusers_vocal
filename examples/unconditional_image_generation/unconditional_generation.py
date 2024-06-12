from diffusers import DiffusionPipeline

from PIL import Image
import numpy as np
import torch

# generator = DiffusionPipeline.from_pretrained("ffhq_jun4_256").to("cuda")
# image = generator().images[0]
# image.save("test.png")

generator = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256").to("cuda")
orig_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/celeb_sample.jpeg")
orig_image = np.array(orig_image.resize((256, 256)))
orig_image = orig_image / 127.5 - 1.0 # Convert to (-1, 1)
orig_image = torch.from_numpy(orig_image).unsqueeze(0).permute(0, 3, 1, 2)
orig_image = orig_image.to("cuda")[:, :3]
orig_image = orig_image.to(torch.float)

image = generator(original_image=orig_image).images[0]

image.save("test.png")
