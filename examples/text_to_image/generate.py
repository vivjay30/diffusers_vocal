from diffusers import StableDiffusionPipeline
import torch
import numpy as np


from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("vocal_model_Feb22_riffusion_512_1e-7")
pipe = pipe.to("cuda")

# orig_image = Image.open("/gscratch/realitylab/vjayaram/vocal_diffusion/bad_singing/image_1.png")
orig_image = Image.open("/gscratch/realitylab/vjayaram/vocal_diffusion/bad_singing/amazing_grace_bad_0.png")
orig_image = np.array(orig_image.resize((512, 512)))
orig_image = orig_image / 127.5 - 1.0 # Convert to (-1, 1)
orig_image = torch.from_numpy(orig_image).unsqueeze(0).permute(0, 3, 1, 2)
orig_image = orig_image.to("cuda")
orig_image = orig_image.to(torch.float)

with torch.no_grad():
    encoded = pipe.vae.encode(orig_image)
    latents = encoded.latent_dist.sample()
    latents *= pipe.vae.config.scaling_factor

# model_id = "riffusion/riffusion-model-v1"

start_step = 25
# Hard coded for inference steps 50
timesteps = [981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        161, 141, 121, 101,  81,  61,  41,  21,   1]

a_t = pipe.scheduler.alphas_cumprod[timesteps[start_step]]
latents = latents * (a_t ** 0.5) + ((1 - a_t) ** 0.5) * torch.randn_like(latents)


prompt = "singing"
image = pipe(prompt, 512, 512, num_inference_steps=50).images[0]  
# image = pipe(prompt, 512, 512, num_inference_steps=200, constrain_latents=latents).images[0]  
# image = pipe(prompt, 512, 512, latents=latents, num_inference_steps=50, constrain_output=orig_image, constrain_latents=latents).images[0]  

image.save("test.png")
