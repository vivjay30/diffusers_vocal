from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler
import torch
import os
import numpy as np
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.embeddings import TimestepEmbedding, Timesteps



from PIL import Image

# pipe = StableDiffusionPipeline.from_pretrained("sd-pokemon-model")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
ddpm_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    prediction_type="epsilon")
# ddpm_scheduler = DDIMScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     prediction_type="epsilon")
# ddpm_scheduler.set_timesteps(1000)
pipe.scheduler_alternate = ddpm_scheduler

pipe.safety_checker = None
pipe = pipe.to("cuda")

# orig_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/text_to_image/machamp.png")
orig_image = Image.open("/gscratch/realitylab/vjayaram/ffhq-dataset/test/69908.png")
# =orig_image = Image.open("/gscratch/realitylab/vjayaram/vocal_diffusion/bad_singing/amazing_grace_bad_0.png")
orig_image = np.array(orig_image.resize((512, 512)))
orig_image = orig_image / 127.5 - 1.0 # Convert to (-1, 1)
orig_image = torch.from_numpy(orig_image).unsqueeze(0).permute(0, 3, 1, 2)
orig_image = orig_image.to("cuda")[:, :3]
orig_image = orig_image.to(torch.float)

with torch.no_grad():
    encoded = pipe.vae.encode(orig_image)
    latents = encoded.latent_dist.sample()
    latents *= pipe.vae.config.scaling_factor

likelihood_model = Decoder(
    in_channels=4,
    out_channels=6,
    up_block_types=[
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    norm_num_groups=32,
    act_fn="silu",
    actual_temb_channels=512
)

state_dict = torch.load(os.path.join("ffhq-likelihood", "vae_trainable_may8.pth"))
# Remove 'module.' prefix
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load the adjusted state dictionary into your model
likelihood_model.load_state_dict(new_state_dict)
likelihood_model = likelihood_model.to("cuda")
likelihood_model.requires_grad_(True)
likelihood_model.eval()
for name, param in likelihood_model.named_parameters():
    print(name, param.requires_grad)


# Load the Embedding
time_embed_dim = 512
timestep_input_dim = 128
time_proj = Timesteps(128, flip_sin_to_cos=True, downscale_freq_shift=0)
time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, "silu")
state_dict = torch.load(os.path.join("ffhq-likelihood", "time_embeddings_may8.pth"))
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
time_embedding.load_state_dict(new_state_dict)
time_embedding.requires_grad_(True)
time_embedding.eval()
time_embedding = time_embedding.to("cuda")

# For doing the last little bit of diffusion
# noisy_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/text_to_image/intermediate_outputs/900.png")
# # =orig_image = Image.open("/gscratch/realitylab/vjayaram/vocal_diffusion/bad_singing/amazing_grace_bad_0.png")
# noisy_image = np.array(noisy_image.resize((512, 512)))
# noisy_image = noisy_image / 127.5 - 1.0 # Convert to (-1, 1)
# noisy_image = torch.from_numpy(noisy_image).unsqueeze(0).permute(0, 3, 1, 2)
# noisy_image = noisy_image.to("cuda")[:, :3]
# noisy_image = noisy_image.to(torch.float)

# with torch.no_grad():
#     encoded = pipe.vae.encode(orig_image)
#     latents = encoded.latent_dist.sample()
#     latents *= pipe.vae.config.scaling_factor

# # model_id = "riffusion/riffusion-model-v1"

# start_step = 25
# # Hard coded for inference steps 50
# timesteps = [981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
#         721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
#         441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
#         161, 141, 121, 101,  81,  61,  41,  21,   1]

# a_t = pipe.scheduler.alphas_cumprod[timesteps[start_step]]
# latents = latents * (a_t ** 0.5) + ((1 - a_t) ** 0.5) * torch.randn_like(latents)


prompt = "A high quality photo of a face"
image = pipe(prompt, 512, 512,
    num_inference_steps=999, likelihood_model=likelihood_model,
    original_image=orig_image, time_proj=time_proj, time_embedding=time_embedding).images[0]  
# image = pipe(prompt, 512, 512, num_inference_steps=200, constrain_latents=latents).images[0]  
# image = pipe(prompt, 512, 512, latents=latents, num_inference_steps=50, constrain_output=orig_image, constrain_latents=latents).images[0]  

image.save("test.png")