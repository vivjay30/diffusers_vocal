import argparse
import yaml

from diffusers import DiffusionPipeline
from diffusers import PNDMScheduler, DDIMScheduler
from dps_model.dps_unet import create_model
from dps_model.measurements import SuperResolutionOperator
from random_pixel_selector import ConsistentRandomPixelSelector

import torch.nn.functional as F


from PIL import Image
import numpy as np
import torch
import json

super_res_operator = SuperResolutionOperator((1, 3, 256, 256), 4, torch.device("cuda"))


def create_mask_function(reference_tensor):
    # Ensure the input tensor has the correct shape and type
    assert reference_tensor.shape == (1, 3, 256, 256)
    assert reference_tensor.dtype == torch.uint8

    # Create a boolean mask where True indicates non-zero values
    reference_tensor = reference_tensor.to(torch.float).mean(dim=1, keepdim=True)
    mask = (reference_tensor > 30).to(reference_tensor.device)

    # Create and return a function that applies this mask to new tensors
    def mask_function(new_tensor):
        # Ensure the new tensor has the same shape and type
        assert new_tensor.shape == (1, 3, 256, 256)

        # Apply the mask (this will convert the result to a float tensor)
        masked_tensor = new_tensor * mask
        return masked_tensor

    return mask_function


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main(args):
    print(f"i {args.original_image_prefix}")
    # generator = DiffusionPipeline.from_pretrained("ffhq_jun4_256").to("cuda")
    # image = generator().images[0]
    # image.save("test.png")
    generator = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256").to("cuda")

    model_config = load_yaml("dps_model/model_config.yaml")
    # Load model
    model = create_model(**model_config)
    model = model.to("cuda")
    model.eval()

    # # Load the weights from the file
    # weights = torch.load('ffhq_10m.pt')

    # # Get the state dictionary of generator.unet
    # unet_state_dict = generator.unet.state_dict()

    # # Create a new state dictionary to hold the mapped weights
    # mapped_weights = {}

    # # Mapping function (this may need to be adjusted based on the exact structure)
    # def map_key(key):
    #     mapping = {
    #         'time_embed': 'time_embedding',
    #         'input_blocks': 'down_blocks',
    #         'middle_block': 'mid_block',
    #         'output_blocks': 'up_blocks',
    #         'out': 'conv_out',
    #     }
    #     for k in mapping:
    #         if key.startswith(k):
    #             return key.replace(k, mapping[k])
    #     return key

    # # Map the keys from weights to unet_state_dict
    # for key in weights.keys():
    #     mapped_key = map_key(key)
    #     if mapped_key in unet_state_dict:
    #         mapped_weights[mapped_key] = weights[key]

    # # Load the mapped weights into generator.unet
    # unet_state_dict.update(mapped_weights)
    # generator.unet.load_state_dict(unet_state_dict, strict=False)


    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        skip_prk_steps=True,
        prediction_type="epsilon",
        timestep_spacing="leading",
        steps_offset=0,
    )

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        timestep_spacing="leading",
        steps_offset=0,
    )

    generator.scheduler = ddim_scheduler

    NOISE = 0.1
    # OPERATOR = super_res_operator.forward
    OPERATOR = ConsistentRandomPixelSelector(device="cuda")
    # OPERATOR = neighboring_diffs
    # OPERATOR = operator_A

    # generator = DiffusionPipeline.from_pretrained("Hug-fsneng1/ddpm-phoenix-512").to("cuda")
    # orig_image = Image.open(f"/gscratch/realitylab/vjayaram/diffusers/examples/text_to_image/face_input_512.jpg")
    # orig_image = Image.open(f"/gscratch/realitylab/vjayaram/celebhq/{args.original_image_prefix}.jpg")
    try:
        original_image = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset/ffhq256/test/{args.original_image_prefix}.png")
    except:
        original_image = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset/ffhq256/train/{args.original_image_prefix}.png")


    # original_image = Image.open(f"partial_4.png")
    # mask = Image.open(f"mask_4.png")
    # gt = Image.open(f"gt_4.png")qgitq
    # orig_image = Image.open(f"sparse_3d.png")
    # orig_image = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset//{args.original_image_prefix}.png")
    # orig_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/celeb_256.jpeg")
    # orig_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/lsun_church.png")
    orig_image = np.array(original_image.resize((256, 256), Image.BICUBIC))
    # orig_image = np.array(orig_image)
    orig_image = torch.from_numpy(orig_image).unsqueeze(0).permute(0, 3, 1, 2)
    orig_image = orig_image.to("cuda")[:, :3]
    # OPERATOR = create_mask_function(torch.from_numpy(np.array(original_image)).unsqueeze(0).permute(0, 3, 1, 2).to("cuda")[:, :3])
    orig_image = orig_image / 127.5 - 1.0 # Convert to (-1, 1)
    orig_image = orig_image.to(torch.float)
    observation = OPERATOR(orig_image)
    # orig_image_denoised = torch.from_numpy(np.array(gt)).unsqueeze(0).permute(0, 3, 1, 2).to("cuda")[:, :3] / 127.5 - 1.0

    observation += torch.randn_like(observation) * NOISE

    observation_visualize = (observation.permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5).clip(0, 1)
    output = generator.numpy_to_pil(observation_visualize)[0]#.resize((256, 256))
    output.save(f"test_1.png")

    # mnist_image = np.array(Image.open("cifar_example.png"))
    # orig_image = mnist_image / 127.5

    result = generator(
        original_image=orig_image,
        observation=observation,
        noise=NOISE,
        num_inference_steps=20,
        K=args.K,
        dps_model=model,
        operator=OPERATOR,
        original_image_denoised=None)
    image, grads = result[0].images[0], result[1]

    # with open(f"grad_experiments/random_inpainting/ffhq/20_steps_kl/{args.original_image_prefix}_grads.json", "w") as f:
    #     json.dump(grads, f)

    # image.save(f"grad_experiments/random_inpainting/ffhq/100_steps_l2/{args.original_image_prefix}.png")
    image.save(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/ffhq/inpainting_random/kl_20steps_{args.K}k/{args.original_image_prefix}.png")
    # image.save(f"test.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("original_image_prefix", type=str)
    parser.add_argument("K", type=int)
    main(parser.parse_args())
