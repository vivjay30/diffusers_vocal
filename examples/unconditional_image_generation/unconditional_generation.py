import argparse
import yaml

from diffusers import DiffusionPipeline
from diffusers import PNDMScheduler, DDIMScheduler
from dps_model.dps_unet import create_model
from dps_model.measurements import SuperResolutionOperator
from random_pixel_selector import ConsistentRandomPixelSelector
from random_box_selector import ConsistentBoxMasker
from dps_model.measurements import GaussialBlurOperator

import torch.nn.functional as F


from PIL import Image
import numpy as np
import torch
import json


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
    print("here-1")
    generator = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256").to("cuda")
    print("here0")

    model_config = load_yaml("dps_model/model_config.yaml")
    # Load model
    model = create_model(**model_config)
    model = model.to("cuda")
    model.eval()
    print("here1")

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
    # super_res_operator = SuperResolutionOperator((1, 3, 256, 256), 4, torch.device("cuda"))
    # OPERATOR = super_res_operator.forward

    OPERATOR = ConsistentRandomPixelSelector(device="cuda")

    # OPERATOR = ConsistentBoxMasker(device="cuda")

    # gaussian_blur_operator = GaussialBlurOperator(61, 3.0, torch.device("cuda"))
    # OPERATOR = gaussian_blur_operator.forward

    # OPERATOR = neighboring_diffs
    # OPERATOR = operator_A
    # OPERATOR = lambda x: x
    # OPERATOR = lambda x: x[:, :, 128:, :]
    # OPERATOR = lambda x: x[:, 2:, :, :]

    # generator = DiffusionPipeline.from_pretrained("Hug-fsneng1/ddpm-phoenix-512").to("cuda")
    # orig_image = Image.open(f"/gscratch/realitylab/vjayaram/diffusers/examples/text_to_image/face_input_512.jpg")
    # orig_image = Image.open(f"/gscratch/realitylab/vjayaram/celebhq/{args.original_image_prefix}.jpg")
    try:
        # original_image = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset/ffhq256/test/{args.original_image_prefix}.png")
        original_image = Image.open(f"/gscratch/realitylab/vjayaram/imagenet_val/{args.original_image_prefix}.png")

    except:
        # original_image = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset/ffhq256/train/{args.original_image_prefix}.png")
        pass

    torch.manual_seed(np.random.randint(0, 1000))
    # original_image = Image.open(f"abe_lincoln.jpg")

    # mask = Image.open(f"mask_4.png")
    # gt = Image.open(f"gt_4.png")qgitq
    # orig_image = Image.open(f"sparse_3d.png")
    # orig_image = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset//{args.original_image_prefix}.png")
    # orig_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/celeb_256.jpeg")
    # orig_image = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/lsun_church.png")
    orig_image = np.array(original_image.resize((256, 256), Image.BICUBIC))

    # POISON NOISE
    # orig_image = np.random.poisson(lam=(orig_image * 0.017).astype(np.uint8)) / 0.017

    # orig_image = np.array(orig_image)
    orig_image = torch.from_numpy(orig_image).unsqueeze(0).permute(0, 3, 1, 2)
    orig_image = orig_image.to("cuda")[:, :3]
    # OPERATOR = create_mask_function(torch.from_numpy(np.array(original_image)).unsqueeze(0).permute(0, 3, 1, 2).to("cuda")[:, :3])
    orig_image = orig_image / 127.5 - 1.0 # Convert to (-1, 1)
    orig_image = orig_image.to(torch.float)
    with torch.no_grad():
        observation = OPERATOR(orig_image)
    # orig_image_denoised = torch.from_numpy(np.array(gt)).unsqueeze(0).permute(0, 3, 1, 2).to("cuda")[:, :3] / 127.5 - 1.0

    observation += torch.randn_like(observation) * NOISE
    # observation += (torch.randint(low=0, high=2, size=observation.shape) * 2 - 1).to("cuda")

    observation_visualize = (observation.permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5).clip(0, 1)
    output = generator.numpy_to_pil(observation_visualize)[0]#.resize((256, 256))
    output.save(f"test_1.png")
    # output.save(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/imagenet/superres/input/{args.original_image_prefix}.png")
    # return
    # mnist_image = np.array(Image.open("cifar_example.png"))
    # orig_image = mnist_image / 127.5

    import pdb
    pdb.set_trace()
    NUM_STEPS = 50
    result = generator(
        original_image=orig_image,
        observation=observation,
        noise=NOISE,
        num_inference_steps=NUM_STEPS,
        K=args.K,
        dps_model=model,
        operator=OPERATOR,
        original_image_denoised=None)
    image, grads = result[0].images[0], result[1]

    # with open(f"grad_experiments/gaussian_blur/ffhq/50_steps_kl/{args.original_image_prefix}_grads.json", "w") as f:
    #     json.dump(grads, f)

    # image.save(f"grad_experiments/random_inpainting/ffhq/100_steps_l2/{args.original_image_prefix}.png")
    # image.save(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/imagenet/superres/l2_{NUM_STEPS}steps_{args.K}k/{args.original_image_prefix}.png")
    # image.save(f"lincoln/{steps}_{np.random.randint(0, 1000000)}.png")
    image.save(f"test.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("original_image_prefix", type=str)
    parser.add_argument("K", type=int)
    main(parser.parse_args())
