from PIL import Image
import numpy as np
import lpips
import torch

all_lpips = []
loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
for i in range(0, 1000):
    # Load the images using PIL
    # img1 = Image.open("test.png")
    # img1 = Image.open(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/ffhq/gaussian_blur/kl_50steps_3k/{i:05d}.png")
    # img2 = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset/ffhq256/train/{i:05d}.png")

    img1 = Image.open(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/imagenet/superres/kl_25steps_1k/{i:05d}.png")
    img2 = Image.open(f"/gscratch/realitylab/vjayaram/imagenet_val/{i:05d}.png")

    img1_array = np.array(img1).astype(np.float32) / 127.5 - 1.0
    img2_array = np.array(img2).astype(np.float32) / 127.5 - 1.0

    img1_array = torch.Tensor(img1_array).to("cuda").permute(2, 0, 1)
    img2_array = torch.Tensor(img2_array).to("cuda").permute(2, 0, 1)

    # Calculate PSNR
    lpips_value = loss_fn_alex(img1_array, img2_array).item()
    print(f"i {i} LPIPS: {lpips_value} dB")
    all_lpips.append(lpips_value)
    print(np.mean(all_lpips))

print(np.mean(all_lpips))
