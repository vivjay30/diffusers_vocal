from PIL import Image
import numpy as np


# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

all_psnrs = []
for i in range(0, 1000):
    try:
        # Load the images using PIL
        # img1 = Image.open("test.png")
        # img1 = Image.open(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/ffhq/inpainting_box/kl_25steps_1k/{i:05d}.png")
        # img2 = Image.open("/gscratch/realitylab/vjayaram/celebhq/00000.jpg")
        # img2 = Image.open(f"/gscratch/realitylab/vjayaram/ffhq-dataset/ffhq256/train/{i:05d}.png")
        # img2 = Image.open("/gscratch/realitylab/vjayaram/diffusers/examples/text_to_image/face_input_512.jpg")

        img1 = Image.open(f"/gscratch/realitylab/vjayaram/diffusers/examples/unconditional_image_generation/experiments/imagenet/inpainting_random/l2_25steps_1k/{i:05d}.png")
        img2 = Image.open(f"/gscratch/realitylab/vjayaram/imagenet_val/{i:05d}.png")

        # Resize the 1024x1024 image to match the dimensions of the 512x512 image
        # img1_resized = img1.resize((256, 256), Image.BILINEAR)
        # img2 = img2.resize((256, 256), Image.BICUBIC)

        # Convert images to NumPy arrays
        img1_array = np.array(img1).astype(np.float32)
        img2_array = np.array(img2).astype(np.float32)


        # Calculate PSNR
        psnr_value = calculate_psnr(img1_array, img2_array)
        print(f"i {i} PSNR: {psnr_value} dB")
        all_psnrs.append(psnr_value)
        print(np.mean(all_psnrs))
    except:
        continue

print(np.mean(all_psnrs))
