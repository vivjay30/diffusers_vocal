# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import os
import torch

from torchvision.transforms import ToPILImage

from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


def save_to_image(tensor, filename):
    to_save = (tensor + 1) / 2
    to_save = to_save.clamp(0, 1)

    # Convert to PIL Image
    transform = ToPILImage()
    img = transform(to_save)

    # Save the image
    img.save(filename)


class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        original_image = None
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # image = self.scheduler.add_noise(original_image, torch.randn_like(image), torch.LongTensor([t]))
            # save_to_image(image[0], os.path.join("intermediate_outputs_ours_ffhq", f"{t}.png"))

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t-1] if t-1 >= 0 else self.scheduler.one
            alpha_prod_t_prev_prev = self.scheduler.alphas_cumprod[t-2] if t-2 >= 0 else self.scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t
            current_beta_variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

            eta = 0.1 * current_beta_variance # 1 / (1 / (current_beta_variance) + (alpha_prod_t_prev) / (1 - alpha_prod_t_prev))
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
            z_prev_mu = image.clone().detach()
            image = image + torch.randn_like(image) * (current_beta_variance ** 0.5)
            z_0 = (image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            
            # if i % 10 == 0:
            save_to_image(image[0], os.path.join("intermediate_outputs_ours_ffhq", f"{t}.png"))
            save_to_image(z_0[0], os.path.join("intermediate_outputs_ours_ffhq", f"{t}_z_0.png"))

            for _ in range(100):
                if t <= 1:
                    # Eta is 0 at t = 0
                    break
                image = image.clone().detach().requires_grad_()
                with torch.enable_grad():
                    model_output = self.unet(image, t - 1).sample
                    z_0 = (image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                    next_image = self.scheduler.step(model_output, t - 1, image, generator=generator).prev_sample
                    logprob = 1/2 * (alpha_prod_t_prev / (1 - alpha_prod_t_prev)) * (z_0 - original_image) ** 2
                    logprob[:, :, 128:, :].sum().backward()
                    print(((z_0 - original_image) ** 2).mean())
                # likelihood_grad = (1 / (alpha_prod_t_prev ** 0.5) * image - original_image) * (1 / alpha_prod_t_prev ** 0.5) * (alpha_prod_t_prev / (1 - alpha_prod_t_prev))
                z_pred_grad = (image - z_prev_mu) / current_beta_variance
                # superres = torch.zeros_like(likelihood_grad)
                # superres[:, :, ::8, ::8] = 1
                # likelihood_grad *= superres
                # likelihood_grad[:, :, :128, :] = 0
                image -= (eta * (2 * image.grad + z_pred_grad) + (2 * eta) ** 0.5 * torch.randn_like(image))



        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
