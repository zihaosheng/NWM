"""Video generation with ego motion and complete bounding-box controls."""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import tensor2vid
from diffusers.utils.torch_utils import randn_tensor


class NWMPipeline(StableVideoDiffusionPipeline):
    def __init__(
        self, vae, image_encoder, unet, controlnet, scheduler, feature_extractor, null_model
    ):
        DiffusionPipeline.__init__(self)
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            null_model=null_model,
        )
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.inference_mode()
    def __call__(
        self,
        image,
        bbox_images,
        ego_trajectory,
        *,
        num_inference_steps=30,
        min_guidance_scale=1.0,
        max_guidance_scale=9.0,
        decode_chunk_size=8,
        generator=None
    ):
        """Return 25 RGB frames at 512 x 320, conditioned at every frame."""
        device = self._execution_device
        height, width, num_frames = 320, 512, 25
        image_embeddings = self._encode_image(image, device, 1, False)
        dtype = image_embeddings.dtype

        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + 0.01 * noise

        vae_dtype = self.vae.dtype
        if vae_dtype == torch.float16 and self.vae.config.force_upcast:
            self.vae.to(dtype=torch.float32)
        image_latents = self._encode_vae_image(image, device, 1, False).to(dtype)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # Encode all spatial controls without applying the VAE scaling factor.
        boxes = bbox_images.to(device=device, dtype=self.vae.dtype)
        box_latents = self.vae.encode(boxes).latent_dist.mode().unsqueeze(0).to(dtype)
        trajectory = ego_trajectory.reshape(1, 8).to(device=device, dtype=dtype)

        # Stable Video Diffusion uses fps - 1 in the additional time embedding.
        added_time_ids = self._get_add_time_ids(5, 127, 0.01, dtype, 1, 1, False).to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self.prepare_latents(
            1,
            num_frames,
            self.unet.config.in_channels,
            height,
            width,
            dtype,
            device,
            generator,
            None,
        )
        guidance = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames)
        guidance = guidance.to(device=device, dtype=dtype).reshape(1, num_frames, 1, 1, 1)

        for timestep in self.progress_bar(self.scheduler.timesteps):
            model_input = self.scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat([model_input, image_latents], dim=2)
            down_residuals, mid_residual = self.controlnet(
                model_input,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                control_cond=box_latents,
                ego_traj=trajectory,
                return_dict=False,
            )
            prediction = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residuals=mid_residual,
                return_dict=False,
            )[0]
            reference = self.null_model(
                model_input,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]
            prediction = reference + guidance * (prediction - reference)
            latents = self.scheduler.step(prediction, timestep, latents).prev_sample

        frames = self.decode_latents(latents.to(self.vae.dtype), num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type="np")[0]
        self.vae.to(dtype=vae_dtype)
        return frames
