# Copyright 2025 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import argparse
import json
import time
import os
from PIL import Image

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import LCMScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from rknnlite.api import RKNNLite

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import DiffusionPipeline
        >>> import torch

        >>> pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        >>> prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

        >>> # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        >>> num_inference_steps = 4
        >>> images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0).images
        >>> images[0].save("image.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class RKNN2Model:
    """ Wrapper for running RKNPU2 models """

    def __init__(self, model_dir : str, unload_after_first_call : bool = True):
        
        # Ensure the model exists
        assert os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "model.rknn"))
        
        # Save the model path and config
        self.model_dir = os.path.join(model_dir, "model.rknn")
        self.config = json.load(open(os.path.join(model_dir, "config.json")))
        self.modelname = self.model_dir.split("/")[-1]
        
        # Only cretae the RKNNLite object without loading it
        self.rknnlite = RKNNLite()
        
        # Variables to control the load of the model (to save memory in the pipeline)
        self.loaded = False
        self.unload_after_first_call = unload_after_first_call


    def __call__(self, **kwargs) -> List[np.ndarray]:
       
        # Load the model if not yet loaded
        if not self.loaded:
            self.load_model()      
        
       # Construct the list of input and transform them to Numpy if needed
        input_list = [value for key, value in kwargs.items()]
        input_list_np = []
        for i, input in enumerate(input_list):
            print(f"INPUT {i}: type={type(input)}, shape={None if not hasattr(input, 'shape') else input.shape}")
            if isinstance(input, np.ndarray):
                input_list_np.append(input)
            else:
                input_np = input.detach().cpu().numpy().astype(np.float32)
                input_list_np.append(input_np)
                print(f"** TRANSFORMED** -> INPUT {i}: type={type(input_np)}, shape={None if not hasattr(input_np, 'shape') else input_np.shape}")

        # Call the RKNN model wit the input
        results = self.rknnlite.inference(inputs=input_list_np, data_format='nchw')
        for res in results:
            print(f"Output shape: {res.shape}")
        
        # Unload the model automatically if requried
        if self.unload_after_first_call:
           self.unload_model()
        
        # Return the result
        return results

    
    def load_model(self):
        
        logger.info(f"Loading model: {self.model_dir}")
        
        # Take the init time
        start = time.time()
        
        # Load the model and run time of RKNN
        self.rknnlite.load_rknn(self.model_dir)
        self.rknnlite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        
        # Take the finish time
        load_time = time.time() - start
        
        logger.info(f"Load Done. Took {load_time:.1f} seconds.")
        
        # Set the loaded flag
        self.loaded = True

    
    def unload_model(self):
        
        logger.info(f"Unloading model: {self.model_dir}")
        
        # Take the init time
        start = time.time()
        
        # Unload the model
        self.rknnlite.release()
        
        # Take the finish time
        load_time = time.time() - start
        
        logger.info(f"Unload Done. Took {load_time:.1f} seconds.")
        
        # Set the loaded flag
        self.loaded = False


class RKNNLatentConsistencyModelPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using a latent consistency model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae_decoder ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Currently only
            supports [`LCMScheduler`].
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
    
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "denoised", "prompt_embeds", "w_embedding"]

    def __init__(
        self,
        vae_decoder: RKNN2Model,
        text_encoder: RKNN2Model,
        tokenizer: CLIPTokenizer,
        unet: RKNN2Model,
        scheduler: LCMScheduler,
        tokenizer_2: CLIPTokenizer,
        text_encoder_2: Optional[RKNN2Model] = None
    ):
        super().__init__()

        self.register_modules(
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            tokenizer_2 = tokenizer_2,
            text_encoder_2 = text_encoder_2,
        )
        
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.vae_scale_factor = 2 ** (len(self.vae_decoder.config["block_out_channels"]) - 1) if getattr(self, "vae_decoder", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # tokenizer and  text_encoder
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
       
            prompt_embeds = self.text_encoder(input_ids=text_input_ids.to(device).numpy().astype(np.float32))

            # Initialize the optional second embeds
            text_embeds = None
            
            if self.text_encoder_2:
                # tokenizer_2 and  text_encoder_2
                text_inputs_2 = self.tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids_2 = text_inputs_2.input_ids
                untruncated_ids_2 = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids_2.shape[-1] >= text_input_ids_2.shape[-1] and not torch.equal(
                    text_input_ids_2, untruncated_ids_2
                ):
                    removed_text_2 = self.tokenizer_2.batch_decode(
                        untruncated_ids_2[:, self.tokenizer_2.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer_2.model_max_length} tokens: {removed_text_2}"
                    )
        
                text_embeds = self.text_encoder_2(input_ids=text_input_ids_2.to(device).numpy().astype(np.float32))

        # Return the two embeddings
        return prompt_embeds, text_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, np.random.RandomState):
                latents = generator.randn(*shape).astype(dtype)
            elif isinstance(generator, torch.Generator):
                latents = torch.randn(*shape, generator=generator).numpy().astype(dtype)
            else:
                raise ValueError(
                    f"Expected `generator` to be of type `np.random.RandomState` or `torch.Generator`, but got"
                    f" {type(generator)}."
                )
        elif latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        return latents

    
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=None):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings
        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        w = w * 1000
        half_dim = embedding_dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
        emb = w[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)

        if embedding_dim % 2 == 1:  # zero pad
            emb = np.pad(emb, [(0, 0), (0, 1)])

        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config["addition_time_embed_dim"] * len(add_time_ids) + text_encoder_projection_dim
        )
        #expected_add_embed_dim = self.unet.add_embedding.linear_1.in_featuores
        expected_add_embed_dim = passed_add_embed_dim

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    # Currently StableDiffusionPipeline.check_inputs with negative prompt stuff removed
    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        callback_steps: int,
        prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return False

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        original_inference_steps: int = None,
        timesteps: List[int] = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[np.random.RandomState, torch.Generator]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds:Optional[np.ndarray] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps use to generate a linearly-spaced timestep schedule, from which
                we will draw `num_inference_steps` evenly spaced timesteps from as our final timestep schedule,
                following the Skipping-Step method in the paper (see Section 4.3). If not set this will default to the
                scheduler's `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps on the original LCM training/distillation timestep schedule are used. Must be in descending
                order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                Note that the original latent consistency models paper uses a different CFG formulation where the
                guidance scales are decreased by 1 (so in the paper formulation CFG is enabled when `guidance_scale >
                0`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`Optional[Union[np.random.RandomState, torch.Generator]]`, defaults to `None`):
                A np.random.RandomState to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        original_size = (height, width)
        target_size = (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if generator is None:
            generator = np.random.RandomState()
        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        # NOTE: when a LCM is distilled from an LDM via latent consistency distillation (Algorithm 1) with guided
        # distillation, the forward pass of the LCM learns to approximate sampling from the LDM using CFG with the
        # unconditional prompt "" (the empty string). Due to this, LCMs currently do not support negative prompts.
        start_time = time.time()
        prompt_embeds_np, text_embeds_np = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        encode_prompt_time = time.time() - start_time
        logger.debug(f"Prompt encoding time: {encode_prompt_time:.2f}s")
        
        prompt_embeds = prompt_embeds_np[0]
        text_embeds = text_embeds_np[0] if text_embeds_np else None
        
          
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )

        # 5. Prepare latent variable
        num_channels_latents = self.unet.config["in_channels"]
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
	    
        bs = batch_size * num_images_per_prompt

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = np.int64

        # 6. Get Guidance Scale Embedding
        w = np.full(bs, guidance_scale - 1, dtype=prompt_embeds.dtype)
        w_embedding = self.get_guidance_scale_embedding(
            w, embedding_dim=self.unet.config["time_cond_proj_dim"], dtype=prompt_embeds.dtype
        )

        # 7. Prepare added time ids if required
        if self.text_encoder_2:
            text_encoder_projection_dim = self.text_encoder_2.config["projection_dim"]
            time_ids = self._get_add_time_ids(
                original_size,
                (0, 0), # crops_coords_top_left
                target_size,
                dtype=torch.float32,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            time_ids = time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. LCM MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        # Begin image generation
        inference_start = time.time()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Auxiliary calculations
                timestep = np.array([t], dtype=timestep_dtype)

                # model prediction (v-prediction, eps, x)
                if self.text_encoder_2:
                   # LCM SSD1B

                   # Auxiliary calculations
                   hidden_state = np.concatenate([prompt_embeds, text_embeds_np[1]], axis=-1)

                   # Run the model
                   model_pred = self.unet(
                      sample=latents,
                      timestep=timestep,
                      encoder_hidden_states=hidden_state,
                      timestep_cond=w_embedding,
                      time_ids=time_ids,
                      text_embeds=text_embeds,  
                   )[0]
                
                else:
                   # LCM SSD 1.5

                   # Run the model
                   model_pred = self.unet(
                      sample=latents,
                      timestep=timestep,
                      encoder_hidden_states=prompt_embeds,
                      timestep_cond=w_embedding
                   )[0]
                
                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = self.scheduler.step(torch.from_numpy(model_pred), t, torch.from_numpy(latents), return_dict=False)
                latents, denoised = latents.numpy(), denoised.numpy()
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        
        inference_time = time.time() - inference_start
        logger.debug(f"Inference time: {inference_time:.2f}s")

        # Unload the UNET model for memory saving
        self.unet.unload_model()

        #denoised = denoised.to(prompt_embeds.dtype)
        decode_start = time.time()
        if not output_type == "latent":
            denoised /= self.vae_decoder.config["scaling_factor"]
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=denoised[i : i + 1])[0] for i in range(denoised.shape[0])]
            )
            # image, has_nsfw_concept = self.run_safety_checker(image)
            has_nsfw_concept = None  # skip safety checker
        else:
            image = denoised
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        decode_time = time.time() - decode_start
        print(f"Decode time: {decode_time:.2f}s")

        total_time = encode_prompt_time + inference_time + decode_time
        print(f"Total time: {total_time:.2f}s")

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    @staticmethod
    def postprocess(
        image: np.ndarray,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
        ):
        def numpy_to_pil(images: np.ndarray):
            """
            Convert a numpy image or a batch of images to a PIL image.
            """
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                pil_images = [Image.fromarray(image) for image in images]

            return pil_images
        
        def denormalize(images: np.ndarray):
            """
            Denormalize an image array to [0,1].
            """
            return np.clip(images / 2 + 0.5, 0, 1)
    
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support np array"
            )
        if output_type not in ["latent", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            logger.warning(deprecation_message)
            output_type = "np"

        if output_type == "latent":
            return image
        
        if do_denormalize is None:
            raise ValueError("do_denormalize is required for postprocessing")

        image = np.stack(
            [denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])], axis=0
        )
        image = image.transpose((0, 2, 3, 1))

        if output_type == "pil":
            image = numpy_to_pil(image)

        return image


def generate_image(model_dir, prompt, size, seed, num_inference_steps, guidance_scale):
    logger.info(f"Setting random seed to {seed}")

    # load scheduler from /scheduler/scheduler_config.json
    scheduler_config_path = os.path.join(model_dir, "scheduler/scheduler_config.json")
    with open(scheduler_config_path, "r") as f:
        scheduler_config = json.load(f)
    user_specified_scheduler = LCMScheduler.from_config(scheduler_config)

    logger.debug("user_specified_scheduler", user_specified_scheduler)

    if os.path.exists(os.path.join(model_dir, "text_encoder_2")):
        logger.info(f"Running LCM SSD1B")
        pipe = RKNNLatentConsistencyModelPipeline(
	        scheduler=user_specified_scheduler,
            vae_decoder=RKNN2Model(os.path.join(model_dir, "vae_decoder")),
            unet=RKNN2Model(os.path.join(model_dir, "unet"), False),
            text_encoder=RKNN2Model(os.path.join(model_dir, "text_encoder")),
            text_encoder_2=RKNN2Model(os.path.join(model_dir, "text_encoder_2")),
            tokenizer=CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"),
	        tokenizer_2=CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"),
        )
    else:
        logger.info(f"Running LCM SD 1.5")
        pipe = RKNNLatentConsistencyModelPipeline(
            scheduler=user_specified_scheduler,
            vae_decoder=RKNN2Model(os.path.join(model_dir, "vae_decoder")),
            unet=RKNN2Model(os.path.join(model_dir, "unet"), False),
            text_encoder=RKNN2Model(os.path.join(model_dir, "text_encoder")),
            text_encoder_2=None,
            tokenizer=CLIPTokenizer.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="tokenizer"),
            tokenizer_2=None,
        )

    logger.info("Beginning image generation.")
    image = pipe(
        prompt=prompt,
        height=int(size.split("x")[0]),
        width=int(size.split("x")[1]),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=np.random.RandomState(seed),
    )

    # Return the image
    return image["images"][0]