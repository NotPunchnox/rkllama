
import argparse
import json
import time

import PIL
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import (
    LCMScheduler
)

import logging

logging.basicConfig()
logger = logging.getLogger("rkllama.image_generator")
logger.setLevel(logging.INFO)

import numpy as np
import os

import torch  # Only used for `torch.from_tensor` in `pipe.scheduler.step()`
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from typing import Callable, List, Optional, Union, Tuple
from PIL import Image

from rknnlite.api import RKNNLite

class RKNN2Model:
    """ Wrapper for running RKNPU2 models """

    def __init__(self, model_dir):
        logger.info(f"Loading {model_dir}")
        start = time.time()
        self.config = json.load(open(os.path.join(model_dir, "config.json")))
        assert os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "model.rknn"))
        self.rknnlite = RKNNLite()
        self.rknnlite.load_rknn(os.path.join(model_dir, "model.rknn"))
        self.rknnlite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) # Multi-core will cause kernel crash
        load_time = time.time() - start
        logger.info(f"Done. Took {load_time:.1f} seconds.")
        self.modelname = model_dir.split("/")[-1]
        self.inference_time = 0

    def __call__(self, **kwargs) -> List[np.ndarray]:
        # np.savez(f"rknn_out/{self.modelname}_input_{self.inference_time}.npz", **kwargs)
        # self.inference_time += 1
        #print(kwargs)
        input_list = [value for key, value in kwargs.items()]
        for i, input in enumerate(input_list):
            if isinstance(input, np.ndarray):
                logger.debug(f"input {i} shape: {input.shape}")

        results = self.rknnlite.inference(inputs=input_list, data_format='nchw')
        
        for res in results:
            logger.debug(f"output shape: {res.shape}")
        return results

class RKNN2LatentConsistencyPipeline(DiffusionPipeline):

    def __init__(
            self,
            text_encoder: RKNN2Model,
            unet: RKNN2Model,
            vae_decoder: RKNN2Model,
            scheduler: LCMScheduler,
            tokenizer: CLIPTokenizer,
            force_zeros_for_empty_prompt: Optional[bool] = True,
            feature_extractor: Optional[CLIPFeatureExtractor] = None,
            text_encoder_2: Optional[RKNN2Model] = None,
            tokenizer_2: Optional[CLIPTokenizer] = None
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt
        self.safety_checker = None

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.vae_decoder = vae_decoder

        VAE_DECODER_UPSAMPLE_FACTOR = 8
        self.vae_scale_factor = VAE_DECODER_UPSAMPLE_FACTOR

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

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, list]],
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`Union[str, List[str]]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L217
    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int],
        width: Optional[int],
        callback_steps: int,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
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

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
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

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.22.0/src/diffusers/pipelines/latent_consistency/pipeline_latent_consistency.py#L264
    def __call__(
        self,
        prompt: Union[str, List[str]] = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        original_inference_steps: int = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[np.random.RandomState, torch.Generator]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`Optional[Union[str, List[str]]]`, defaults to None):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`Optional[int]`, defaults to None):
                The height in pixels of the generated image.
            width (`Optional[int]`, defaults to None):
                The width in pixels of the generated image.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            generator (`Optional[Union[np.random.RandomState, torch.Generator]]`, defaults to `None`):
                A np.random.RandomState to make generation deterministic.
            latents (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (Optional[Callable], defaults to `None`):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            guidance_rescale (`float`, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.unet.config["sample_size"] * self.vae_scale_factor
        width = width or self.unet.config["sample_size"] * self.vae_scale_factor

        # Don't need to get negative prompts due to LCM guided distillation
        negative_prompt = None
        negative_prompt_embeds = None

        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random.RandomState()

        start_time = time.time()
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            False,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        encode_prompt_time = time.time() - start_time
        logger.debug(f"Prompt encoding time: {encode_prompt_time:.2f}s")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, original_inference_steps=original_inference_steps)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config["in_channels"],
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        bs = batch_size * num_images_per_prompt
        # get Guidance Scale Embedding
        w = np.full(bs, guidance_scale - 1, dtype=prompt_embeds.dtype)
        w_embedding = self.get_guidance_scale_embedding(
            w, embedding_dim=self.unet.config["time_cond_proj_dim"], dtype=prompt_embeds.dtype
        )

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = np.int64
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        inference_start = time.time()
        for i, t in enumerate(self.progress_bar(timesteps)):
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=w_embedding,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), return_dict=False
            )
            latents, denoised = latents.numpy(), denoised.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        inference_time = time.time() - inference_start
        logger.debug(f"Inference time: {inference_time:.2f}s")

        decode_start = time.time()
        if output_type == "latent":
            image = denoised
            has_nsfw_concept = None
        else:
            denoised /= self.vae_decoder.config["scaling_factor"]
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=denoised[i : i + 1])[0] for i in range(denoised.shape[0])]
            )
            # image, has_nsfw_concept = self.run_safety_checker(image)
            has_nsfw_concept = None  # skip safety checker

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        decode_time = time.time() - decode_start
        logger.debug(f"Decode time: {decode_time:.2f}s")

        total_time = encode_prompt_time + inference_time + decode_time
        logger.debug(f"Total time: {total_time:.2f}s")

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    # Adapted from https://github.com/huggingface/diffusers/blob/v0.22.0/src/diffusers/pipelines/latent_consistency/pipeline_latent_consistency.py#L264
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


def prepare_controlnet_cond(image_path, height, width):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((height, width), resample=Image.LANCZOS)
    image = np.array(image).transpose(2, 0, 1) / 255.0
    return image


def generate_image(model_dir, prompt, size, seed, num_inference_steps, guidance_scale):
    logger.info(f"Setting random seed to {seed}")

    # load scheduler from /scheduler/scheduler_config.json
    scheduler_config_path = os.path.join(model_dir, "scheduler/scheduler_config.json")
    with open(scheduler_config_path, "r") as f:
        scheduler_config = json.load(f)
    user_specified_scheduler = LCMScheduler.from_config(scheduler_config)

    logger.debug("user_specified_scheduler", user_specified_scheduler)

    pipe = RKNN2LatentConsistencyPipeline(
        text_encoder=RKNN2Model(os.path.join(model_dir, "text_encoder")),
        unet=RKNN2Model(os.path.join(model_dir, "unet")),
        vae_decoder=RKNN2Model(os.path.join(model_dir, "vae_decoder")),
        scheduler=user_specified_scheduler,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16"),
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

    return image["images"][0]
