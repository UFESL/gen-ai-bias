# ***
# Vanilla PyTorch training. Does not use accelerate or other acceleration libraries e.g. lightning.
# ***
import os
import math
import statistics
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    ControlNetModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from PIL import Image
from datetime import datetime

from dim_cn import config
from dim_cn.dataset import ADE20kDataset
from dim_cn.controlnet import DACNConditioningEmbedding
from dim_cn.metrics import get_eval_metrics


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    unet,
    dacn,
    args,
    step,
    weight_dtype,
    log_dir,
):
    print("Running validation")
    device = get_device()

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=dacn,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    validation_prompts = args.validation_prompt
    validation_images = args.validation_image

    inference_ctx = torch.autocast("cuda")

    scores = []

    for validation_prompt, validation_image in zip(
        validation_prompts, validation_images
    ):
        transform = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ]
        )
        im_transform = transforms.ToPILImage()
        validation_im1 = Image.open(f"{validation_image}_segment.png").convert("RGB")
        validation_im1 = transform(validation_im1)
        validation_im1 = im_transform(validation_im1)
        # validation_im2 = Image.open(f"{validation_image}_depth.png").convert("RGB")
        # validation_im2 = transform(validation_im2)
        # overlay = validation_im1 * validation_im2
        # val_image = im_transform(overlay)
        ground_truth = []
        ground_truth1 = Image.open(f"{validation_image}.jpg").convert("RGB")
        ground_truth.append(transform(ground_truth1))
        ground_truth2 = Image.open("ADE_train_00000310.jpg").convert("RGB")
        ground_truth.append(transform(ground_truth2))

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt,
                    validation_im1,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]
            images.append(image)

        transformed_imgs = []
        for img in images:
            transformed_imgs.append(transform(img))

        score = get_eval_metrics(ground_truth, transformed_imgs, validation_prompt)
        scores.append(score)

        make_image_grid(images, 1, len(images)).save(
            os.path.join(args.output_dir, log_dir, f"images-{step}.png")
        )

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # return evaluation metric scores
    return scores[0]


def main():
    args = config.get_training_args()
    time = datetime.now()
    log_dir = time.strftime("%Y-%m-%d-%H:%M:%S")
    os.makedirs(os.path.join(args.output_dir, log_dir), exist_ok=True)
    fid_scores = []
    clip_scores = []

    # Reproducibility if set
    if args.seed:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)

    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", use_fast=False
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")

    if args.pretrained_controlnet:
        print("Loading exsiting DACN weights")
        dacn = ControlNetModel.from_pretrained(args.pretrained_controlnet)
    else:
        print("Initializing DACN weights from unet")
        dacn = ControlNetModel.from_unet(unet)

    dacn.controlnet_cond_embedding = DACNConditioningEmbedding(320)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    device = get_device()
    dacn.to(device)
    dacn.train()

    params = dacn.parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = ADE20kDataset(root="data/ADEChallengeData2016", transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size
    )

    weight_dtype = torch.float32

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    print("=== RUNNING TRAINING ===")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    if args.seed is not None: print(f"Seed: {args.seed}")

    steps_per_epoch = len(train_dataloader)
    total_steps = args.num_epochs * steps_per_epoch
    global_step = 0

    progress_bar = tqdm(range(0, total_steps), initial=global_step, desc="Steps")
    img_logs = None
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Convert to latent space and concat to make one latent sample
            latents = vae.encode(
                batch["orig"].to(device, dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample random timestep
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            timesteps = timesteps.long()

            # Forward diffusion, add noise at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Conditioning text embeddings
            ids = tokenizer(
                batch["label"],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids = ids.input_ids
            encoder_hidden_states = text_encoder(ids.to(device), return_dict=False)[0]

            # Conditioning image embeddings
            # controlnet_image = batch["overlay"].to(device, dtype=weight_dtype)
            segment = batch["segment"].to(device, dtype=weight_dtype)
            depth = batch["depth"].to(device, dtype=weight_dtype)
            controlnet_image = [segment, depth]

            down_block_res_samples, mid_block_res_sample = dacn(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict noise residual
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=weight_dtype
                ),
                return_dict=False,
            )[0]

            # Loss target
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"ckpt-{global_step}")
                # save model state, double check how to save normally todo

            if (
                args.validation_prompt is not None
                and global_step % args.validation_steps == 0
            ):
                scores = log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    dacn,
                    args,
                    global_step,
                    weight_dtype,
                    log_dir,
                )
                print(scores)
                fid_scores.append(scores['fid'].item())
                clip_scores.append(scores['clip'].item())
                print(f"Averaged evaluation metric scores:\n\tFID: {statistics.mean(fid_scores)}\n\tCLIP: {statistics.mean(clip_scores)}")

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

    dacn.save_pretrained(args.output_dir)

    # Run final validation
    if args.validation_prompt is not None:
        scores = log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            dacn,
            args,
            global_step,
            weight_dtype,
            log_dir,
        )
        fid_scores.append(scores['fid'].item())
        clip_scores.append(scores['clip'].item())
        print(f"Averaged evaluation metric scores:\n\tFID: {statistics.mean(fid_scores)}\n\tCLIP: {statistics.mean(clip_scores)}")



if __name__ == "__main__":
    main()
