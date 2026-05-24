from __future__ import annotations
import argparse
import logging
import os
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import bitsandbytes as bnb 
from src.data.dataset import FluxDataset
from src.data.caching import build_cache
from src.lora.lora_linear import inject_lora
from src.utils import load_config, seed_everything, dtype_from_str, lr_lambda, sample_timesteps
from src.utils import add_noise, save_lora_checkpoint, pack_latents, unpack_latents, prepare_img_ids

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("train")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type = str, default = "config/my_config.yaml")
    return p.parse_args()

def build_transformer(cfg, device):
    model_id = cfg["model"]["name"]
    dtype = dtype_from_str(cfg["model"]["dtype"])
    kwargs = {"subfolder": "transformer", "torch_dtype": dtype}
    if cfg["model"].get("quantize_base", False):
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        log.info("Loading Transformer Base in 8-bit")
    else:
        log.info(f"Loding Transformer Base in {dtype}")

    transformer = FluxTransformer2DModel.from_pretrained(model_id, **kwargs)
    for p in transformer.parameters():
        p.requires_grad = False

    inject_lora(transformer, r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
                target_modules=cfg["lora"].get("target_modules"))
    if cfg["train"].get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()

    if not cfg["model"].get("quantize_base", False):
        transformer.to(device)

    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_total  = sum(p.numel() for p in transformer.parameters())
    log.info(f"LoRA params: {n_trainable:,} / {n_total:,} ({100 * n_trainable/ max(1, n_total):.4f}%)")
    print(f"Trainable Parameters:{n_trainable}")
    print(f"Total Parameters:{n_total}")
    return transformer

def build_optimizer(transformer, cfg) -> torch.optim.Optimizer:
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found after injection.")
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    if cfg["train"].get("optimizer", "adamw").lower() == "adamw8bit":
        return bnb.optim.AdamW8bit(lora_params, lr = lr, weight_decay=wd)
    return torch.optim.AdamW(lora_params, lr=lr, weight_decay=wd)

@torch.no_grad()
def run_validation_samples(cfg, model, step, pipe_ref):
    out_dir = Path(cfg["save"]["output_dir"])/"samples"/f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building FluxPipeline for validation sampling")
    pipe = FluxPipeline.from_pretrained(
        cfg["model"]["name"], torch_dtype=dtype_from_str(cfg["model"]["dtype"])
    )
    del pipe.transformer
    torch.cuda.empty_cache()

    src = model.module if hasattr(model, "module") else model
    pipe.transformer = src
    pipe.to('cuda')
    pipe.transformer.eval()

    base_seed = int(cfg["sample"]["seed"])
    walk = bool(cfg["sample"].get("walk_seed", False))
    try:
        for i, prompt in enumerate(cfg["sample"]["prompts"]):
            seed = base_seed + (step if walk else 0) + i
            gen = torch.Generator(device="cuda").manual_seed(seed)
            image = pipe(
                prompt = prompt,
                width = cfg["sample"]["width"],
                height = cfg["sample"]["height"],
                num_inference_steps = cfg["sample"]["steps"],
                guidance_scale = cfg["sample"]["guidance_scale"],
                generator = gen).images[0]
            safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:60]
            image.save(out_dir/f"{i:02d}_{safe_name}.png")
        log.info(f"Wrote {len(cfg['sample']['prompts'])} samples: {out_dir}")
    finally:
        pipe.transformer = None
        pipe.to('cpu')
        del pipe
        torch.cuda.empty_cache()
        src.train()

def train(cfg):
    seed_everything(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("CUDA NOT AVAILABLE")
    log.info("Building Data Cache")
    build_cache(
        images_dir=cfg["data"]["images_dir"],
        captions_dir=cfg["data"]["captions_dir"],
        cache_dir=cfg["data"]["cache_dir"],
        model_id = cfg["model"]["name"],
        device = str(device)
    )
    dataset = FluxDataset(cfg["data"]["cache_dir"])
    log.info(f"Dataset: {len(dataset)} samples")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle = True, num_workers=0,
        pin_memory=True, drop_last=True)
    transformer = build_transformer(cfg, device)
    optimizer = build_optimizer(transformer, cfg)
    warmup = int(cfg["train"]["warmup_steps"])
    steps = int(cfg["train"]["steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, warmup, steps)
    )
    ema_cfg = cfg["train"].get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_transformer: AveragedModel | None = None
    if ema_enabled:
        ema_transformer = AveragedModel(
            transformer, multi_avg_fn = get_ema_multi_avg_fn(float(ema_cfg.get("decay", 0.99)))
        )
        log.info(f"EMA enabled decay = {ema_cfg.get('decay', 0.99)}")
    accum = int(cfg["train"]["gradient_accumulation_steps"])
    grad_clip = int(cfg["train"]["grad_clip_norm"])
    cap_dropout = float(cfg["data"]["caption_dropout"])
    guidance_val = float(cfg["train"]["noise"]["guidance_scale"])
    noise_offset = float(cfg["train"]["noise"].get("offset", 0.0))
    save_every = int(cfg["save"]["save_every"])
    sample_every = int(cfg["sample"].get("sample_every", 0))
    save_dtype = dtype_from_str(cfg["save"]["dtype"])
    output_dir = Path(cfg["save"]["output_dir"])
    max_keep = int(cfg["save"]["max_checkpoints"])
    pipe_ref = {}

    transformer.train()
    data_iter = iter(loader)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(1, steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            latents, pooled, seq_embeds = batch
            latents = latents.to(device, dtype = torch.bfloat16, non_blocking = True)
            pooled = pooled.to(device, dtype = torch.bfloat16, non_blocking = True)
            seq_embeds = seq_embeds.to(device, dtype = torch.bfloat16, non_blocking = True)

            if random.random() < 0.5:
                latents = torch.flip(latents, dims=[-1])

            if random.random() < cap_dropout:
                pooled = torch.zeros_like(pooled)
                seq_embeds = torch.zeros_like(seq_embeds)

            B, C, H, W = latents.shape
            noise = torch.randn_like(latents)
            t, t_norm = sample_timesteps(B, device)
            x_t = add_noise(latents, noise, t_norm.to(latents.dtype), noise_offset)

            x_t_packed = pack_latents(x_t)
            img_ids = prepare_img_ids(H, W, device, x_t_packed.dtype).squeeze(0)
            txt_ids = torch.zeros(seq_embeds.shape[1], 3, device=device, dtype=x_t_packed.dtype)
            guidance = torch.full((B,), guidance_val, device=device, dtype=torch.bfloat16)

            v_pred_packed = transformer(
                hidden_states=x_t_packed,
                timestep=t_norm,
                pooled_projections=pooled,
                encoder_hidden_states=seq_embeds,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            ).sample

            v_pred = unpack_latents(v_pred_packed, H, W)
            v_target = (noise - latents).detach()
            loss = F.mse_loss(v_pred.float(), v_target.float()) / accum
            loss.backward()
            running_loss += loss.item()

        lora_params = [p for p in transformer.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if ema_transformer is not None:
            ema_transformer.update_parameters(transformer)

        if step % 50 == 0:
            pbar.set_postfix(loss=f"{running_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0

        if step % save_every == 0 or step == steps:
            target = ema_transformer if ema_transformer is not None else transformer
            save_lora_checkpoint(target, step, output_dir, save_dtype, max_keep, log)

        if sample_every > 0 and cfg["sample"].get("enabled", False):
            if step % sample_every == 0 or step == steps:
                target = ema_transformer if ema_transformer is not None else transformer
                run_validation_samples(cfg, target, step, pipe_ref)

    log.info("Training complete.")

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    load_dotenv()
    import transformers
    transformers.logging.set_verbosity_error()
    if not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN not set - FLUX.1-dev download will fail.")

    args = parse_args()
    cfg = load_config(args.config)
    log.info(f"config: {args.config}")
    train(cfg)


if __name__ == "__main__":
    main()