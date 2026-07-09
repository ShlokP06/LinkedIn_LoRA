"""
Modal serverless inference — FLUX.1-dev + custom LoRA checkpoints.

GPU        : L40S (48 GB VRAM)
Quant      : 8-bit transformer + 8-bit T5  →  ~24 GB peak, no CPU offloading
Model cost : $0/month  (baked into container image, not a Volume)
LoRA cost  : ~$0.01/month  (tiny Volume, ~100 MB)

Setup (run once)
-----------------
1. Modal dashboard → Secrets → create "huggingface-secret" with key HF_TOKEN
2. modal deploy deploy/modal_inference.py       # builds image, bakes FLUX (~10-15 min first time)
3. python deploy/upload_loras.py                # push your checkpoints to the LoRA Volume

Compare LoRA checkpoints (CLI)
------------------------------
    modal run deploy/modal_inference.py --lora-step 1500 --output s1500.png
    modal run deploy/modal_inference.py --lora-step 3000 --output s3000.png

Web endpoints (live after modal deploy)
----------------------------------------
    POST <app-url>/generate-api?prompt=...&lora_step=1500&size=1024  →  image/png
    GET  <app-url>/loras-api                                          →  {"steps":[1500,...]}
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import modal

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_DIR = "/model/flux-dev"
LORA_DIR = "/loras"
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "proj_mlp", "proj_out"]

app = modal.App("flux-lora-inference")
hf_secret = modal.Secret.from_name("huggingface-secret")

# Only the LoRA weights live in a Volume — they're tiny (~10-20 MB per checkpoint).
# The 35 GB base model is baked into the image layer instead, costing $0/month.
lora_volume = modal.Volume.from_name("flux-lora-weights", create_if_missing=True)


def _download_flux():
    """Called by Modal during image build to bake FLUX.1-dev into the image."""
    import os
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} → {MODEL_DIR} …")
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.msgpack", "flax_model*", "*.ot"],
    )
    print("Download complete.")


# run_function comes BEFORE add_local_python_source so that changes to src/
# don't invalidate the expensive cached model download step.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0",
        "diffusers>=0.29.0",
        "transformers>=5.0.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.3",
        "bitsandbytes>=0.43.0",
        "huggingface_hub[hf_transfer]",
        "Pillow",
        "fastapi[standard]",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TRANSFORMERS_VERBOSITY": "error",
    })
    .run_function(_download_flux, secrets=[hf_secret], timeout=3600)
    .add_local_python_source("src")
)


@app.cls(
    gpu="L40S",
    image=image,
    volumes={LORA_DIR: lora_volume},
    timeout=600,
    secrets=[hf_secret],
    scaledown_window=60,  # stay warm 1 min after last request
    # Snapshot the container (incl. GPU memory) after the pipeline is loaded so
    # cold starts restore the ready-to-run pipeline instead of re-loading and
    # re-quantizing FLUX + T5 from scratch. GPU snapshot is required here because
    # the weights are 8-bit quantized on-device (can't be done on CPU-only snaps).
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class FluxLoRAInference:
    """
    Loads FLUX.1-dev once per container. Hot-swaps LoRA weights per request (<1 s).

    Memory layout on L40S (48 GB):
        transformer  8-bit  ~12 GB
        T5-XXL       8-bit  ~11 GB
        CLIP         bf16   ~0.6 GB
        VAE          bf16   ~0.5 GB
        ─────────────────────────────
        total               ~24 GB   (24 GB headroom for activations)
    """

    @modal.enter(snap=True)
    def load_pipeline(self) -> None:
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig, T5EncoderModel

        from src.lora.lora_linear import inject_lora

        bnb8 = BitsAndBytesConfig(load_in_8bit=True)

        print("Loading transformer (8-bit) …")
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_DIR,
            subfolder="transformer",
            quantization_config=bnb8,
            torch_dtype=torch.bfloat16,
        )

        print("Loading T5 text encoder (8-bit) …")
        t5 = T5EncoderModel.from_pretrained(
            MODEL_DIR,
            subfolder="text_encoder_2",
            quantization_config=bnb8,
            torch_dtype=torch.bfloat16,
        )

        print("Injecting LoRA structure (zero weights) …")
        inject_lora(
            transformer,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
        )

        print("Loading CLIP + VAE …")
        pipe = FluxPipeline.from_pretrained(
            MODEL_DIR,
            transformer=transformer,
            text_encoder_2=t5,
            torch_dtype=torch.bfloat16,
        )
        # CLIP and VAE are regular bf16 modules — move them to GPU explicitly.
        # (transformer and T5 are already on GPU via BitsAndBytes.)
        pipe.text_encoder = pipe.text_encoder.to("cuda")
        pipe.vae = pipe.vae.to("cuda")

        self.pipe = pipe
        self.transformer = transformer
        self.current_lora_step: int | None = None
        print("Pipeline ready.")

    def _available_steps(self) -> list[int]:
        steps = [
            int(re.search(r"(\d+)", f.stem).group())
            for f in Path(LORA_DIR).glob("lora_step_*.safetensors")
        ]
        return sorted(steps)

    def _resolve_step(self, lora_step: int | None) -> int:
        if lora_step is not None:
            return lora_step
        steps = self._available_steps()
        if not steps:
            raise RuntimeError("No LoRA checkpoints found. Run upload_loras.py first.")
        return steps[0]

    def _load_lora(self, step: int) -> None:
        if step == self.current_lora_step:
            return
        import torch
        from safetensors.torch import load_file

        path = Path(LORA_DIR) / f"lora_step_{step:06d}.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint for step {step}. Available: {self._available_steps()}"
            )
        # LoRA weights saved as float16; LoRALayer parameters are float32 — cast before loading.
        sd = {k: v.to(torch.float32) for k, v in load_file(str(path)).items()}
        self.transformer.load_state_dict(sd, strict=False)
        self.current_lora_step = step
        print(f"LoRA step {step} loaded.")

    def _run(
        self,
        prompt: str,
        lora_step: int | None,
        size: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> bytes:
        import torch

        step = self._resolve_step(lora_step)
        self._load_lora(step)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            width=size,
            height=size,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")  # lossless, full resolution
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        prompt: str,
        lora_step: int | None = None,
        size: int = 768,
        num_steps: int = 40,
        guidance_scale: float = 4.0,
        seed: int = 42,
    ) -> bytes:
        """Generate an image, return raw PNG bytes.

        lora_step: checkpoint step (e.g. 1500, 1750 …). None = latest available.
        size: square output resolution in pixels (e.g. 768, 1024, 2048).
        """
        return self._run(prompt, lora_step, size, num_steps, guidance_scale, seed)

    @modal.method()
    def list_loras(self) -> list[int]:
        """Return sorted list of uploaded LoRA checkpoint steps."""
        return self._available_steps()

    @modal.fastapi_endpoint(method="POST")
    def generate_api(
        self,
        prompt: str,
        lora_step: int | None = None,
        size: int = 768,
        num_steps: int = 40,
        guidance_scale: float = 4.0,
        seed: int = 42,
    ) -> object:
        """POST /generate-api — returns image/png directly.

        All params are query strings:
            POST /generate-api?prompt=LinkedIn+Profile+of+Shl0k...&lora_step=1500&size=1024
        """
        from fastapi.responses import Response

        size = (size // 8) * 8  # FLUX requires multiples of 8
        img = self._run(prompt, lora_step, size, num_steps, guidance_scale, seed)
        return Response(content=img, media_type="image/png")

    @modal.fastapi_endpoint(method="GET")
    def loras_api(self) -> dict:
        """GET /loras-api — returns available checkpoint steps as JSON."""
        return {"steps": self._available_steps()}


@app.local_entrypoint()
def main(
    prompt: str = (
        "LinkedIn Profile picture of Shl0k wearing a charcoal grey suit "
        "and white dress shirt, looking directly at the camera with a confident "
        "expression, soft studio lighting, plain white background"
    ),
    lora_step: int = None,
    size: int = 768,
    num_steps: int = 40,
    guidance_scale: float = 4.0,
    seed: int = 42,
    output: str = "output.png",
) -> None:
    """Run inference remotely and save the PNG locally.

    Tip — fair LoRA comparison: use the same --prompt, --seed, and --size across steps.
        modal run deploy/modal_inference.py --lora-step 1500 --output s1500.png
        modal run deploy/modal_inference.py --lora-step 2000 --output s2000.png
        modal run deploy/modal_inference.py --lora-step 3000 --output s3000.png
    """
    model = FluxLoRAInference()

    available = model.list_loras.remote()
    if not available:
        print("No LoRA checkpoints found. Run  python deploy/upload_loras.py  first.")
        return
    print(f"Available LoRA steps: {available}")

    effective_step = lora_step if lora_step is not None else available[-1]
    print(f"Generating  step={effective_step}  size={size}px  steps={num_steps} …")

    img = model.generate.remote(prompt, lora_step, size, num_steps, guidance_scale, seed)
    Path(output).write_bytes(img)
    print(f"Saved {len(img) // 1024} KB PNG → {Path(output).resolve()}")
