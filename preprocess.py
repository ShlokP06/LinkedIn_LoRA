from pathlib import Path
import numpy as np
import mediapipe as mp
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

Raw_dir = Path("data/raw")
Img_dir = Path("data/raw/images")
Cap_dir = Path("data/raw/captions")

Trigger = "Shl0k"
Img_type = {'.jpg', '.jpeg', '.png'}
Buckets = [
    (512, 512), (768, 768), (1024, 1024)
]

def pick_bucket(orig_w, orig_h):
    best = None
    best_loss = float("inf")
    for bw, bh in Buckets:
        scale = max(bw / orig_w, bh / orig_h)
        loss = scale * orig_w * orig_h - bw * bh
        if loss < best_loss:
            best_loss = loss
            best = (bw, bh, scale)
    return best

def bucket_resize_crop(image):
    ow, oh = image.size
    bw, bh, scale = pick_bucket(ow, oh)
    rw, rh = round(ow * scale), round(oh * scale)
    resized = image.resize((rw, rh), Image.LANCZOS)

    left = (rw - bw) // 2
    top = (rh - bh) // 2
    return resized.crop((left, top, left + bw, top + bh))

def load_florence(device):
    print("Loading Florence-2....")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", torch_dtype = torch.float16, trust_remote_code = True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code = True)
    return model, processor

@torch.no_grad
def generate_caption(image, model, processor, device):
    task = "<DETAILED_CAPTION>"
    inputs = processor(text=task, images = image, return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()
    ids = model.generate(
        input_ids = inputs["input_ids"],
        pixel_values = inputs["pixel_values"],
        max_new_tokens = 512,
        num_beams = 3,
        do_sample = False
    )
    raw = processor.batch_decode(ids, skip_special_tokens = False)[0]
    parsed = processor.post_process_generation(
        raw, task = task, image_size = (image.width, image.height)
    )
    return parsed[task].strip()

def main():
    images = sorted(p for p in Raw_dir.iterdir() if p.suffix.lower() in Img_type)
    if not images:
        raise SystemExit(f"No images found in {Raw_dir}")
    print(f"Found {len(images)} images")
    Img_dir.mkdir(parents = True, exist_ok=True)
    Cap_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, processor = load_florence(device)
    bucket_counts = []
    fails = 0
    for idx, src in enumerate(tqdm(images, desc = "Processing"), start = 1):
        try:
            img = ImageOps.exif_transpose(Image.open(src).convert("RGB"))
            out = bucket_resize_crop(img)
            key = f"{out.width}x{out.height}"
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
            out.save(Img_dir / f"{idx}.jpg", "JPEG", quality=95)
            caption = generate_caption(out, model, processor, device)
            (Cap_dir / f"{idx}.txt").write_text(caption, encoding = "utf-8")
        except Exception as exc:
            print(f"Failed on {src.name}: {exc}")
            fails += 1

    total = len(images)
    print(f"\nDone. {total - fails}/{total} processed")
    print("Bucket Distribution:")
    for k, v in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"{k} : {v}")

if __name__ == "__main__":
    main()

