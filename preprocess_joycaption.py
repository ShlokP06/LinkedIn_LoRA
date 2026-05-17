from pathlib import Path
import numpy as np
import mediapipe as mp
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import LlavaForCausalLM, AutoProcessor

Raw_dir = Path("data/raw")
Img_dir = Path("data/raw/processed/images")
Cap_dir = Path("data/raw/processed/captions")

Trigger = "Shl0k"
Img_type = {'.jpg', '.jpeg', '.png'}
Buckets = [
    (512, 512), (768, 768), (1024, 1024)
]
CAPTION_PROMPT = """Write a long descriptive caption for this image in a formal tone.
                    Focus on the person's facial deatures, hair, skin tone, expression,
                    clothing details, pose, and background environment. """

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

def load_joycaption() -> tuple:
    print("Loading JoyCaption..")
    model = LlavaForCausalLM.from_pretrained(
        "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        dtype = torch.bfloat16,
        device_map = 'auto'
    ).eval()
    processor = AutoProcessor.from_pretrained("fancyfeast/llama-joycaption-alpha-two-hf-llava")
    return model, processor

@torch.no_grad
def generate_caption(image, model, processor, device):
    conversation = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_PROMPT}
            ]
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenization=False, add_generation_prompt = True)
    inputs = processor(images = image, text = prompt, return_tensors="pt").to(model.device, torch.bfloat16)
    ids = model.generate(
        **inputs,
        max_new_tokens = 512,
        do_sample = True,
        temperature = 0.6,
        top_p = 0.9
    )
    generated = ids[0][inputs['input_ids'].shape[1]:]
    return processor.decode(generated, skip_special_tokens = True).strip()

def main():
    images = sorted(p for p in Raw_dir.iterdir() if p.suffix.lower() in Img_type)
    if not images:
        raise SystemExit(f"No images found in {Raw_dir}")
    print(f"Found {len(images)} images")
    Img_dir.mkdir(parents = True, exist_ok=True)
    Cap_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, processor = load_joycaption()
    bucket_counts = {}
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

