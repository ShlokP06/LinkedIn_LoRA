from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm.auto import tqdm

Raw_dir = Path("data/raw")
Img_dir = Path("data/processed/images")

Img_type = {'.jpg', '.jpeg', '.png'}
Buckets = [
    # Square
    (512, 512), (768, 768), (1024, 1024), (1280, 1280), (1536, 1536), (2048, 2048),
    # 4:3 and 3:4
    (768, 576), (576, 768),
    (1024, 768), (768, 1024),
    (1280, 960), (960, 1280),
    (1536, 1152), (1152, 1536),
    (2048, 1536), (1536, 2048),
    # 5:4 and 4:5
    (1280, 1024), (1024, 1280),
    (1600, 1280), (1280, 1600),
    # 3:2 and 2:3
    (768, 512), (512, 768),
    (1152, 768), (768, 1152),
    (1536, 1024), (1024, 1536),
    (2048, 1365), (1365, 2048),
    # 16:9 and 9:16
    (1024, 576), (576, 1024),
    (1280, 720), (720, 1280),
    (1536, 864), (864, 1536),
    (2048, 1152), (1152, 2048),
]

def pick_bucket(orig_w: int, orig_h: int) -> tuple:
    best = None
    best_score = float("inf")
    for bw, bh in Buckets:
        scale = max(bw / orig_w, bh / orig_h)
        crop_fraction = (scale**2 * orig_w * orig_h - bw * bh) / (bw * bh)
        score = crop_fraction - (bw * bh) * 1e-12
        if score < best_score:
            best_score = score
            best = (bw, bh, scale)
    return best

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_bust_crop(image: Image.Image, padding_top: float = 0.4, padding_bottom: float = 1.8, padding_side: float = 0.5) -> Image.Image:
    """Crop to face + shoulders. Falls back to original if no face detected."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        w, h = image.size
        crop_h = min(h, int(w * 1.3))
        return image.crop((0, 0, w, crop_h))

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    w, h = image.size

    left   = max(0, x - padding_side   * fw)
    top    = max(0, y - padding_top    * fh)
    right  = min(w, x + fw + padding_side   * fw)
    bottom = min(h, y + fh + padding_bottom * fh)

    return image.crop((int(left), int(top), int(right), int(bottom)))

def bucket_resize_crop(image: Image.Image) -> Image.Image:
    ow, oh = image.size
    bw, bh, scale = pick_bucket(ow, oh)
    rw, rh = round(ow * scale), round(oh * scale)
    resized = image.resize((rw, rh), Image.LANCZOS)

    left = (rw - bw) // 2
    top = (rh - bh) // 2
    return resized.crop((left, top, left + bw, top + bh))

def main():
    images = sorted(p for p in Raw_dir.iterdir() if p.suffix.lower() in Img_type)
    if not images:
        raise SystemExit(f"No images found in {Raw_dir}")
    print(f"Found {len(images)} images")
    Img_dir.mkdir(parents=True, exist_ok=True)

    bucket_counts = {}
    fails = 0
    for idx, src in enumerate(tqdm(images, desc="Processing"), start=1):
        try:
            img = ImageOps.exif_transpose(Image.open(src).convert("RGB"))
            img = face_bust_crop(img)
            out = bucket_resize_crop(img)
            key = f"{out.width}x{out.height}"
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
            out.save(Img_dir / f"{idx}.jpg", "JPEG", quality=95)
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
