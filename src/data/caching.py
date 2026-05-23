import torch 
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKL
from tqdm import tqdm

VAE_SCALE = 0.3611
VAE_SHIFT = 0.1159

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def build_cache(images_dir, captions_dir, cache_dir, model_id, device):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents = True, exist_ok=True)
    latents_path = cache_dir / "latents.pt"
    embeddings_path = cache_dir / "embeddings.pt"
    if latents_path.exists() and embeddings_path.exists():
        print("Cache exists, skipping encoding")
        return 
    
    image_paths = sorted(Path(images_dir).glob("*.jpg"))
    caption_paths = sorted(Path(captions_dir).glob("*.txt"))
    assert len(image_paths) == len(caption_paths), "Image/caption count mismatch"
    captions = [p.read_text(encoding = "utf-8").strip() for p in caption_paths]

    print("Encoding images with VAE....")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval()
    TARGET = (768, 768)
    latents = []
    with torch.no_grad():
        for p in tqdm(image_paths, desc = "VAE Encode"):
            img = Image.open(p).convert("RGB").resize(TARGET, Image.LANCZOS)
            pv = to_tensor(img).unsqueeze(0).to(device, dtype = torch.bfloat16)
            lat = vae.encode(pv).latent_dist.sample()
            lat = (lat - VAE_SHIFT) * VAE_SCALE
            latents.append(lat.cpu())
    
    torch.save(latents, latents_path)
    del vae; torch.cuda.empty_cache()

    print("Encoding captions with CLIP...")
    clip_id = "openai/clip-vit-large-patch14"
    clip = CLIPTextModel.from_pretrained(clip_id, torch_dtype = torch.bfloat16).to(device).eval()
    clip_tokenizer = AutoTokenizer.from_pretrained(clip_id)
    with torch.no_grad():
        tokens = clip_tokenizer(captions, return_tensors = 'pt', padding = True,
                                truncation = True, max_length = 77).to(device)
        pooled = clip(**tokens).pooler_output.cpu()
    del clip; torch.cuda.empty_cache()

    print("Encoding captions with T5...")
    t5_id = "google/t5-v1_1-xxl"
    t5 = T5EncoderModel.from_pretrained(t5_id, torch_dtype = torch.bfloat16).to(device).eval()
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_id, legacy=True)
    with torch.no_grad():
        tokens = t5_tokenizer(captions, return_tensors="pt", padding=True,
                              truncation=True, max_length=256).to(device)
        sequence = t5(**tokens).last_hidden_state.cpu()
    
    del t5; torch.cuda.empty_cache()
    torch.save({"pooled": pooled, "seq": sequence}, embeddings_path)
    print(f"Cache Saved: {cache_dir}")
    

