from pathlib import Path
import torch
from torch.utils.data import Dataset

class FluxDataset(Dataset):
    def __init__(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        self.latents = torch.load(cache_dir/"latents.pt", weights_only = True)
        embeds = torch.load(cache_dir/"embeddings.pt", weights_only = True)
        self.pooled = embeds["pooled"]
        self.seq = embeds["seq"]

    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, index):
        return self.latents[index].squeeze(0), self.pooled[index], self.seq[index]
    
