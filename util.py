import math

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class CosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        decay_iters: int,
        start_lr: float,
        min_lr: float,
        max_lr: float,
        verbose: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.verbose = verbose
        self.iter = 1

    def step(self) -> None:
        if self.iter < self.warmup_iters:
            lr = (
                self.max_lr - self.start_lr
            ) / self.warmup_iters * self.iter + self.start_lr
        elif self.iter > self.decay_iters:
            lr = self.min_lr
        else:
            decay = (self.iter - self.warmup_iters) / (
                self.decay_iters - self.warmup_iters
            )
            decay = min(decay, 1.0)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.verbose:
            print(self.iter, lr)
        self.iter += 1


class CLIPDataset(Dataset):
    def __init__(
        self,
        caption_path: str,
        img_dir: str,
        image_size: int = 224,
        max_length: int = 64,
    ) -> None:
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize((image_size, image_size)),
            ]
        )
        self.img_dir = img_dir
        tokenizer = AutoTokenizer.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D"
        )
        data = pd.read_csv(caption_path, sep="|")
        self.images = data.image_name.tolist()
        data = tokenizer.batch_encode_plus(
            data.comment.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )
        self.tokens = data["input_ids"]
        self.attn_mask = data["attention_mask"]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> torch.Tensor:
        image = Image.open(f"{self.img_dir}/{self.images[idx]}").convert("RGB")
        image = self.transform(image)
        token = self.tokens[idx]
        attn_mask = self.attn_mask[idx]
        return image, token, attn_mask
