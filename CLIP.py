import torch
import torch.nn.functional as F
from ImageEncoder import ImageEncoder
from TextEncoder import TextEncoder
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from util import CLIPDataset, CosineScheduler


class CLIPModel(torch.nn.Module):
    def __init__(
        self,
        image_cfg: dict,
        text_cfg: dict,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_model = ImageEncoder(**image_cfg)
        self.text_model = TextEncoder(**text_cfg)
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        img: torch.Tensor,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_emb = self.image_model(img)[:, 0, :]
        text_emb = self.text_model(tokens, attn_mask)[:, 0, :]
        temp = self.temperature.exp()
        sim = (text_emb @ image_emb.T) * temp
        labels = torch.arange(img.size(0), device=img.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


image_cfg = {
    "n_layers": 6,
    "image_size": 224,
    "n_head": 8,
    "emb_dim": 256,
    "patch_size": 16,
}
text_cfg = {
    "n_layers": 4,
    "vocab_size": 30522,
    "n_head": 8,
    "emb_dim": 256,
    "max_len": 64,
}


caption_path = "/content/results.csv"
img_dir = "/content/dataset"
epochs = 16
warmup_epochs = 4
decay_epochs = 2
batch_size = 256
start_lr = 3e-5
max_lr = 1e-3
min_lr = 1e-7
weight_decay = 0.03
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = CLIPDataset(
    caption_path,
    img_dir,
    image_cfg["image_size"],
    text_cfg["max_len"],
)
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
epoch_iters = len(train_loader)

model = CLIPModel(image_cfg, text_cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scaler = torch.cuda.amp.GradScaler()
scheduler = CosineScheduler(
    optimizer,
    warmup_epochs * epoch_iters,
    (epochs - decay_epochs) * epoch_iters,
    start_lr,
    min_lr,
    max_lr,
)

for epoch in tqdm(range(1, epochs + 1)):
    train_loss = 0.0
    for iters, (image, token, mask) in enumerate(train_loader, start=1):
        image = image.to(device)
        token = token.to(device)
        mask = mask.to(device)
        scheduler.step()
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = model(image, token, mask.to(torch.bool))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        if iters % (epoch_iters // 5) == 0:
            print(
                f"Epochs: {epoch}/{epochs} | Iters: {iters}/{epoch_iters} | Train_loss {train_loss / iters:.4f} | Last loss {loss.item():.4f}"
            )
    torch.save(model.image_model.state_dict(), "image_model.pt")
    torch.save(model.text_model.state_dict(), "text_model.pt")
