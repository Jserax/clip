import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        h_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.w1 = nn.Linear(dim, h_dim)
        self.w2 = nn.Linear(dim, h_dim)
        self.w3 = nn.Linear(h_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    def __init__(
        self,
        n_head: int,
        emb_dim: int,
        bias: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias)
        self.out = nn.Linear(emb_dim, emb_dim, bias)
        self.heads = n_head
        self.head_dim = emb_dim // n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(self.qkv(x), "b l (n h d) -> n b h l d", n=3, h=self.heads)
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out(out)


class RMSNorm(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-6,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weights


class ImageEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        emb_dim: int,
        patch_size: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.token_emb = nn.Conv2d(
            3, emb_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.pos_emb = nn.Parameter(
            torch.randn(((image_size // patch_size) ** 2 + 1, emb_dim))
        )
        self.target_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.token_emb(x)
        x = rearrange(x, "b d x y -> b (x y) d")
        cls_token = self.target_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        return x + self.pos_emb


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_head: int,
        emb_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attention = Attention(n_head, emb_dim)
        self.ff = FeedForward(emb_dim, 4 * emb_dim)
        self.norm_1 = RMSNorm(emb_dim)
        self.norm_2 = RMSNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x))
        out = x + self.ff(self.norm_2(x))
        return out


class ImageEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        image_size: int,
        n_head: int,
        emb_dim: int,
        patch_size: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.emb = ImageEmbedding(image_size, emb_dim, patch_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(n_head, emb_dim) for i in range(n_layers)]
        )
        self.norm = RMSNorm(emb_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def params_count(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
