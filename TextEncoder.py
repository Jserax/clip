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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(self.qkv(x), "b l (n h d) -> n b h l d", n=3, h=self.heads)
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask)
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


class TextEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        max_len: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn((max_len, emb_dim)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x), attn_mask)
        out = x + self.ff(self.norm_2(x))
        return out


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        n_head: int,
        emb_dim: int,
        max_len: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.emb = TextEmbedding(vocab_size, emb_dim, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(n_head, emb_dim) for i in range(n_layers)]
        )
        self.norm = RMSNorm(emb_dim)
        self.n_head = n_head

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = torch.unsqueeze(attn_mask, 1).expand(-1, tokens.size(1), -1)
        attn_mask = torch.unsqueeze(attn_mask, 1).expand(-1, self.n_head, -1, -1)
        x = self.emb(tokens)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)

    def params_count(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
