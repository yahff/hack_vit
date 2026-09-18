import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

from julia_bridge import julia_matmul

class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, sequence_length, embedding_dim = query.shape
        qkv = self.qkv(query).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)

        attn = julia_matmul(q, k, transpose_b=True) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = julia_matmul(attn, v, transpose_b=False)

        x = x.reshape(batch_size, sequence_length, embedding_dim)
        x = self.proj(x)

        return x, None


def hack_vit(model):
    """Replace each Vision Transformer attention module with the Julia-backed version."""
    for block in model.encoder.layers:
        if not isinstance(block, EncoderBlock):
            continue

        embed_dim = block.self_attention.embed_dim
        num_heads = block.num_heads
        dropout = block.self_attention.dropout

        # Create custom attention with same parameters
        custom_attn = CustomAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        with torch.no_grad():
            custom_attn.qkv.weight.copy_(block.self_attention.in_proj_weight)
            custom_attn.qkv.bias.copy_(block.self_attention.in_proj_bias)
            custom_attn.proj.weight.copy_(block.self_attention.out_proj.weight)
            custom_attn.proj.bias.copy_(block.self_attention.out_proj.bias)

        block.self_attention = custom_attn

    return model
