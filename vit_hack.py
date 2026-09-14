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
        # nn.MultiheadAttention expects (query, key, value, ...)
        # In ViT, query == key == value
        if key is None:
            key = query
        if value is None:
            value = query
        x = query
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each has shape (B, num_heads, N, head_dim)
        
        # Reshape for Julia: (B, N, num_heads, head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, N, num_heads, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, N, num_heads, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, N, num_heads, head_dim)
                
        # Attention computation: Q @ K^T
        # q: (B, N, num_heads, head_dim), k: (B, N, num_heads, head_dim)
        # Result should be: (B, N, num_heads, N)
        attn = julia_matmul(q, k, transpose_b=True) * self.scale
        
        # Apply softmax along the last dimension (attention over keys/sequence)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: attn @ V
        # attn: (B, N, num_heads, N), v: (B, N, num_heads, head_dim)
        # Result should be: (B, N, num_heads, head_dim)
        x = julia_matmul(attn, v, transpose_b=False)
        
        # Reshape back to (B, N, C)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        
        return x, None


def hack_vit(model):
    """julia version replacing regular vit"""
    for block in model.encoder.layers:
        if not isinstance(block, EncoderBlock):
            continue
            
        # Get the embedding dimension from the existing attention layer
        embed_dim = block.self_attention.embed_dim
        num_heads = block.num_heads
        dropout = block.self_attention.dropout
        
        # Create custom attention with same parameters
        custom_attn = CustomAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Copy the weights from the original attention layer
        with torch.no_grad():
            # Copy QKV projection weights
            custom_attn.qkv.weight.copy_(block.self_attention.in_proj_weight)
            custom_attn.qkv.bias.copy_(block.self_attention.in_proj_bias)
            
            # Copy output projection weights
            custom_attn.proj.weight.copy_(block.self_attention.out_proj.weight)
            custom_attn.proj.bias.copy_(block.self_attention.out_proj.bias)
        
        # Replace the attention module
        block.self_attention = custom_attn
        
    return model
