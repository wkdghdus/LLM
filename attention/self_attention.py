# ====================================================
# Class-based Self-Attention (Manual Weights)
# ====================================================
# Encapsulates linear projections and attention logic into a PyTorch module.
# Uses torch.nn.Parameter for explicit weight creation and control.

import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # Manually defined trainable parameters for query, key, and value projections
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # Linear projections
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value

        # Scaled dot-product attention
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Weighted sum to get context vectors
        context_vec = attn_weights @ values
        return context_vec

### Example usage
# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
# print("Output from SelfAttention_v1:")
# print(sa_v1(inputs))


# =====================================================
# Self-Attention using nn.Linear Layers
# =====================================================
# Uses PyTorch’s built-in Linear layers for cleaner, scalable code.
# Optionally supports bias in projection layers.

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # Linear projection layers (better practice than manual weights)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # Project input embeddings into query, key, and value vectors
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # Scaled dot-product attention
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Weighted sum to get contextualized output
        context_vec = attn_weights @ values
        return context_vec

### Example usage
# torch.manual_seed(789)
# sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
# print("Output from SelfAttention_v2:")
# print(sa_v2(inputs))
