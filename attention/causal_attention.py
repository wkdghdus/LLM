import torch
import torch.nn as nn

# ================================================================
# CausalAttention – Self-Attention With Masking Logic
# ================================================================
# This module defines a self-attention block that enforces causality:
# Each token can only attend to itself and tokens before it, never future tokens.
# This is crucial for autoregressive language models like GPT that generate one token at a time.

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        # These linear layers project the input embeddings into separate query, key, and value spaces.
        # They are learned projections and can optionally include biases.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout is applied to the attention weights as a form of regularization
        self.dropout = nn.Dropout(dropout)

        # Mask is a fixed upper triangular matrix (1s above the diagonal), shape = [T, T]
        # This ensures that position t can only attend to positions ≤ t.
        # It's registered as a buffer so it's stored with the model but not trainable.
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x has shape [batch_size, num_tokens, d_in]
        b, num_tokens, d_in = x.shape

        # Project inputs to queries, keys, and values
        # Resulting shapes: [batch_size, num_tokens, d_out]
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # Compute raw attention scores with batch matrix multiplication
        # Shape: [batch_size, num_tokens, num_tokens]
        # Entry (i, j) represents similarity between token i (query) and token j (key)
        attn_scores = queries @ keys.transpose(1, 2)

        # Apply causal mask:
        # The mask is [T, T], where masked positions are 1 above the diagonal
        # We convert it to boolean and slice it to match actual input length
        # Masked positions are set to -inf so softmax gives them zero weight
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            float('-inf')
        )

        # Scale the scores by sqrt(d_k) to improve training stability
        # This avoids softmax gradients becoming too flat or too sharp when d_k is large
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)

        # Dropout is applied to attention weights (not values or projections)
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors: weighted sum of value vectors using attention weights
        # Shape: [batch_size, num_tokens, d_out]
        context_vec = attn_weights @ values

        return context_vec


# ====================================================
# Example usage: single sequence of 6 tokens as a batch
# ====================================================
# The original 'inputs' tensor was [6, 3]; we add batch dimension to make it [1, 6, 3]
torch.manual_seed(123)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)
batch = inputs.unsqueeze(0)  # shape = [1, 6, 3]

context_length = batch.shape[1]  # should be 6

# Instantiate causal attention layer with 3→2 projection, 0 dropout for deterministic result
ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)

# Apply model to batch
context_vecs = ca(batch)

# Output: 1 context vector per token, each of dimension d_out
print("Context vectors (causal):\n", context_vecs)
print("context_vecs.shape:", context_vecs.shape)
