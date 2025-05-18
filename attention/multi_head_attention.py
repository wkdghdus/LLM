import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # STEP 1: Linear projections to obtain queries, keys, and values from the input.
        # Each projection outputs [batch, tokens, d_out] and will be split across heads.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final projection after concatenating all heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout layer for attention weights
        self.dropout = nn.Dropout(dropout)

        # STEP 2: Causal mask to ensure each token can only attend to its past and current positions.
        # Shape = [context_length, context_length], upper-triangular matrix.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x: [batch_size, num_tokens, d_in]
        b, num_tokens, _ = x.shape

        # STEP 1 (cont.): Project input to Q, K, V
        Q = self.W_query(x)  # [b, t, d_out]
        K = self.W_key(x)
        V = self.W_value(x)

        # STEP 2: Split each [b, t, d_out] into [b, num_heads, t, head_dim]
        # This allows each head to work on a subset of the representation
        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # STEP 3: Compute raw attention scores for each head
        # Dot product between queries and keys: [b, h, t, t]
        scores = Q @ K.transpose(2, 3)

        # STEP 4: Apply causal mask to prevent attention to future tokens
        # The mask has shape [t, t] and is broadcasted across heads and batches
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        scores.masked_fill_(mask, float("-inf"))

        # STEP 5: Scale attention scores and apply softmax
        # Scaling by sqrt(d_k) stabilizes gradients
        weights = torch.softmax(scores / self.head_dim**0.5, dim=-1)

        # STEP 6: Optionally apply dropout for regularization
        weights = self.dropout(weights)

        # STEP 7: Compute weighted sum of values using attention weights
        # Each head produces a [b, h, t, head_dim] context
        context = weights @ V

        # STEP 8: Concatenate all heads by rearranging dimensions
        # [b, h, t, head_dim] → [b, t, h * head_dim] = [b, t, d_out]
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # STEP 9: Final linear projection to mix head outputs
        return self.out_proj(context)


# ==========================
# Example: Apply the module
# ==========================
torch.manual_seed(123)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)

batch = inputs.unsqueeze(0)  # [1, 6, 3]

d_in, d_out = 3, 4
context_length = batch.shape[1]
num_heads = 2

mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)
context_vecs = mha(batch)

print("Multi-head context vectors:\n", context_vecs)
print("context_vecs.shape:", context_vecs.shape)
