# ====================================
# STEP 1: Input Embedding Preparation
# ====================================
import torch

# Input: 6 tokens, 3-dimensional embeddings
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)

# Define input/output dimensions
d_in, d_out = 3, 2  # Input dim: 3, Output dim: 2 (smaller for illustration)

# ================================================
# STEP 2: Define Projection Matrices (Trainable)
# ================================================
# Each input token will be linearly projected into a query, key, and value space
# These matrices would be trained during backpropagation in a real model

torch.manual_seed(123)  # For reproducibility

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


# =====================================================
# STEP 3: Apply Linear Projections to Generate Q/K/V
# =====================================================
# Project entire input matrix to generate matrices of queries, keys, and values

queries = inputs @ W_query  # shape: [6, 2]
keys    = inputs @ W_key    # shape: [6, 2]
values  = inputs @ W_value  # shape: [6, 2]

print("queries:\n", queries)
print("keys:\n", keys)
print("values:\n", values)


# ================================================================
# STEP 4: Compute Scaled Dot-Product Attention Scores
# ================================================================
# Attention score = dot(Q, K^T) / sqrt(d_k)
# Scaling helps prevent exploding gradients when d_k is large

d_k = queries.shape[1]  # dimension of key vectors
attn_scores = queries @ keys.T / d_k**0.5  # shape: [6, 6]

print("Scaled attention scores:\n", attn_scores)


# ===================================================
# STEP 5: Normalize Scores to Attention Weights
# ===================================================
# Apply softmax row-wise to get a probability distribution per query

attn_weights = torch.softmax(attn_scores, dim=-1)

print("Attention weights:\n", attn_weights)
print("Row-wise sums (should be 1):", attn_weights.sum(dim=-1))


# =========================================================
# STEP 6: Compute Context Vectors via Weighted Values
# =========================================================
# Weighted sum of value vectors using attention weights
# Output: one context vector per token

context_vectors = attn_weights @ values  # shape: [6, 2]

print("Context vectors:\n", context_vectors)
