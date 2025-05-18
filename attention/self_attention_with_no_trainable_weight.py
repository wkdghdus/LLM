
# ======================================================
# Computing attention weight for a single input token. 
# ======================================================

# ====================================
# STEP 1: Input Embedding Preparation
# ====================================
# Define a sequence of tokens represented by 3D embedding vectors (already tokenized & embedded)
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# Select the 2nd input token ("journey") to be the query for attention computation
query = inputs[1]


# ============================================================
# STEP 2: Compute Raw Attention Scores (Dot Product Similarity)
# ============================================================
# Idea: Measure similarity between the query and each input token using dot product

attn_scores_2 = torch.empty(inputs.shape[0])  # Allocate space for 6 similarity scores

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)  # Higher score = more relevant to the query

print(attn_scores_2)  # Unnormalized similarity values


# =======================================
# Subsection: Manual Dot Product Check
# =======================================
# Verify dot product calculation between first token and the query
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)                        # Manually calculated
print(torch.dot(inputs[0], query))  # PyTorch-computed (should match)


# ===========================================================
# STEP 3: Normalize Attention Scores → Compute Attention Weights
# ===========================================================
# Goal: Convert similarity scores into weights that sum to 1
#       so they can be used to compute a weighted sum

# -----------------------------------------------------------
# Subsection A: Manual Normalization (Proportional Scaling)
# -----------------------------------------------------------
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Manual normalized weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())  # Should be 1.0

# -----------------------------------------------------------
# Subsection B: Softmax Normalization (More Stable)
# -----------------------------------------------------------
# Define a naive softmax function
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Naive softmax weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())  # Should be 1.0

# Use PyTorch's numerically stable softmax function (recommended in real models)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("PyTorch softmax weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())  # Should be 1.0


# ====================================================
# STEP 4: Compute Context Vector (Weighted Combination)
# ====================================================
# Idea: Use attention weights to compute a new representation (context vector)
#       that blends all inputs based on their relevance to the query

# Re-declare for clarity
query = inputs[1]

# Initialize context vector (same size as embedding)
context_vec_2 = torch.zeros(query.shape)

# Weighted sum: combine input vectors using attention weights
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# Output the enriched context vector for token 2
print("Context vector z(2):", context_vec_2)





# ======================================================
# Generalized Self-Attention – All Positions
# ======================================================
# Goal: Let each token attend to every other token (including itself).
# Each token becomes a query in turn, producing its own context vector.


# ====================================
# STEP 1: Input Embedding Preparation
# ====================================
# Define a batch of token embeddings (sequence of 6 tokens, each with 3 features)
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # x(1) - Your
   [0.55, 0.87, 0.66],  # x(2) - journey
   [0.57, 0.85, 0.64],  # x(3) - starts
   [0.22, 0.58, 0.33],  # x(4) - with
   [0.77, 0.25, 0.10],  # x(5) - one
   [0.05, 0.80, 0.55]]  # x(6) - step
)


# ============================================================
# STEP 2: Compute Raw Attention Scores (Dot Product Similarity)
# ============================================================
# Goal: Let each token act as a query and compute its similarity to all others
#       using the dot product (self-attention)

# Option A: Manual nested loop (for illustration)
attn_scores = torch.empty(6, 6)  # shape = [num_tokens, num_tokens]
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print("Manual pairwise attention scores:")
print(attn_scores)

# Option B: Efficient matrix multiplication
# inputs @ inputs.T yields the same result
attn_scores = inputs @ inputs.T  # shape = [6, 6]
print("Efficient attention scores (via matmul):")
print(attn_scores)


# ===========================================================
# STEP 3: Normalize Attention Scores → Compute Attention Weights
# ===========================================================
# Goal: Convert each row of scores into a probability distribution (weights)

# -----------------------------------------------------------
# Subsection A: Manual Normalization (for intuition)
# -----------------------------------------------------------
# For each row, divide by the row sum (non-softmax normalization)
attn_weights_manual = attn_scores / attn_scores.sum(dim=-1, keepdim=True)
print("Manual normalized attention weights (row-wise):")
print(attn_weights_manual)
print("Row sums:", attn_weights_manual.sum(dim=-1))  # Should all be 1

# -----------------------------------------------------------
# Subsection B: Softmax Normalization (Standard in Practice)
# -----------------------------------------------------------
# Use softmax to ensure numerical stability and probabilistic interpretation
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Softmax attention weights:")
print(attn_weights)
print("Row sums:", attn_weights.sum(dim=-1))  # Should all be 1


# ====================================================
# STEP 4: Compute Context Vectors (Weighted Combination)
# ====================================================
# Use attention weights to compute a weighted sum of value vectors
# Here, queries, keys, and values are all equal to 'inputs'

# Each row of output = context vector for each input token
context_vectors = attn_weights @ inputs  # shape = [6, 3]
print("Context vectors (one per token):")
print(context_vectors)

# Optional: Compare context vector for x(2) with result from section 3.3.1
print("Context vector z(2):", context_vectors[1])
