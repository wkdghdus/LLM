import torch
from data_preparation_and_sampling.dataset import create_dataloader_v1

# Define the vocabulary size (from GPT tokenizer) and desired embedding dimension
vocab_size = 50257        # Token count from GPT-2's BPE tokenizer
output_dim = 256          # Dimensionality of embedding vectors

# Create the token embedding layer (learns token ID -> vector mappings)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Load training text ("The Verdict") from file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Define model's context size (i.e., number of tokens per training sample)
max_length = 4

# Create a DataLoader to return tokenized training sequences in batches
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

# Fetch the first batch of input and target sequences
data_iter = iter(dataloader)
inputs, targets = next(data_iter)  # Each has shape [8, 4]

# Print the raw token IDs for visualization
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Convert token IDs into dense vector representations
# Output shape: [batch_size, context_length, embedding_dim] → [8, 4, 256]
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# Define positional embedding layer:
# This assigns a unique vector to each position (0 through max_length - 1)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Generate a position index: tensor([0, 1, 2, 3])
# Each index is mapped to its corresponding positional embedding
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)   # Shape: [4, 256] (one for each position)

# Add token and positional embeddings:
# PyTorch will broadcast [4, 256] positional embeddings across the batch dimension (8)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)  # Final shape: [8, 4, 256]

# Now, input_embeddings can be passed into the transformer model's attention blocks
