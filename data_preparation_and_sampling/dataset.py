import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

# Custom PyTorch Dataset for generating input-target token ID pairs for LLM training
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []   # Holds all input sequences
        self.target_ids = []  # Holds corresponding target sequences (shifted by one)

        # Encode the raw text into token IDs using the GPT-2 tokenizer
        # Note: <|endoftext|> is treated as a special token
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Ensure that the text has enough tokens to generate at least one full sequence
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to generate overlapping sequences
        # Each window creates one input-target pair
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]                    # Input sequence
            target_chunk = token_ids[i + 1: i + max_length + 1]          # Target sequence (shifted right)
            self.input_ids.append(torch.tensor(input_chunk))            # Convert list to PyTorch tensor
            self.target_ids.append(torch.tensor(target_chunk))          # Both will have shape [max_length]

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Return a single input-target pair by index
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Factory function to create a PyTorch DataLoader from raw text
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize Byte Pair Encoding tokenizer used in GPT-2 and GPT-3
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create a dataset with overlapping input-target token pairs
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap the dataset in a DataLoader to enable batching and parallel loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,     # Number of input-target pairs per batch
        shuffle=shuffle,           # Randomize the order of samples (important for training)
        drop_last=drop_last,       # Drop last batch if it has fewer samples than batch_size
        num_workers=num_workers    # Parallelism for data loading
    )

    return dataloader

### Test section

# Load raw text from file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Test DataLoader with batch size of 1, max_length=4, stride=1
# This demonstrates token-by-token sliding (maximum overlap)
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)     # First input-target pair
print(first_batch)

second_batch = next(data_iter)    # Second pair, shifted by 1
print(second_batch)

# Test DataLoader with batch size of 8, max_length=4, stride=4
# This creates non-overlapping sequences
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)        # Shape: [8, 4] (8 sequences, 4 tokens each)
print("\nTargets:\n", targets)    # Each row is the next-token sequence for the corresponding input row
