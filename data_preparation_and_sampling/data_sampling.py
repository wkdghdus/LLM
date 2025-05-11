### Sliding window approach to sampling datasets to train GPT style LLM.

from data_preparation_and_sampling.byte_pair_encoding import tokenizer

# Load the entire short story "The Verdict" as raw text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Encode the entire text into token IDs using the BPE tokenizer
enc_text = tokenizer.encode(raw_text)

# Discard the first 50 tokens for more interesting sample context
# (e.g., to skip introductions and focus on narrative-rich portions)
enc_sample = enc_text[50:]

# Define the context size (i.e., how many tokens the LLM can "see")
context_size = 4

# Extract an input sequence of size 4
x = enc_sample[:context_size]

# Extract the corresponding target sequence by shifting x by one token
# The model will try to predict y[i] from x[i]
y = enc_sample[1:context_size+1]

# Display the raw token IDs for both input and target
print(f"x: {x}")
print(f"y:      {y}")

# Visualize input-target token alignment using a sliding window
# This mimics how LLMs learn next-token prediction
for i in range(1, context_size+1):
    context = enc_sample[:i]        # Input context up to i tokens
    desired = enc_sample[i]         # The next token to predict

    # Show the actual token IDs and the corresponding decoded text
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
