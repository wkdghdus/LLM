import tiktoken

# Instantiate the byte pair encoding (BPE) tokenizer used in GPT-2.
# This tokenizer breaks text into subword units and assigns each a token ID.
# The 'gpt2' encoder includes a predefined vocabulary of 50,257 tokens.
tokenizer = tiktoken.get_encoding("gpt2")

# Sample text to be tokenized. The <|endoftext|> is a special token used by GPT models
# to indicate the end of a document or separate different text segments.
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

# Encode the text into a list of token IDs using the BPE tokenizer.
# 'allowed_special' ensures that special tokens like <|endoftext|> are preserved as-is.
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# Print the resulting list of token IDs.
# Each ID corresponds to a subword or character from the input text.
print(integers)

# Decode the list of token IDs back into a human-readable string.
# This step verifies that encoding and decoding are consistent.
strings = tokenizer.decode(integers)

# Print the reconstructed string, which should match the original input text
# (except for formatting of special tokens and handling of unknown words via subword splits).
print(strings)
