import urllib.request

#get file from the textbook repository
url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)    # tokenizing the raw text
preprocessed = [item.strip() for item in preprocessed if item.strip()]


### Now converting tokens to token ID
### This creates set of vocabs for LLM to use.

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
