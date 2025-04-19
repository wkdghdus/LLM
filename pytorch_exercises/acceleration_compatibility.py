import torch
print(torch.cuda.is_available()) # -> False
print(torch.backends.mps.is_available()) # -> True