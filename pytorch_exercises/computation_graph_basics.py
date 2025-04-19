import torch
import torch.nn.functional as F

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0])  # bias unit

z = x1 * w1 + b          # net input
a = torch.sigmoid(z)     # activation & output

loss = F.binary_cross_entropy(a, y)

print(loss)

from torch.autograd import grad
#grad (Gradient) = how much a small change in a parameter chengs the final output (in this case loss)

grad_L_w1 = grad(loss, w1, retain_graph=True)       # manually grabs gradient of w1 (weight parameter)
grad_L_b = grad(loss, b, retain_graph=True)         # manually grabs gradient of b (bias unit)

print(grad_L_w1)
print(grad_L_b)

loss.backward()     # Automatically computes the gradients of all leaf nodes

print(w1.grad)
print(b.grad)