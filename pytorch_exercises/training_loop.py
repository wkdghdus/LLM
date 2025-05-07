import torch
import torch.nn.functional as F

from pytorch_exercises.multilayer_neural_network_forward import NeuralNetwork
from pytorch_exercises.datasets_and_dataloaders import X_train, y_train, train_loader

# Set random seed for reproducibility
torch.manual_seed(123)

# Initialize the model with 2 inputs and 2 outputs
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# Set up the optimizer (Stochastic Gradient Descent) with a learning rate of 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

# Training loop: iterate over the dataset multiple times (epochs)
for epoch in range(num_epochs):
    
    model.train()  # Set the model to training mode
    
    # Iterate through batches of the training data
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)  # Forward pass: compute raw outputs (logits)
        loss = F.cross_entropy(logits, labels)  # Compute loss; softmax is internally applied
        optimizer.zero_grad()  # Clear accumulated gradients
        
        # ---
        # Note: After loss.backward(), each model parameter's .grad attribute is populated.
        # Optimizer.step() reads these .grad values to update the model.
        # That's why optimizer.step() does not need loss as input.
        # Example:
        # print(model.linear_layer.weight.grad)  # Before backward: None
        # loss.backward()
        # print(model.linear_layer.weight.grad)  # After backward: tensor of gradients
        # optimizer.step()  # Updates weights using these gradients
        # ---

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()  # Set the model to evaluation mode (important for some layers like dropout/batchnorm)

# After training, use the model to make predictions on the training data
model.eval()

with torch.no_grad():  # Disable gradient tracking for inference
    outputs = model(X_train)

print(outputs)  # Logits output (raw scores)

# Convert logits to class membership probabilities
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

# Predict class labels based on highest probability
predictions = torch.argmax(probas, dim=1)
print(predictions)

# Alternatively, directly use logits for prediction (no softmax needed)
predictions = torch.argmax(outputs, dim=1)
print(predictions)

# Compare predictions with true labels
torch.sum(predictions == y_train)  # Count number of correct predictions

# Function to compute accuracy on any dataloader
def compute_accuracy(model, dataloader):
    model = model.eval()  # Ensure model is in evaluation mode
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)  # Get predicted class
        compare = labels == predictions  # Boolean tensor: correct or not
        correct += torch.sum(compare)  # Sum number of correct predictions
        total_examples += len(compare)  # Total number of samples

    return (correct / total_examples).item()  # Return accuracy as float
