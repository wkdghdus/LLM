import torch


class NeuralNetwork(torch.nn.Module):   
    """
    custom neural network often subclasses torch.nn.Module.
    we define the network layers in the __init__ constructor 
    and specify how the layers interact in the forward method.
    """

    def __init__(self, num_inputs, num_outputs):
        """
        __init__ constructor to define the structure of our custom neural network
        """
        super().__init__()

        self.layers = torch.nn.Sequential(
                
            ### 1st hidden layer
            torch.nn.Linear(num_inputs, 30),    # Linear layer takes # of input and # of output nodes as arguments.

            torch.nn.ReLU(),    # ReLU is an activation function to introduce non-linearity
                                # which lets the network learn more complicated patterns 
                                # (like curves, edges, images, or text relationships).

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        """
        The forward method describes how the input data passes through the network 
        and comes together as a computation graph. 
        """

        logits = self.layers(x)
        return logits   #outputs of the last layers are called logits