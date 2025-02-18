import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        ## if x > 0, f(x) = x 
        ## else alpha * x
        x = np.array(x)  # Ensure x is a NumPy array
        self.inputs = x
        return Tensor(np.where(x > 0, x, self.alpha * x))

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        inputs_array = np.array(self.inputs)  # Ensure stored inputs are a NumPy array
        grad = np.where(inputs_array > 0, 1, self.alpha)  # Compute gradients
        return [Tensor(grad)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        x = np.array(x)
        self.inputs = x
        return Tensor(1 / (1 + np.exp(-x)))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        f'(x) = f(x) * (1 - f(x))
        """
        sigmoid_output = 1 / (1 + np.exp(-self.inputs))
        grad = sigmoid_output * (1 - sigmoid_output)
        return [Tensor(grad)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        x = np.array(x)
        self.inputs = x

        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        outs = exps / np.sum(exps, axis=-1, keepdims=True)

        self.outputs = outs
        return Tensor(outs)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        y = self.outputs
        bn, n = y.shape
        grad = np.zeros(shape=(bn, n, n), dtype=y.dtype)
        
        # TODO: Implement softmax gradient
        
        for i in range(bn):  # for a sample in batch
            y_i = y[i].reshape(-1, 1)  # convert it to column vector
            jacobian = np.outer(y_i, y_i)  # compute outer product
            np.fill_diagonal(jacobian, y_i.flatten() * (1 - y_i.flatten()))
            grad[i] = jacobian

        return [Tensor(grad)]