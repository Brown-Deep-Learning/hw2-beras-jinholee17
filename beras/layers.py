import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        return self.w, self.b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        self.inputs = x  # this stores the inputs for backpropagation
        return np.matmul(x, self.w) + self.b

    def get_input_gradients(self) -> list[Tensor]:
        return [self.w]
        # return [np.matmul(np.ones_like(self.inputs), self.w.T)]

    def get_weight_gradients(self) -> list[Tensor]:
        # weight_gradient = Tensor(np.matmul(self.inputs.T, np.ones_like(self.b)))
        weight_gradient = np.matmul(np.expand_dims(self.inputs[0], axis=-1), np.ones(np.expand_dims(self.w, axis=1).shape))
        bias_gradient = np.ones_like(self.b)
        return [weight_gradient, bias_gradient]
    
        # grad_output = np.ones_like(self.inputs)
        # weight_gradient = np.matmul(self.inputs.T, grad_output)  
        # bias_gradient = np.sum(grad_output, axis=0, keepdims=True) 
        # return [weight_gradient, bias_gradient]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        if initializer == "zero":
            w = np.zeros((input_size, output_size))
        elif initializer == "normal":
            w = np.random.randn(input_size, output_size) * 0.01
        elif initializer == "xavier":
            limit = np.sqrt(2 / (input_size + output_size))
            w = np.random.normal(0, limit, (input_size, output_size))
        
        elif initializer == "kaiming":
            w = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        b = np.zeros((1, output_size)) # initalize bias to 0
        return Variable(w), Variable(b)
