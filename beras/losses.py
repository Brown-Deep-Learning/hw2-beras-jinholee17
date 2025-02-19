import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        error = (y_true - y_pred) ** 2
        return np.mean(np.mean(error, axis=-1))

    def get_input_gradients(self) -> list[Tensor]:
        batch_size = self.inputs[0].shape[0]
        grad = -2 * (self.inputs[1] - self.inputs[0]) / batch_size
        return grad

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
