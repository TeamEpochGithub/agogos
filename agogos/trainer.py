from abc import abstractmethod
import numpy as np

from agogos._core._block import Block


class Trainer(Block):
    """The trainer block is for blocks that need the x and y data."""

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement train method.")

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict method.")
