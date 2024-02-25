from abc import abstractmethod
import numpy as np

from agogos._core._block import Block


class Transformer(Block):
    """The transformer block transforms the data it could be x or y data."""

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement transform method.")