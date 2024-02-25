from typing import Any

from abc import abstractmethod

from agogos._core._block import Block


class Refiner(Block):
    """The refiner block processes the predictions of the model."""

    @abstractmethod
    def predict(self, y: Any) -> Any:
        """Predict the target variable.

        :param y: The predictions to refine.
        :return: The predictions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict method."
        )
