from abc import abstractmethod
from typing import Any

from agogos._core._block import _Block


class Trainer(_Block):
    """The trainer block is for blocks that need to train on two inputs and predict on one.

    ### Methods:
    ```python
    @abstractmethod
    def train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]: # Train the block.

    @abstractmethod
    def predict(self, x: Any, **kwargs: Any) -> Any: # Predict the target variable.

    def get_hash(self) -> str: # Get the hash of the block.
    ```

    ### Usage:
    ```python
    from agogos.trainer import Trainer

    class MyTrainer(Trainer):

        def train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
            # Train the block.
            return x, y

        def predict(self, x: Any, **kwargs: Any) -> Any:
            # Predict the target variable.
            return x

    my_trainer = MyTrainer()
    predictions, labels = my_trainer.train(x, y)
    predictions = my_trainer.predict(x)
    ```
    """

    @abstractmethod
    def train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement train method."
        )

    @abstractmethod
    def predict(self, x: Any, **kwargs: Any) -> Any:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict method."
        )
