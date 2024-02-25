from typing import Any
from agogos._core._system import System
from agogos.trainer import Trainer


class TrainingSystem(System):
    """A system that trains on the input data and labels."""

    def __post_init__(self) -> None:
        """Post init method for the TrainingSystem class."""

        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            assert issubclass(
                step.__class__, Trainer
            ), f"{step} is not a subclass of Trainer"

        super().__post_init__()

    def predict(self, x: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each step and call the predict method
        for step in self.steps:
            if isinstance(step, Trainer):
                x = step.predict(x)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x

    def train(self, x: Any, y: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system."""

        # Loop through each step and call the train method
        for step in self.steps:
            if isinstance(step, Trainer):
                x, y = step.train(x, y)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x, y
