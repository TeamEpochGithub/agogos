from typing import Any
from agogos._core._system import _System
from agogos.trainer import Trainer


class TrainingSystem(_System):
    """A system that trains on the input data and labels."""

    def __post_init__(self) -> None:
        """Post init method for the TrainingSystem class."""

        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            assert issubclass(step.__class__, Trainer) or issubclass(
                step.__class__, TrainingSystem
            ), f"{step} is not a subclass of Trainer"

        super().__post_init__()

    def predict(self, x: Any, pred_args: dict[str, Any] | None = None) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each step and call the predict method
        for step in self.steps:
            step_name = step.__class__.__name__
            step_pred_args = (
                pred_args[step_name] if pred_args and step_name in pred_args else {}
            )
            if isinstance(step, Trainer):
                x = step.predict(x, **step_pred_args)
            elif isinstance(step, TrainingSystem):
                x = step.predict(x, step_pred_args)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x

    def train(self, x: Any, y: Any, train_args: dict[str, Any] = {}) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system."""

        # Loop through each step and call the train method
        for step in self.steps:
            step_name = step.__class__.__name__
            step_train_args = (
                train_args[step_name] if train_args and step_name in train_args else {}
            )
            if isinstance(step, Trainer):
                x, y = step.train(x, y, **step_train_args)
            elif isinstance(step, TrainingSystem):
                x, y = step.train(x, y, step_train_args)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x, y
