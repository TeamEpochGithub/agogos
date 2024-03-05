from typing import Any
from agogos._core._system import _System
from agogos.transformer import Transformer


class TransformingSystem(_System):
    """A system that transforms the input data."""

    def __post_init__(self) -> None:
        """Post init method for the TransformingSystem class."""

        # Assert all steps are a subclass of Transformer
        for step in self.steps:
            assert issubclass(step.__class__, Transformer) or issubclass(
                step.__class__, TransformingSystem
            ), f"{step} is not a subclass of Transformer"

        super().__post_init__()

    def transform(self, x: Any, transform_args: dict[str, Any] = {}) -> Any:
        """Transform the input data.

        :param x: The input data.
        :return: The transformed data.
        """

        # Loop through each step and call the transform method
        for step in self.steps:
            step_name = step.__class__.__name__
            step_transform_args = (
                transform_args[step_name]
                if transform_args and step_name in transform_args
                else {}
            )
            if isinstance(step, Transformer):
                x = step.transform(x, **step_transform_args)
            elif isinstance(step, TransformingSystem):
                x = step.transform(x, step_transform_args)
            else:
                raise TypeError(f"{step} is not a subclass of Transformer")

        return x

    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        return self.transform(x, transform_args=pred_args)
