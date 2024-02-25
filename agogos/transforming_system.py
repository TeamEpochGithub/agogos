from typing import Any
from agogos._core._system import System
from agogos.transformer import Transformer


class TransformingSystem(System):
    """A system that transforms the input data."""

    def __post_init__(self) -> None:
        """Post init method for the TransformingSystem class."""

        # Assert all steps are a subclass of Transformer
        for step in self.steps:
            assert issubclass(
                step.__class__, Transformer
            ), f"{step} is not a subclass of Transformer"

        super().__post_init__()

    def transform(self, x: Any) -> Any:
        """Transform the input data.

        :param x: The input data.
        :return: The transformed data.
        """

        # Loop through each step and call the transform method
        for step in self.steps:
            if isinstance(step, Transformer):
                x = step.transform(x)
            else:
                raise TypeError(f"{step} is not a subclass of Transformer")

        return x

    def predict(self, x: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        return self.transform(x)
