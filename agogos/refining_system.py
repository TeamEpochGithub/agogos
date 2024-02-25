from typing import Any

from agogos._core._system import System
from agogos.refiner import Refiner


class RefiningSystem(System):
    """A system that refines the output of the training system."""

    def __post_init__(self) -> None:
        """Post init method for the RefiningSystem class."""

        # Assert all steps are a subclass of Refiner
        for step in self.steps:
            assert issubclass(
                step.__class__, Refiner
            ), f"{step} is not a subclass of Refiner"

        super().__post_init__()

    def predict(self, x: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each step and call the predict method
        for step in self.steps:
            if isinstance(step, Refiner):
                x = step.predict(x)
            else:
                raise TypeError(f"{step} is not a subclass of Refiner")

        return x
