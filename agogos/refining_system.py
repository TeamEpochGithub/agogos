from dataclasses import field
import numpy as np

from agogos._core._system import System
from agogos.refiner import Refiner


class RefiningSystem(System):

    steps: list[Refiner] = field(default_factory=list)

    def __post_init__(self):
        """Post init method for the RefiningSystem class."""
        # Assert all steps are a subclass of Refiner
        for step in self.steps:
            assert issubclass(step.__class__, Refiner), f'{step} is not a subclass of Refiner'

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the system.
        
        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each step and call the predict method
        for step in self.steps:
            x = step.predict(x)

        return x