import numpy as np

from agogos._core.system import System
from agogos.refiner import Refiner


class RefiningSystem(System):

    steps: list[Refiner]

    def __post_init__(self):
        # Assert all steps are a subclass of Refiner
        for step in self.steps:
            assert issubclass(step, Refiner), f'{step} is not a subclass of Refiner'

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Loop through each step and call the predict method
        for step in self.steps:
            x = step.predict(x)

        return x