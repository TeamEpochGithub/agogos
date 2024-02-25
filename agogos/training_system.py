from dataclasses import dataclass
import numpy as np
from agogos._core._system import System
from agogos.trainer import Trainer

class TrainingSystem(System):

    steps: list[Trainer]

    def __post_init__(self):
        """Post init method for the TrainingSystem class."""

        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            assert issubclass(step.__class__, Trainer), f'{step} is not a subclass of Trainer'

        super().__post_init__()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the system.
        
        :param x: The input to the system.
        :return: The output of the system.
        """
        
        # Loop through each step and call the predict method
        for step in self.steps:
            x = step.predict(x)

        return x
    
    def train(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Train the system.
        
        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system."""

        # Loop through each step and call the train method
        for step in self.steps:
            x, y = step.train(x, y)
        
        return x, y