from dataclasses import field
import numpy as np
from agogos._core._system import System
from agogos.transformer import Transformer

class TransformingSystem(System):
    """A system that transforms the input data."""
    
    steps: list[Transformer] = field(default_factory=list)
    
    def __post_init__(self):
        """Post init method for the TransformingSystem class."""
        
        # Assert all steps are a subclass of Transformer
        for step in self.steps:
            assert issubclass(step.__class__, Transformer), f'{step} is not a subclass of Transformer'
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the input data.
        
        :param x: The input data.
        :return: The transformed data.
        """
        
        # Loop through each step and call the transform method
        for step in self.steps:
            x = step.transform(x)

        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the system.
        
        :param x: The input to the system.
        :return: The output of the system.
        """
        return self.transform(x)