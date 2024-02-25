import numpy as np
from agogos._core.system import System
from agogos.transformer import Transformer


class TransformerSystem(System):
    """A system that transforms the input data."""
    
    steps: list[Transformer]
    
    def __post_init__(self):
        # Assert all steps are a subclass of Transformer
        for step in self.steps:
            assert issubclass(step, Transformer), f'{step} is not a subclass of Transformer'
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the system.
        
        :param x: The input to the system.
        :return: The output of the system.
        """
        
        # Loop through each step and call the predict method
        for step in self.steps:
            x = step.predict(x)
        
        return x