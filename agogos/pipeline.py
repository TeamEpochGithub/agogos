from dataclasses import dataclass

import numpy as np
from agogos.refining_system import RefiningSystem
from agogos.training_system import TrainingSystem

from agogos.transforming_system import TransformingSystem


@dataclass
class Pipeline:
    """A pipeline of systems that can be trained and predicted.
    
    :param x_system: The system to transform the input data.
    :param y_system: The system to transform the labelled data.
    :param training_system: The system to train the data.
    :param refining_system: The system to refine the output data.
    """

    x_system: TransformingSystem | None = None
    y_system: TransformingSystem | None = None
    training_system: TrainingSystem | None = None
    refining_system: RefiningSystem | None = None

    def __post_init__(self):
        """Post init method for the Pipeline class."""
        self._set_hash('')

    def train(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Train the system.
        
        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        if self.x_system is not None:
            x = self.x_system.transform(x)
        if self.y_system is not None:
            y = self.y_system.transform(y)
        if self.training_system is not None:
            x, y = self.training_system.train(x, y)
        if self.refining_system is not None:
            y = self.refining_system.predict(y)
        
        return x, y
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the system.
        
        :param x: The input to the system.
        :return: The output of the system.
        """
        if self.x_system is not None:
            x = self.x_system.transform(x)
        if self.training_system is not None:
            x = self.training_system.predict(x)
        if self.y_system is not None:
            x = self.y_system.transform(x)
        
        return x

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the pipeline.
        
        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        xy_hash = ''
        if self.x_system is not None:
            xy_hash += self.x_system.get_hash()
        if self.y_system is not None:
            xy_hash += self.y_system.get_hash()
        
        self._hash = hash(self._hash + xy_hash)

        if self.training_system is not None:
            self._hash = hash(self._hash + self.training_system.get_hash())
        if self.refining_system is not None:
            self._hash = hash(self._hash + self.refining_system.get_hash())
        
    def get_hash(self) -> str:
        """Get the hash of the pipeline.
        
        :return: The hash of the pipeline.
        """
        return self._hash
