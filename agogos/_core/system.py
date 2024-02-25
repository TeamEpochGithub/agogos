from abc import abstractmethod
from dataclasses import dataclass
import numpy as np

from agogos._core.block import Block


@dataclass
class System:

    steps: list[Block]

    def __post_init__(self):
        self._set_hash('')

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the system.
        
        :param x: The input to the system.
        :return: The output of the system.
        """
        raise NotImplementedError

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.
        
        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        for step in self.steps:
            self._hash = hash(self._hash + step.get_hash())

    def get_hash(self) -> str:
        """Get the hash of the block.
        
        :return: The hash of the block.
        """
        return self._hash