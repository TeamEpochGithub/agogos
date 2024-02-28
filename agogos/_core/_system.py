from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any
from joblib import hash

from agogos._core._block import Block


@dataclass
class System:
    """The system class is the base class for all systems. It is a collection of blocks that can be predict can be called on."""

    steps: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init method for the System class."""
        self._set_hash("")

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict method."
        )

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
