from abc import abstractmethod
from dataclasses import field, dataclass
from typing import Any
from joblib import hash

from agogos._core._base import _Base


@dataclass
class _System(_Base):
    """The system class is the base class for all systems. It is a collection of blocks that can be predict can be called on."""

    steps: list[_Base] = field(default_factory=list)

    @abstractmethod
    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any:
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
            self._hash = hash(self.get_hash() + step.get_hash())
