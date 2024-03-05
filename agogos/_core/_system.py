from abc import abstractmethod
from dataclasses import field, dataclass
from typing import Any
from joblib import hash

from agogos._core._base import _Base


@dataclass
class _System(_Base):
    """The _System class is the base class for all systems.

    ### Parameters:
    - steps (list[_Base]): The steps in the system.

    ### Methods:
    ```python
    @abstractmethod
    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any: # Predict the output of the system.

    def get_hash(self) -> str: # Get the hash of the system.

    def _set_hash(self, prev_hash: str) -> None: # Set the hash of the system.
    ```

    ### Usage:
    ```python
    from agogos._core._system import _System

    class System(_System):
        def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any:
            # Predict the output of the system.
            return x
    ```
    """

    steps: list[_Base] = field(default_factory=list)

    @abstractmethod
    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :param pred_args: The arguments to pass to the predict method.
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
