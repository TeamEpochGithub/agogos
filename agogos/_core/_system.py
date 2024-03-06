from dataclasses import field, dataclass
from joblib import hash

from agogos._core._base import _Base


@dataclass
class _System(_Base):
    """The _System class is the base class for all systems.

    ### Parameters:
    - steps (list[_Base]): The steps in the system.

    ### Methods:
    ```python
    def get_hash(self) -> str: # Get the hash of the system.

    def _set_hash(self, prev_hash: str) -> None: # Set the hash of the system.
    ```
    """

    steps: list[_Base] = field(default_factory=list)

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        for step in self.steps:
            self._hash = hash(self.get_hash() + step.get_hash())
