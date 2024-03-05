from abc import abstractmethod
from dataclasses import dataclass
from joblib import hash


@dataclass
class _Base:
    """The _Base class is the base class for all classes in the agogos package.

    ### Methods:
    ```python
    def get_hash(self) -> str: # Get the hash of the block.

    @abstractmethod
    def _set_hash(self, prev_hash: str) -> None: # Set the hash of the block.
        # Called by the __post_init__ method of the block.
    ```

    ### Usage:
    ```python
    from agogos._core._base import _Base

    class Block(_Base):

        def _set_hash(self, prev_hash: str) -> None:
            # Set the hash of the block.
            self._hash = hash(prev_hash + str(self))
    """

    def __post_init__(self) -> None:
        """Initialize the block."""
        self._set_hash("")

    @abstractmethod
    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _set_hash method."
        )

    def get_hash(self) -> str:
        """Get the hash of the block.

        :return: The hash of the block.
        """
        return self._hash
