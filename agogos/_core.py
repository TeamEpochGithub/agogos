"""This module contains the core classes for all classes in the agogos package."""
from dataclasses import field, dataclass
from joblib import hash
from abc import abstractmethod


@dataclass
class _Base:
    """The _Base class is the base class for all classes in the agogos package.

    Methods:
    .. code-block:: python
        @abstractmethod
        def _set_hash(self, prev_hash: str) -> None:
            # Set the hash of the block.
            # Called by the __post_init__ method of the block.

        def get_hash(self) -> str:
            # Get the hash of the block.

    Usage:
    .. code-block:: python
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


class _Block(_Base):
    """The _Block class is the base class for all blocks.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str: # Get the hash of the block.

        def _set_hash(self, prev_hash: str) -> None: # Set the hash of the block.
    """

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))


@dataclass
class _System(_Base):
    """The _System class is the base class for all systems.

    Parameters:
    - steps (list[_Base]): The steps in the system.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str: # Get the hash of the system.

        def _set_hash(self, prev_hash: str) -> None: # Set the hash of the system.
    """

    steps: list[_Base] = field(default_factory=list)

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        for step in self.steps:
            self._hash = hash(self.get_hash() + step.get_hash())
