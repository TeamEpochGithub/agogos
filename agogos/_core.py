"""This module contains the core classes for all classes in the agogos package."""
from dataclasses import field, dataclass
from joblib import hash
from abc import abstractmethod
from typing import Any


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
        self._set_parent(None)
        self._set_children([])

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

    def get_parent(self) -> Any:
        """Get the parent of the block.

        :return: Parent of the block
        """
        return self._parent

    def get_children(self) -> list[Any]:
        """Get the children of the block.

        :return: Children of the block"""
        return self._children

    def _set_parent(self, parent: Any) -> None:
        """Set the parent of the block.

        :param parent: Parent of the block
        """
        self._parent = parent

    def _set_children(self, children: list[Any]) -> None:
        """Set the children of the block.

        :param children: Children of the block
        """
        self._children = children


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

        def get_parent(self) -> Any: # Get the parent of the system.

        def get_children(self) -> list[Any]: # Get the children of the system

        def _set_hash(self, prev_hash: str) -> None: # Set the hash of the system.
    """

    steps: list[_Base] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class"""
        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step._set_parent(self)

        self._set_children(self.steps)

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        for step in self.steps:
            self._hash = hash(self.get_hash() + step.get_hash())
