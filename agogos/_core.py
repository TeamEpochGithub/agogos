"""Core classes for all classes in the agogos package."""
import numbers
import os
from abc import abstractmethod, ABC
from collections.abc import Iterable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, Generic, override

from joblib import hash

_ParentT = TypeVar("_ParentT", bound="_Base")
_ChildT = TypeVar("_ChildT", bound="_Base")
_DT = TypeVar("_DT")

@dataclass(slots=True)
class _Base(Generic[_ParentT, _ChildT]):
    """The _Base class is the base class for all classes in the agogos package.

    :param parent: The parent of the class.
    :param children: The children of the class.
    """

    parent: _ParentT | None = None
    children: Iterable[_ChildT] = field(default_factory=list)
    _hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Set the hash."""
        self.hash = ""

    @property
    def hash(self) -> str:
        """Return the hash of the class.

        :return: The hash of the class.
        """
        return self._hash

    @hash.setter
    def hash(self, prev_hash: str) -> None:
        """Set the hash of the class based on the previous hash.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))

    def save_to_html(self, file_path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        """Write html representation of class to file.

        :param file_path: File path to write to.
        """
        html = self._repr_html_()
        with open(file_path, "w") as file:
            file.write(html)

    def _repr_html_(self) -> str:
        """Return representation of class in html format.

        :return: String representation of html
        """
        html = "<div style='border: 1px solid black; padding: 10px;'>"
        html += f"<p><strong>Class:</strong> {self.__class__.__name__}</p>"
        html += "<ul>"
        html += f"<li><strong>Hash:</strong> {self.hash}</li>"
        html += f"<li><strong>Parent:</strong> {self.parent}</li>"
        html += "<li><strong>Children:</strong> "
        if self.children:
            html += "<ul>"
            for child in self.children:
                html += f"<li>{child._repr_html_()}</li>"
            html += "</ul>"
        else:
            html += "None"
        html += "</li>"
        html += "</ul>"
        html += "</div>"
        return html


class _Block(_Base):
    """The _Block class is the base class for all blocks.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """


@dataclass(slots=True)
class _ParallelSystem(ABC, Generic[_DT], _Base):
    """The _System class is the base class for all systems.

    :param steps: The steps in the system.
    :param weights: Weights of steps in the system, if not specified they are equal.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, original_data: Any), data_to_concat: Any, weight: float = 1.0) -> Any:
            # Specifies how to concat data after parallel computations
    """

    steps: Sequence[_ChildT] = field(default_factory=list)
    weights: Sequence[numbers.Real] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class."""
        # Sort the steps by name, to ensure consistent ordering of parallel computations
        self.steps = sorted(self.steps, key=lambda x: x.__class__.__name__)

        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step.parent = self
        self.children = self.steps

        # Set weights if they don't exist or normalize them
        if not self.weights:
            self.weights = [1 / len(self.steps)] * len(self.steps)
        elif len(self.weights) == len(self.steps):
            self.weights = [w / sum(self.weights) for w in self.weights]
        else:
            raise ValueError("Mismatch between weights and steps")

    @override
    @super.hash.setter
    def hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        # System has no steps and as such hash should not be affected
        if len(self.steps) == 0:
            return

        # System is one step and should act as such
        if len(self.steps) == 1:
            step = self.steps[0]
            step.hash = prev_hash
            self._hash = step.hash
            return

        # System has at least two steps so hash should become a combination
        total = self._hash
        for step in self.steps:
            step.hash = prev_hash
            total = total + step.hash

        self._hash = hash(total)

    @abstractmethod
    def concat(self, original_data: _DT, data_to_concat: _DT, weight: numbers.Real = 1.0) -> _DT:
        """Concatenate the transformed data.

        :param original_data: The first input data.
        :param data_to_concat: The second input data.
        :param weight: Weight of data to concat.
        :return: The concatenated data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement concat method.")


@dataclass
class _SequentialSystem(_Base):
    """The _System class is the base class for all systems.

    :param steps: The steps in the system.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    steps: list[_Base] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class."""
        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step.parent = self

        self.children = self.steps

    @override
    @super.hash.setter
    def hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        # Set hash of each step using previous hash and then update hash with last step
        for step in self.steps:
            step.hash = super().hash
            self._hash = step.hash
