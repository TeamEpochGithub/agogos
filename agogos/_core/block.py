"""The block module contains the Block class."""
from dataclasses import dataclass
from joblib import hash


@dataclass
class Block:
    """A block in the machine learning pipeline."""

    def __post_init__(self) -> None:
        """Initialize the block."""
        self._set_hash("")

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))

    def get_hash(self) -> str:
        """Get the hash of the block.

        :return: The hash of the block.
        """
        return self._hash
