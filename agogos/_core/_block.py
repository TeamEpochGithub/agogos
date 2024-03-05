"""The block module contains the Block class."""
from joblib import hash

from agogos._core._base import _Base


class _Block(_Base):
    """A block in the machine learning pipeline."""

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))
