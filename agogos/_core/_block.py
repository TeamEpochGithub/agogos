"""The block module contains the Block class."""
from joblib import hash

from agogos._core._base import _Base


class _Block(_Base):
    """The _Block class is the base class for all blocks.

    ### Methods:
    ```python
    def get_hash(self) -> str: # Get the hash of the block.

    def _set_hash(self, prev_hash: str) -> None: # Set the hash of the block.
    ```

    ### Usage:
    ```python
    from agogos._core._block import _Block

    class Block(_Block):

        def _set_hash(self, prev_hash: str) -> None:
            # Set the hash of the block.
            self._hash = hash(prev_hash + str(self))
    ```
    """

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))
