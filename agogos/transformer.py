from abc import abstractmethod
from typing import Any

from agogos._core._block import _Block


class Transformer(_Block):
    """The transformer block transforms any data it could be x or y data.

    ### Methods:
    ```python
    @abstractmethod
    def transform(self, data: Any, **kwargs: Any) -> Any: # Transform the input data.

    def get_hash(self) -> str: # Get the hash of the block.
    ```

    ### Usage:
    ```python
    from agogos.transformer import Transformer

    class MyTransformer(Transformer):
        def transform(self, data: Any, **kwargs: Any) -> Any:
            # Transform the input data.
            return data

    my_transformer = MyTransformer()
    transformed_data = my_transformer.transform(data)
    ```
    """

    @abstractmethod
    def transform(self, data: Any, **kwargs: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :param kwargs: Keyword arguments.
        :return: The transformed data."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement transform method."
        )
