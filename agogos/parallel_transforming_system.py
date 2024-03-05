from abc import abstractmethod
from typing import Any
from agogos._core._system import _System
from agogos.transformer import Transformer
from agogos.transforming_system import TransformingSystem


class ParallelTransformingSystem(_System):
    """A system that transforms the input data in parallel.

    ### Parameters:
    - steps (list[Transformer | TransformingSystem | ParallelTransformingSystem]): The steps in the system.

    ### Methods:
    ```python
    @abstractmethod
    def concat(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed data.

    def transform(self, x: Any) -> Any: # Transform the input data.

    def get_hash(self) -> str: # Get the hash of the system.
    ```

    ### Usage:
    ```python
    from agogos.parallel_transforming_system import ParallelTransformingSystem

    transformer_1 = CustomTransformer()
    transformer_2 = CustomTransformer()

    class CustomParallelTransformingSystem(ParallelTransformingSystem):
        def concat(self, data1: Any, data2: Any) -> Any:
            # Concatenate the transformed data.
            return data1 + data2

    transforming_system = CustomParallelTransformingSystem(steps=[transformer_1, transformer_2])

    transformed_data = transforming_system.transform(data)
    ```
    """

    def __post_init__(self) -> None:
        """Post init method for the ParallelTransformingSystem class."""

        # Assert all steps are a subclass of Transformer or TransformingSystem
        for step in self.steps:
            assert issubclass(step.__class__, Transformer) or issubclass(
                step.__class__, TransformingSystem
            ), f"{step} is not a subclass of Transformer or TransformingSystem"

        super().__post_init__()

    def transform(self, data: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data.
        """
        # Loop through each step and call the transform method
        for step in self.steps[:1]:
            if isinstance(step, Transformer) or isinstance(step, TransformingSystem):
                data = step.transform(data)
            else:
                raise TypeError(
                    f"{step} is not a subclass of Transformer or TransformingSystem"
                )

        for step in self.steps[1:]:
            if isinstance(step, Transformer) or isinstance(step, TransformingSystem):
                data = self.concat(data, step.transform(data))
            else:
                raise TypeError(
                    f"{step} is not a subclass of Transformer or TransformingSystem"
                )

        return data

    @abstractmethod
    def concat(self, data1: Any, data2: Any) -> Any:
        """Concatenate the transformed data.

        :param data1: The first input data.
        :param data2: The second input data.
        :return: The concatenated data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement concat method."
        )
