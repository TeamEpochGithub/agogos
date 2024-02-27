from abc import abstractmethod
from typing import Any
from agogos._core._system import System
from agogos.trainer import Trainer
from agogos.training_system import TrainingSystem



class ParallelTrainingSystem(System):
    """A system that trains the input data in parallel.

    :param trainers: The trainers to train the input data.
    """

    trainers: list[Trainer | TrainingSystem]

    def __post_init__(self) -> None:
        """Post init method for the ParallelTrainingSystem class."""

        # Assert all steps are a subclass of Trainer or TrainingSystem
        for step in self.steps:
            assert issubclass(
                step.__class__, Trainer
            ) or issubclass(
                step.__class__, TrainingSystem
            ), f"{step} is not a subclass of Trainer or TrainingSystem"

        super().__post_init__()

    def train(self, x: Any, y: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """

        # Loop through each step and call the train method
        for trainer in self.trainers:
            if isinstance(trainer, Trainer) or isinstance(trainer, TrainingSystem):
                new_x, new_y = trainer.train(x, y)
                x, y = self.concat(x, new_x), self.concat_labels(y, new_y)
            else:
                raise TypeError(f"{trainer} is not a subclass of Trainer or TrainingSystem")

        return x, y
    
    def predict(self, x: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        
        # Loop through each trainer and call the predict method
        for trainer in self.trainers:
            if isinstance(trainer, Trainer) or isinstance(trainer, TrainingSystem):
                x = self.concat(x, trainer.predict(x))
            else:
                raise TypeError(f"{trainer} is not a subclass of Trainer or TrainingSystem")

    
    def concat_labels(self, data1: Any, data2: Any) -> Any:
        """Concatenate the transformed labels. Will use concat method if not overridden.

        :param data1: The first input data.
        :param data2: The second input data.
        :return: The concatenated data.
        """
        return self.concat(data1, data2)
        
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

    
