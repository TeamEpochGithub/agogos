from abc import abstractmethod
from typing import Any
from agogos._core._system import _System
from agogos.trainer import Trainer
from agogos.training_system import TrainingSystem


class ParallelTrainingSystem(_System):
    """A system that trains the input data in parallel.

    :param steps: The steps to train the input data.
    """

    def __post_init__(self) -> None:
        """Post init method for the ParallelTrainingSystem class."""

        # Assert all steps are a subclass of Trainer or TrainingSystem
        for step in self.steps:
            assert issubclass(step.__class__, Trainer) or issubclass(
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
        for trainer in self.steps[:1]:
            if isinstance(trainer, Trainer) or isinstance(trainer, TrainingSystem):
                x, y = trainer.train(x, y)
            else:
                raise TypeError(
                    f"{trainer} is not a subclass of Trainer or TrainingSystem"
                )

        for trainer in self.steps[1:]:
            if isinstance(trainer, Trainer) or isinstance(trainer, TrainingSystem):
                new_x, new_y = trainer.train(x, y)
                x, y = self.concat(x, new_x), self.concat_labels(y, new_y)
            else:
                raise TypeError(
                    f"{trainer} is not a subclass of Trainer or TrainingSystem"
                )

        return x, y

    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each trainer and call the predict method
        for trainer in self.steps[:1]:
            if isinstance(trainer, Trainer):
                trainer_class_name = trainer.__class__.__name__
                trainer_pred_args = (
                    pred_args[trainer_class_name]
                    if pred_args and trainer_class_name in pred_args
                    else {}
                )
                x = trainer.predict(x, **trainer_pred_args)
            elif isinstance(trainer, TrainingSystem):
                x = trainer.predict(
                    x,
                    pred_args[trainer.__class__.__name__]
                    if pred_args is not None and trainer.__class__.__name__ in pred_args
                    else None,
                )
            else:
                raise TypeError(
                    f"{trainer} is not a subclass of Trainer or TrainingSystem"
                )

        for trainer in self.steps[1:]:
            if isinstance(trainer, Trainer):
                trainer_class_name = trainer.__class__.__name__
                trainer_pred_args = (
                    pred_args[trainer_class_name]
                    if pred_args and trainer_class_name in pred_args
                    else {}
                )
                x_new = trainer.predict(x, **trainer_pred_args)
                x = self.concat(x, x_new)
            elif isinstance(trainer, TrainingSystem):
                trainer_class_name = trainer.__class__.__name__
                trainer_pred_args = (
                    pred_args[trainer_class_name]
                    if pred_args and trainer_class_name in pred_args
                    else {}
                )
                x_new = trainer.predict(x, trainer_pred_args)
                x = self.concat(x, x_new)
            else:
                raise TypeError(
                    f"{trainer} is not a subclass of Trainer or TrainingSystem"
                )

        return x

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
