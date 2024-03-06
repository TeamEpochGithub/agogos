from abc import abstractmethod
from typing import Any

from agogos._core import _Block, _System


class Trainer(_Block):
    """The trainer block is for blocks that need to train on two inputs and predict on one.

    ### Methods:
    ```python
    @abstractmethod
    def train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]: # Train the block.

    @abstractmethod
    def predict(self, x: Any, **kwargs: Any) -> Any: # Predict the target variable.

    def get_hash(self) -> str: # Get the hash of the block.
    ```

    ### Usage:
    ```python
    from agogos.trainer import Trainer

    class MyTrainer(Trainer):

        def train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
            # Train the block.
            return x, y

        def predict(self, x: Any, **kwargs: Any) -> Any:
            # Predict the target variable.
            return x

    my_trainer = MyTrainer()
    predictions, labels = my_trainer.train(x, y)
    predictions = my_trainer.predict(x)
    ```
    """

    @abstractmethod
    def train(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement train method."
        )

    @abstractmethod
    def predict(self, x: Any, **kwargs: Any) -> Any:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict method."
        )


class TrainingSystem(_System):
    """A system that trains on the input data and labels.

    ### Parameters:
    - steps (list[Trainer | TrainingSystem | ParallelTrainingSystem]): The steps in the system.

    ### Methods:
    ```python
    def train(self, x: Any, y: Any, train_args: dict[str, Any] = {}) -> tuple[Any, Any]: # Train the system.

    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any: # Predict the output of the system.

    def get_hash(self) -> str: # Get the hash of the system.
    ```

    ### Usage:
    ```python
    from agogos.training_system import TrainingSystem

    trainer_1 = CustomTrainer()
    trainer_2 = CustomTrainer()

    training_system = TrainingSystem(steps=[trainer_1, trainer_2])
    trained_x, trained_y = training_system.train(x, y)
    predictions = training_system.predict(x)
    ```
    """

    def __post_init__(self) -> None:
        """Post init method for the TrainingSystem class."""

        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            assert issubclass(step.__class__, Trainer) or issubclass(
                step.__class__, TrainingSystem
            ), f"{step} is not a subclass of Trainer"

        super().__post_init__()

    def predict(self, x: Any, pred_args: dict[str, Any] | None = None) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each step and call the predict method
        for step in self.steps:
            step_name = step.__class__.__name__
            step_pred_args = (
                pred_args[step_name] if pred_args and step_name in pred_args else {}
            )
            if isinstance(step, Trainer):
                x = step.predict(x, **step_pred_args)
            elif isinstance(step, TrainingSystem):
                x = step.predict(x, step_pred_args)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x

    def train(self, x: Any, y: Any, train_args: dict[str, Any] = {}) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system."""

        # Loop through each step and call the train method
        for step in self.steps:
            step_name = step.__class__.__name__
            step_train_args = (
                train_args[step_name] if train_args and step_name in train_args else {}
            )
            if isinstance(step, Trainer):
                x, y = step.train(x, y, **step_train_args)
            elif isinstance(step, TrainingSystem):
                x, y = step.train(x, y, step_train_args)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x, y


class ParallelTrainingSystem(_System):
    """A system that trains the input data in parallel.

    ### Parameters:
    - steps (list[Trainer | TrainingSystem | ParallelTrainingSystem]): The steps in the system.

    ### Methods:
    ```python
    @abstractmethod
    def concat(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed data.

    def train(self, x: Any, y: Any) -> tuple[Any, Any]: # Train the system.

    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any: # Predict the output of the system.

    def concat_labels(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed labels.

    def get_hash(self) -> str: # Get the hash of the system.
    ```

    ### Usage:
    ```python
    from agogos.parallel_training_system import ParallelTrainingSystem

    trainer_1 = CustomTrainer()
    trainer_2 = CustomTrainer()

    class CustomParallelTrainingSystem(ParallelTrainingSystem):
        def concat(self, data1: Any, data2: Any) -> Any:
            # Concatenate the transformed data.
            return data1 + data2

    training_system = CustomParallelTrainingSystem(steps=[trainer_1, trainer_2])
    trained_x, trained_y = training_system.train(x, y)
    predictions = training_system.predict(x)
    ```
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
