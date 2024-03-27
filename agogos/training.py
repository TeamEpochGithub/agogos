from abc import abstractmethod
from joblib import hash
from typing import Any
from dataclasses import dataclass
from agogos._core import _Block, _System, _Base
from agogos.transforming import TransformingSystem


class Trainer(_Block):
    """The trainer block is for blocks that need to train on two inputs and predict on one.

    Methods:
    .. code-block:: python
        @abstractmethod
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]: # Train the block.

        @abstractmethod
        def predict(self, x: Any, **pred_args: Any) -> Any: # Predict the target variable.

        def get_hash(self) -> str: # Get the hash of the block.

    Usage:
    .. code-block:: python
        from agogos.trainer import Trainer

        class MyTrainer(Trainer):

            def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
                # Train the block.
                return x, y

            def predict(self, x: Any, **pred_args: Any) -> Any:
                # Predict the target variable.
                return x

        my_trainer = MyTrainer()
        predictions, labels = my_trainer.train(x, y)
        predictions = my_trainer.predict(x)
    """

    @abstractmethod
    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement train method."
        )

    @abstractmethod
    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict method."
        )


class TrainingSystem(_System):
    """A system that trains on the input data and labels.

    Parameters:
    - steps (list[Trainer | TrainingSystem | ParallelTrainingSystem]): The steps in the system.

    Methods:
    .. code-block:: python
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, **pred_args: Any) -> Any: # Predict the output of the system.

        def get_hash(self) -> str: # Get the hash of the system.

    Usage:
    .. code-block:: python
        from agogos.training_system import TrainingSystem

        trainer_1 = CustomTrainer()
        trainer_2 = CustomTrainer()

        training_system = TrainingSystem(steps=[trainer_1, trainer_2])
        trained_x, trained_y = training_system.train(x, y)
        predictions = training_system.predict(x)
    """

    def __post_init__(self) -> None:
        """Post init method for the TrainingSystem class."""

        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            assert issubclass(step.__class__, Trainer) or issubclass(
                step.__class__, TrainingSystem
            ), f"{step} is not a subclass of Trainer"

        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system."""

        # Loop through each step and call the train method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = train_args.get(step_name, {})

            if (
                isinstance(step, Trainer)
                or isinstance(step, TrainingSystem)
                or isinstance(step, ParallelTrainingSystem)
                or isinstance(step, Pipeline)
            ):
                x, y = step.train(x, y, **step_args)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x, y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each step and call the predict method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = pred_args.get(step_name, {})

            if (
                isinstance(step, Trainer)
                or isinstance(step, TrainingSystem)
                or isinstance(step, ParallelTrainingSystem)
                or isinstance(step, Pipeline)
            ):
                x = step.predict(x, **step_args)
            else:
                raise TypeError(f"{step} is not a subclass of Trainer")

        return x


class ParallelTrainingSystem(_System):
    """A system that trains the input data in parallel.

    Parameters:
    - steps (list[Trainer | TrainingSystem | ParallelTrainingSystem]): The steps in the system.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed data.

        def train(self, x: Any, y: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any: # Predict the output of the system.

        def concat_labels(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed labels.

        def get_hash(self) -> str: # Get the hash of the system.

    Usage:
    .. code-block:: python
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
    """

    def __post_init__(self) -> None:
        """Post init method for the ParallelTrainingSystem class."""

        # Assert all steps are a subclass of Trainer or TrainingSystem
        for step in self.steps:
            assert issubclass(step.__class__, Trainer) or issubclass(
                step.__class__, TrainingSystem
            ), f"{step} is not a subclass of Trainer or TrainingSystem"

        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """

        # Loop through each step and call the train method
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__

            step_args = train_args.get(step_name, {})

            if (
                isinstance(step, Trainer)
                or isinstance(step, TrainingSystem)
                or isinstance(step, ParallelTrainingSystem)
                or isinstance(step, Pipeline)
            ):
                if i == 0:
                    x, y = step.train(x, y, **step_args)
                else:
                    new_x, new_y = step.train(x, y, **step_args)
                    x, y = self.concat(x, new_x), self.concat_labels(y, new_y)
            else:
                raise TypeError(
                    f"{step} is not a subclass of Trainer, TrainingSystem or ParallelTrainingSystem"
                )

        return x, y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """

        # Loop through each trainer and call the predict method
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__

            step_args = pred_args.get(step_name, {})

            if (
                isinstance(step, Trainer)
                or isinstance(step, TrainingSystem)
                or isinstance(step, ParallelTrainingSystem)
                or isinstance(step, Pipeline)
            ):
                if i == 0:
                    x = step.predict(x, **step_args)
                else:
                    x_new = step.predict(x, **step_args)
                    x = self.concat(x, x_new)
            else:
                raise TypeError(
                    f"{step} is not a subclass of Trainer or TrainingSystem"
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


@dataclass
class Pipeline(_Base):
    """A pipeline of systems that can be trained and predicted.

    Parameters:
    - x_sys (TransformingSystem | None): The system to transform the input data.
    - y_sys (TransformingSystem | None): The system to transform the labelled data.
    - train_sys (TrainingSystem | None): The system to train the data.
    - pred_sys (TransformingSystem | None): The system to transform the predictions.
    - label_sys (TransformingSystem | None): The system to transform the labels.

    Methods:
    .. code-block:: python
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, **pred_args) -> Any: # Predict the output of the system.

        def get_hash(self) -> str: # Get the hash of the pipeline.

    Usage:
    .. code-block:: python
        from agogos.pipeline import Pipeline

        x_sys = CustomTransformingSystem()
        y_sys = CustomTransformingSystem()
        train_sys = CustomTrainingSystem()
        pred_sys = CustomTransformingSystem()
        label_sys = CustomTransformingSystem()

        pipeline = Pipeline(x_sys=x_sys, y_sys=y_sys, train_sys=train_sys, pred_sys=pred_sys, label_sys=label_sys)
        trained_x, trained_y = pipeline.train(x, y)
        predictions = pipeline.predict(x)
    """

    x_sys: TransformingSystem | None = None
    y_sys: TransformingSystem | None = None
    train_sys: TrainingSystem | None = None
    pred_sys: TransformingSystem | None = None
    label_sys: TransformingSystem | None = None

    def __post_init__(self) -> None:
        """Post initialization function of the Pipeline."""
        super().__post_init__()

        # Set children and parents
        children = []
        systems = [
            self.x_sys,
            self.y_sys,
            self.train_sys,
            self.pred_sys,
            self.label_sys,
        ]

        for sys in systems:
            if sys is not None:
                sys._set_parent(self)
                children.append(sys)

        self._set_children(children)

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The arguments to pass to the training system. (Default is {})
        :return: The input and output of the system.
        """
        if self.x_sys is not None:
            x = self.x_sys.transform(x, **train_args.get("x_sys", {}))
        if self.y_sys is not None:
            y = self.y_sys.transform(y, **train_args.get("y_sys", {}))
        if self.train_sys is not None:
            x, y = self.train_sys.train(x, y, **train_args.get("train_sys", {}))
        if self.pred_sys is not None:
            x = self.pred_sys.transform(x, **train_args.get("pred_sys", {}))
        if self.label_sys is not None:
            y = self.label_sys.transform(y, **train_args.get("label_sys", {}))

        return x, y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :param pred_args: The arguments to pass to the prediction system. (Default is {})
        :return: The output of the system.
        """
        if self.x_sys is not None:
            x = self.x_sys.transform(x, **pred_args.get("x_sys", {}))
        if self.train_sys is not None:
            x = self.train_sys.predict(x, **pred_args.get("train_sys", {}))
        if self.pred_sys is not None:
            x = self.pred_sys.transform(x, **pred_args.get("pred_sys", {}))

        return x

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the pipeline.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        xy_hash = ""
        if self.x_sys is not None:
            xy_hash += self.x_sys.get_hash()
        if self.y_sys is not None:
            xy_hash += self.y_sys.get_hash()

        if xy_hash != "":
            self._hash = hash(self._hash + xy_hash)

        if self.train_sys is not None:
            self.train_sys._set_hash(self._hash)
            training_hash = self.train_sys.get_hash()
            if training_hash != "":
                self._hash = hash(self._hash + training_hash)

        predlabel_hash = ""
        if self.pred_sys is not None:
            predlabel_hash += self.pred_sys.get_hash()
        if self.label_sys is not None:
            predlabel_hash += self.label_sys.get_hash()

        if predlabel_hash != "":
            if self._hash == "":
                self._hash = predlabel_hash
            else:
                self._hash = hash(self._hash + predlabel_hash)
