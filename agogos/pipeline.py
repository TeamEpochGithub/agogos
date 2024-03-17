from dataclasses import dataclass
from typing import Any

from agogos.training import TrainingSystem
from agogos.transforming import TransformingSystem
from joblib import hash

from agogos._core import _Base


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
