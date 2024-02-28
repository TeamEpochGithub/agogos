from dataclasses import dataclass
from typing import Any

from agogos.training_system import TrainingSystem

from agogos.transforming_system import TransformingSystem
from joblib import hash


@dataclass
class Pipeline:
    """A pipeline of systems that can be trained and predicted.

    :param x_sys: The system to transform the input data.
    :param y_sys: The system to transform the labelled data.
    :param train_sys: The system to train the data.
    :param pred_sys: The system to transform the predictions.
    :param label_sys: The system to transform the labels.
    """

    x_sys: TransformingSystem | None = None
    y_sys: TransformingSystem | None = None
    train_sys: TrainingSystem | None = None
    pred_sys: TransformingSystem | None = None
    label_sys: TransformingSystem | None = None

    def __post_init__(self) -> None:
        """Post init method for the Pipeline class."""
        self._set_hash("")

    def train(
        self, x: Any, y: Any, train_args: dict[str, Any] | None = None
    ) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        if self.x_sys is not None:
            x = self.x_sys.transform(
                x,
                train_args["x_sys"]
                if train_args is not None and "x_sys" in train_args
                else None,
            )
        if self.y_sys is not None:
            y = self.y_sys.transform(
                y,
                train_args["y_sys"]
                if train_args is not None and "y_sys" in train_args
                else None,
            )
        if self.train_sys is not None:
            x, y = self.train_sys.train(
                x,
                y,
                train_args["train_sys"]
                if train_args is not None and "train_sys" in train_args
                else None,
            )
        if self.pred_sys is not None:
            x = self.pred_sys.transform(
                x,
                train_args["pred_sys"]
                if train_args is not None and "pred_sys" in train_args
                else None,
            )
        if self.label_sys is not None:
            y = self.label_sys.transform(
                y,
                train_args["label_sys"]
                if train_args is not None and "label_sys" in train_args
                else None,
            )

        return x, y

    def predict(self, x: Any, pred_args: dict[str, Any] | None = None) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        if self.x_sys is not None:
            x = self.x_sys.transform(
                x,
                pred_args["x_sys"]
                if pred_args is not None and "x_sys" in pred_args
                else None,
            )
        if self.train_sys is not None:
            x = self.train_sys.predict(
                x,
                pred_args["train_sys"]
                if pred_args is not None and "train_sys" in pred_args
                else None,
            )
        if self.pred_sys is not None:
            x = self.pred_sys.transform(
                x,
                pred_args["pred_sys"]
                if pred_args is not None and "pred_sys" in pred_args
                else None,
            )

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

    def get_hash(self) -> str:
        """Get the hash of the pipeline.

        :return: The hash of the pipeline.
        """
        return self._hash
