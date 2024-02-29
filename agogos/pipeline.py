from dataclasses import dataclass
from typing import Any

from agogos.training_system import TrainingSystem

from agogos.transforming_system import TransformingSystem
from joblib import hash

from agogos._core._base import _Base


@dataclass
class Pipeline(_Base):
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

    def train(self, x: Any, y: Any, train_args: dict[str, Any] = {}) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The arguments to pass to the training system. (Default is {})
        :return: The input and output of the system.
        """
        if self.x_sys is not None:
            x_sys_args = (
                train_args["x_sys"] if train_args and "x_sys" in train_args else {}
            )
            x = self.x_sys.transform(x, x_sys_args)
        if self.y_sys is not None:
            y_sys_args = (
                train_args["y_sys"] if train_args and "y_sys" in train_args else {}
            )
            y = self.y_sys.transform(y, y_sys_args)
        if self.train_sys is not None:
            train_sys_args = (
                train_args["train_sys"]
                if train_args and "train_sys" in train_args
                else {}
            )
            x, y = self.train_sys.train(x, y, train_sys_args)
        if self.pred_sys is not None:
            pred_sys_args = (
                train_args["pred_sys"]
                if train_args and "pred_sys" in train_args
                else {}
            )
            x = self.pred_sys.transform(x, pred_sys_args)
        if self.label_sys is not None:
            label_sys_args = (
                train_args["label_sys"]
                if train_args and "label_sys" in train_args
                else {}
            )
            y = self.label_sys.transform(y, label_sys_args)

        return x, y

    def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :param pred_args: The arguments to pass to the prediction system. (Default is {})
        :return: The output of the system.
        """
        if self.x_sys is not None:
            x_sys_args = (
                pred_args["x_sys"] if pred_args and "x_sys" in pred_args else {}
            )
            x = self.x_sys.transform(x, x_sys_args)
        if self.train_sys is not None:
            train_sys_args = (
                pred_args["train_sys"] if pred_args and "train_sys" in pred_args else {}
            )
            x = self.train_sys.predict(x, train_sys_args)
        if self.pred_sys is not None:
            pred_sys_args = (
                pred_args["pred_sys"] if pred_args and "pred_sys" in pred_args else {}
            )
            x = self.pred_sys.transform(x, pred_sys_args)

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
