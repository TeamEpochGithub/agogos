from dataclasses import dataclass
from typing import Any

from agogos.training_system import TrainingSystem

from agogos.transforming_system import TransformingSystem
from joblib import hash


@dataclass
class Pipeline:
    """A pipeline of systems that can be trained and predicted.

    :param x_system: The system to transform the input data.
    :param y_system: The system to transform the labelled data.
    :param training_system: The system to train the data.
    :param prediction_system: The system to transform the predictions.
    :param label_system: The system to transform the labels.
    """

    x_system: TransformingSystem | None = None
    y_system: TransformingSystem | None = None
    training_system: TrainingSystem | None = None
    prediction_system: TransformingSystem | None = None
    label_system: TransformingSystem | None = None

    def __post_init__(self) -> None:
        """Post init method for the Pipeline class."""
        self._set_hash("")

    def train(self, x: Any, y: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        if self.x_system is not None:
            x = self.x_system.transform(x)
        if self.y_system is not None:
            y = self.y_system.transform(y)
        if self.training_system is not None:
            x, y = self.training_system.train(x, y)
        if self.prediction_system is not None:
            x = self.prediction_system.transform(x)
        if self.label_system is not None:
            y = self.label_system.transform(y)

        return x, y

    def predict(self, x: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        if self.x_system is not None:
            x = self.x_system.transform(x)
        if self.training_system is not None:
            x = self.training_system.predict(x)
        if self.prediction_system is not None:
            x = self.prediction_system.transform(x)

        return x

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the pipeline.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        xy_hash = ""
        if self.x_system is not None:
            xy_hash += self.x_system.get_hash()
        if self.y_system is not None:
            xy_hash += self.y_system.get_hash()

        if xy_hash != "":
            self._hash = hash(self._hash + xy_hash)

        if self.training_system is not None:
            self.training_system._set_hash(self._hash)
            training_hash = self.training_system.get_hash()
            if training_hash != "":
                self._hash = hash(self._hash + training_hash)

        predlabel_hash = ""
        if self.prediction_system is not None:
            predlabel_hash += self.prediction_system.get_hash()
        if self.label_system is not None:
            predlabel_hash += self.label_system.get_hash()

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
