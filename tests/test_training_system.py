import pytest
from agogos.trainer import Trainer

from agogos.training_system import TrainingSystem


class TestTrainingSystem:

    def test_training_system_init(self):
        transforming_system = TrainingSystem()
        assert transforming_system is not None

    def test_training_system_init_with_steps(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x
        block1 = SubTrainer()
        transforming_system = TrainingSystem(steps=[block1])
        assert transforming_system is not None

    def test_transforming_system_wrong_step(self):
        class SubTrainer:
            def predict(self, x):
                return x
        with pytest.raises(AssertionError):
            TrainingSystem(steps=[SubTrainer()])

    def test_transforming_system_predict(self):
        class SubTransformer(Trainer):
            def predict(self, x):
                return x
        block1 = SubTransformer()
        transforming_system = TrainingSystem(steps=[block1])
        assert transforming_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_transforming_system_train(self):
        class SubTransformer(Trainer):
            def train(self, x, y):
                return x, y
        block1 = SubTransformer()
        transforming_system = TrainingSystem(steps=[block1])
        assert transforming_system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_training_system_empty_hash(self):
        training_system = TrainingSystem()
        assert training_system.get_hash() == ''