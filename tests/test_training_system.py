import pytest
from agogos.trainer import Trainer

from agogos.training_system import TrainingSystem


class TestTrainingSystem:
    def test_training_system_init(self):
        training_system = TrainingSystem()
        assert training_system is not None

    def test_training_system_init_with_steps(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x

        block1 = SubTrainer()
        training_system = TrainingSystem(steps=[block1])
        assert training_system is not None

    def test_training_system_wrong_step(self):
        class SubTrainer:
            def predict(self, x):
                return x

        with pytest.raises(AssertionError):
            TrainingSystem(steps=[SubTrainer()])

    def test_training_system_steps_changed_predict(self):
        class SubTrainer:
            def predict(self, x):
                return x

        block1 = SubTrainer()
        training_system = TrainingSystem()
        training_system.steps = [block1]
        with pytest.raises(TypeError):
            training_system.predict([1, 2, 3])

    def test_training_system_predict(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x

        block1 = SubTrainer()
        training_system = TrainingSystem(steps=[block1])
        assert training_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_trainsys_predict_with_trainer_and_trainsys(self):
        class SubTrainer(Trainer):
            def predict(self, x):
                return x

        block1 = SubTrainer()
        block2 = SubTrainer()
        block3 = TrainingSystem(steps=[block1, block2])
        training_system = TrainingSystem(steps=[block1, block2, block3])
        assert training_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_training_system_train(self):
        class SubTrainer(Trainer):
            def train(self, x, y):
                return x, y

        block1 = SubTrainer()
        training_system = TrainingSystem(steps=[block1])
        assert training_system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_traiinsys_train_with_trainer_and_trainsys(self):
        class SubTrainer(Trainer):
            def train(self, x, y):
                return x, y

        block1 = SubTrainer()
        block2 = SubTrainer()
        block3 = TrainingSystem(steps=[block1, block2])
        training_system = TrainingSystem(steps=[block1, block2, block3])
        assert training_system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_training_system_steps_changed_train(self):
        class SubTrainer:
            def train(self, x, y):
                return x, y

        block1 = SubTrainer()
        training_system = TrainingSystem()
        training_system.steps = [block1]
        with pytest.raises(TypeError):
            training_system.train([1, 2, 3], [1, 2, 3])

    def test_training_system_empty_hash(self):
        training_system = TrainingSystem()
        assert training_system.get_hash() == ""
