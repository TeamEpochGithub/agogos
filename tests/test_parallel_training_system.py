from agogos.parallel_training_system import ParallelTrainingSystem
from agogos.trainer import Trainer
from agogos.training_system import TrainingSystem
import pytest


class TestParallelTrainingSystem:
    def test_PTrainSys_init(self):
        system = ParallelTrainingSystem()

        assert system is not None

    def test_PTrainSys_init_trainers(self):
        t1 = Trainer()
        t2 = TrainingSystem()

        system = ParallelTrainingSystem(steps=[t1, t2])

        assert system is not None

    def test_PTrainSys_train(self):
        class trainer(Trainer):
            def train(self, x, y):
                return x, y

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2):
                return data1 + data2

            def concat_labels(self, data1, data2):
                return data1 + data2

        t1 = trainer()

        system = pts(steps=[t1])

        assert system is not None
        assert system.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_PTrainSys_trainers(self):
        class trainer(Trainer):
            def train(self, x, y):
                return x, y

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2):
                return data1 + data2

            def concat_labels(self, data1, data2):
                return data1 + data2

        t1 = trainer()
        t2 = trainer()

        system = pts(steps=[t1, t2])

        assert system is not None
        assert system.train([1, 2, 3], [1, 2, 3]) == (
            [1, 2, 3, 1, 2, 3],
            [1, 2, 3, 1, 2, 3],
        )

    def test_PTrainSys_predict(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2):
                return data1 + data2

        t1 = trainer()

        system = pts(steps=[t1])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3]

    def test_PTrainSys_predict_with_trainsys(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2):
                return data1 + data2

        t1 = trainer()
        t2 = TrainingSystem(steps=[t1])

        system = pts(steps=[t2, t1])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    def test_PTrainSys_predict_with_trainer_and_trainsys(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2):
                return data1 + data2

        t1 = trainer()
        t2 = trainer()
        t3 = TrainingSystem(steps=[t1, t2])

        system = pts(steps=[t1, t2, t3])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

    def test_PTrainSys_predictors(self):
        class trainer(Trainer):
            def predict(self, x):
                return x

        class pts(ParallelTrainingSystem):
            def concat(self, data1, data2):
                return data1 + data2

        t1 = trainer()
        t2 = trainer()

        system = pts(steps=[t1, t2])

        assert system is not None
        assert system.predict([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    def test_PTrainSys_concat_labels_throws_error(self):
        system = ParallelTrainingSystem()

        with pytest.raises(NotImplementedError):
            system.concat_labels([1, 2, 3], [4, 5, 6])

    def test_PTrainSys_step_1_changed(self):
        system = ParallelTrainingSystem()

        t1 = ParallelTrainingSystem()
        system.steps = [t1]

        with pytest.raises(TypeError):
            system.train([1, 2, 3], [1, 2, 3])

        with pytest.raises(TypeError):
            system.predict([1, 2, 3])

    def test_PTrainSys_step_2_changed(self):
        system = ParallelTrainingSystem()

        class trainer(Trainer):
            def train(self, x, y):
                return x, y

            def predict(self, x):
                return x

        t1 = trainer()
        t2 = ParallelTrainingSystem()
        system.steps = [t1, t2]

        with pytest.raises(TypeError):
            system.train([1, 2, 3], [1, 2, 3])

        with pytest.raises(TypeError):
            system.predict([1, 2, 3])
