import pytest
from agogos.training import Trainer, TrainingSystem, ParallelTrainingSystem, Pipeline
from agogos.transforming import Transformer, TransformingSystem
import numpy as np


class TestTrainer:
    def test_trainer_abstract_train(self):
        trainer = Trainer()
        with pytest.raises(NotImplementedError):
            trainer.train([1, 2, 3], [1, 2, 3])

    def test_trainer_abstract_predict(self):
        trainer = Trainer()
        with pytest.raises(NotImplementedError):
            trainer.predict([1, 2, 3])

    def test_trainer_train(self):
        class trainerInstance(Trainer):
            def train(self, x, y):
                return x, y

        trainer = trainerInstance()
        assert trainer.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_trainer_predict(self):
        class trainerInstance(Trainer):
            def predict(self, x):
                return x

        trainer = trainerInstance()
        assert trainer.predict([1, 2, 3]) == [1, 2, 3]

    def test_trainer_hash(self):
        trainer = Trainer()
        assert trainer.get_hash() == "0a1fcf1d677d4a1f3f082aa85ffcb684"


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
        assert block2.get_parent() == block3
        assert block1 in block3.get_children()
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

        t1 = Transformer()
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
        t2 = Transformer()
        system.steps = [t1, t2]

        with pytest.raises(TypeError):
            system.train([1, 2, 3], [1, 2, 3])

        with pytest.raises(TypeError):
            system.predict([1, 2, 3])


class TestPipeline:
    def test_pipeline_init(self):
        pipeline = Pipeline()
        assert pipeline is not None

    def test_pipeline_init_with_systems(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        label_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
            label_sys=label_system,
        )
        assert pipeline is not None

    def test_pipeline_train(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        label_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
            label_sys=label_system,
        )
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_y_system(self):
        x_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_x_system(self):
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_train_system(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        post_system = TransformingSystem()
        post_label_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=None,
            pred_sys=post_system,
            label_sys=post_label_system,
        )
        assert pipeline.train([1, 2], [1, 2]) == ([1, 2], [1, 2])

    def test_pipeline_train_no_refining_system(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        pipeline = Pipeline(x_sys=x_system, y_sys=y_system, train_sys=training_system)
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_1_x_transform_block(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        result = pipeline.train(np.array([1, 2, 3]), [1, 2, 3])
        assert np.array_equal(result[0], np.array([2, 4, 6])) and np.array_equal(
            result[1], np.array([1, 2, 3])
        )

    def test_pipeline_predict(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_predict_no_y_system(self):
        x_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_predict_no_systems(self):
        pipeline = Pipeline()
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_get_hash_no_change(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        predicting_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=predicting_system,
        )
        assert pipeline.get_hash() == ""

    def test_pipeline_get_hash_with_change(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.get_hash() == "aa7c4cff990dc48c4e3456deed913e16"

    def test_pipeline_predict_system_hash(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem(steps=[transform1])
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.get_hash() == "842e1162d744e7ab09c941300a43c218"

    def test_pipeline_pre_post_hash(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem(steps=[transform1])
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.get_hash() == "3dda824076fddafd028812e7891fbd8b"
