import numpy as np
from agogos.pipeline import Pipeline
from agogos.refining_system import RefiningSystem
from agogos.training_system import TrainingSystem
from agogos.transformer import Transformer
from agogos.transforming_system import TransformingSystem


class TestPipeline:

    def test_pipeline_init(self):
        pipeline = Pipeline()
        assert pipeline is not None

    def test_pipeline_init_with_systems(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system, refining_system=refining_system)
        assert pipeline is not None

    def test_pipeline_train(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_y_system(self):
        x_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_x_system(self):
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(y_system=y_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_refining_system(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system)
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_1_x_transform_block(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2
        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system, refining_system=refining_system)
        result = pipeline.train(np.array([1, 2, 3]), [1, 2, 3])
        assert np.array_equal(result[0], np.array([2, 4, 6])) and np.array_equal(result[1], np.array([1, 2, 3]))


    def test_pipeline_predict(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_predict_no_y_system(self):
        x_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_get_hash_no_change(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.get_hash() == ''

    def test_pipeline_get_hash_with_change(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2
        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        refining_system = RefiningSystem()
        pipeline = Pipeline(x_system=x_system, y_system=y_system, training_system=training_system, refining_system=refining_system)
        assert pipeline.get_hash() == '19a73b64636386eb588d56fc0a294ba2'