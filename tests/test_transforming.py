import pytest
from agogos.training import Trainer
from agogos.transforming import (
    Transformer,
    TransformingSystem,
    ParallelTransformingSystem,
)
import numpy as np


class TestTransformer:
    def test_transformer_abstract(self):
        transformer = Transformer()

        with pytest.raises(NotImplementedError):
            transformer.transform([1, 2, 3])

    def test_transformer_transform(self):
        class transformerInstance(Transformer):
            def transform(self, data):
                return data

        transformer = transformerInstance()

        assert transformer.transform([1, 2, 3]) == [1, 2, 3]

    def test_transformer_hash(self):
        transformer = Transformer()
        assert transformer.get_hash() == "1cbcc4f2d0921b050d9b719d2beb6529"


class TestTransformingSystem:
    def test_transforming_system_init(self):
        transforming_system = TransformingSystem()
        assert transforming_system is not None

    def test_transforming_system_init_with_steps(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system is not None

    def test_transforming_system_wrong_step(self):
        class SubTransformer:
            def transform(self, x):
                return x

        with pytest.raises(AssertionError):
            TransformingSystem(steps=[SubTransformer()])

    def test_transforming_system_steps_changed(self):
        class SubTransformer:
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem()
        transforming_system.steps = [block1]
        with pytest.raises(TypeError):
            transforming_system.transform([1, 2, 3])

    def test_transforming_system_transform_1_block(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.transform([1, 2, 3]) == [1, 2, 3]

    def test_transforming_system_transform_1_block_with_args(self):
        class SubTransformer(Transformer):
            def transform(self, data):
                return data

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.transform([1, 2, 3], **{"SubTransformer": {}}) == [
            1,
            2,
            3,
        ]

    def test_transforming_system_transform_2_blocks(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x * 2

        block1 = SubTransformer()
        block2 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1, block2])
        result = transforming_system.transform(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([4, 8, 12]))

    def test_transformsys_with_transformsys(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x * 2

        block1 = SubTransformer()
        block2 = TransformingSystem(steps=[block1])
        transforming_system = TransformingSystem(steps=[block2])
        result = transforming_system.transform(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_transforming_system_transform_with_args(self):
        class SubTransformer(Transformer):
            def transform(self, data, multiplier=2):
                return data * multiplier

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        result = transforming_system.transform(
            np.array([1, 2, 3]), **{"SubTransformer": {"multiplier": 2}}
        )
        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_transforming_system_transform_with_args_2_blocks(self):
        class SubTransformer(Transformer):
            def transform(self, data, multiplier=2):
                return data * multiplier

        block1 = SubTransformer()
        block2 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1, block2])
        result = transforming_system.transform(
            np.array([1, 2, 3]), **{"SubTransformer": {"multiplier": 2}}
        )
        assert np.array_equal(result, np.array([4, 8, 12]))

    def test_transforming_system_transform_with_recursive_args(self):
        class SubTransformer(Transformer):
            def transform(self, data, multiplier=2):
                return data * multiplier

        block1 = SubTransformer()
        block2 = SubTransformer()
        block3 = TransformingSystem(steps=[block2])
        block4 = TransformingSystem(steps=[block3])
        transforming_system = TransformingSystem(steps=[block1, block4])
        assert np.array_equal(
            transforming_system.transform(
                np.array([1, 2, 3]), **{"SubTransformer": {"multiplier": 2}}
            ),
            np.array([4, 8, 12]),
        )
        assert np.array_equal(
            transforming_system.transform(
                np.array([1, 2, 3]),
                **{
                    "TransformingSystem": {
                        "TransformingSystem": {"SubTransformer": {"multiplier": 3}}
                    }
                },
            ),
            np.array([6, 12, 18]),
        )

    def test_transforming_system_empty_hash(self):
        transforming_system = TransformingSystem()
        assert transforming_system.get_hash() == ""


class TestParallelTransformingSystem:
    def test_parallel_transforming_system(self):
        # Create an instance of the system
        system = ParallelTransformingSystem()

        # Assert the system is an instance of ParallelTransformingSystem
        assert isinstance(system, ParallelTransformingSystem)
        assert system is not None

    def test_parallel_transforming_system_transformers(self):
        transformer1 = Transformer()
        transformer2 = TransformingSystem()

        system = ParallelTransformingSystem(steps=[transformer1, transformer2])
        assert system is not None

    def test_parallel_transforming_system_transform(self):
        class transformer(Transformer):
            def transform(self, data):
                return data

        class pts(ParallelTransformingSystem):
            def concat(self, data1, data2):
                return data1 + data2

        t1 = transformer()

        system = pts(steps=[t1])

        assert system is not None
        assert system.transform([1, 2, 3]) == [1, 2, 3]

    def test_pts_transformers_transform(self):
        class transformer(Transformer):
            def transform(self, data):
                return data

        class pts(ParallelTransformingSystem):
            def concat(self, data1, data2):
                return data1 + data2

        t1 = transformer()
        t2 = transformer()

        system = pts(steps=[t1, t2])

        assert system is not None
        assert system.transform([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    def test_parallel_transforming_system_concat_throws_error(self):
        system = ParallelTransformingSystem()

        with pytest.raises(NotImplementedError):
            system.concat([1, 2, 3], [4, 5, 6])

    def test_pts_step_1_changed(self):
        system = ParallelTransformingSystem()

        t1 = Trainer()
        system.steps = [t1]

        with pytest.raises(TypeError):
            system.transform([1, 2, 3])

    def test_pts_step_2_changed(self):
        system = ParallelTransformingSystem()

        class transformer(Transformer):
            def transform(self, data):
                return data

        t1 = transformer()
        t2 = Trainer()
        system.steps = [t1, t2]

        with pytest.raises(TypeError):
            system.transform([1, 2, 3])
