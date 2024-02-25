import numpy as np
import pytest
from agogos.transformer import Transformer
from agogos.transforming_system import TransformingSystem


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

    def test_transforming_system_transform_2_blocks(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x * 2

        block1 = SubTransformer()
        block2 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1, block2])
        result = transforming_system.transform(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([4, 8, 12]))

    def test_transforming_system_predict(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_transforming_system_empty_hash(self):
        transforming_system = TransformingSystem()
        assert transforming_system.get_hash() == ""
