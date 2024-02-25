import pytest
from agogos.transformer import Transformer
from agogos.transforming_system import TransformingSystem


class TestTransformingSystem:

    def test_transforming_system_init(self):
        transforming_system = TransformingSystem()
        assert transforming_system is not None

    def test_transforming_system_init_with_steps(self):
        class SubTransformer(Transformer):
            def predict(self, x):
                return x
        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system is not None

    def test_transforming_system_wrong_step(self):
        class SubTransformer:
            def predict(self, x):
                return x
        with pytest.raises(AssertionError):
            TransformingSystem(steps=[SubTransformer()])

    def test_transforming_system_transform(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x
        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.transform([1, 2, 3]) == [1, 2, 3]

    def test_transforming_system_predict(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x
        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.predict([1, 2, 3]) == [1, 2, 3]