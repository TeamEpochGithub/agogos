from agogos.parallel_transforming_system import ParallelTransformingSystem
from agogos.transformer import Transformer
from agogos.transforming_system import TransformingSystem
import pytest


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

        t1 = ParallelTransformingSystem()
        system.steps = [t1]

        with pytest.raises(TypeError):
            system.transform([1, 2, 3])

    def test_pts_step_2_changed(self):
        system = ParallelTransformingSystem()

        class transformer(Transformer):
            def transform(self, data):
                return data

        t1 = transformer()
        t2 = ParallelTransformingSystem()
        system.steps = [t1, t2]

        with pytest.raises(TypeError):
            system.transform([1, 2, 3])
