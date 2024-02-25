import pytest
from agogos.refiner import Refiner
from agogos.refining_system import RefiningSystem


class TestRefiningSystem:
    def test_refining_system_init(self):
        refining_system = RefiningSystem()
        assert refining_system is not None

    def test_refining_system_init_with_steps(self):
        class SubRefiner(Refiner):
            def refine(self, x):
                return x

        block1 = SubRefiner()
        refining_system = RefiningSystem(steps=[block1])
        assert refining_system is not None

    def test_refining_system_wrong_step(self):
        class SubRefiner:
            def refine(self, x):
                return x

        with pytest.raises(AssertionError):
            RefiningSystem(steps=[SubRefiner()])

    def test_refining_system_predict(self):
        class SubRefiner(Refiner):
            def predict(self, x):
                return x

        block1 = SubRefiner()
        refining_system = RefiningSystem(steps=[block1])
        assert refining_system.predict([1, 2, 3]) == [1, 2, 3]

    def test_refining_system_empty_hash(self):
        refining_system = RefiningSystem()
        assert refining_system.get_hash() == ""
