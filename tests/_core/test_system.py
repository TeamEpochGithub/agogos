import numpy as np
import pytest
from agogos._core._block import Block
from agogos._core._system import System


class TestSystem:
    def test_system_init(self):
        system = System()
        assert system is not None

    def test_system_predict(self):
        system = System()
        with pytest.raises(NotImplementedError):
            system.predict(np.array([1, 2, 3]))

    def test_system_hash_no_steps(self):
        system = System()
        assert system.get_hash() == ""

    def test_system_hash_with_steps(self):
        block1 = Block()

        system = System([block1])
        assert system.get_hash() == "87c2f610eca16f4d524177c6382ef6b9"
