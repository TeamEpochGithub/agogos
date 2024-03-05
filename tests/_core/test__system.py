import numpy as np
import pytest
from agogos._core._block import _Block
from agogos._core._system import _System


class TestSystem:
    def test_system_init(self):
        system = _System()
        assert system is not None

    def test_system_predict(self):
        system = _System()
        with pytest.raises(NotImplementedError):
            system.predict(np.array([1, 2, 3]))

    def test_system_hash_no_steps(self):
        system = _System()
        assert system.get_hash() == ""

    def test_system_hash_with_steps(self):
        block1 = _Block()

        system = _System([block1])
        assert system.get_hash() == "3de71998c76824d4e9c46e6894f82460"
