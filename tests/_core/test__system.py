from agogos._core import _System, _Block 


class TestSystem:
    def test_system_init(self):
        system = _System()
        assert system is not None

    def test_system_hash_no_steps(self):
        system = _System()
        assert system.get_hash() == ""

    def test_system_hash_with_steps(self):
        block1 = _Block()

        system = _System([block1])
        assert system.get_hash() == "3de71998c76824d4e9c46e6894f82460"
