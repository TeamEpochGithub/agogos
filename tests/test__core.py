from agogos._core import _Block, _Base, _System
import pytest


class Test_Base:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            _Base()

    def test_get_hash(self):
        with pytest.raises(NotImplementedError):
            _Base().get_hash()

    def test_set_hash(self):
        with pytest.raises(NotImplementedError):
            _Base()._set_hash("prev_hash")


class TestBlock:
    def test_block_init(self):
        block = _Block()
        assert block is not None

    def test_block_set_hash(self):
        block = _Block()
        block._set_hash("")
        hash1 = block.get_hash()
        assert hash1 == "04714d9ee40c9baff8c528ed982a103c"
        block._set_hash(hash1)
        hash2 = block.get_hash()
        assert hash2 == "83196595c42f8eff9218c0ac8f80faf0"
        assert hash1 != hash2

    def test_block_get_hash(self):
        block = _Block()
        block._set_hash("")
        hash1 = block.get_hash()
        assert hash1 == "04714d9ee40c9baff8c528ed982a103c"


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
