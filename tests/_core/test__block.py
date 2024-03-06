from agogos._core import _Block


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
