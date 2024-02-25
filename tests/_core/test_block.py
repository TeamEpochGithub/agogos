from agogos._core.block import Block


class TestBlock:
    def test_block_init(self):
        block = Block()
        assert block is not None

    def test_block_set_hash(self):
        block = Block()
        block._set_hash("")
        hash1 = block.get_hash()
        assert hash1 == "dae14fc4779a593026f01ebaade73e3a"
        block._set_hash(hash1)
        hash2 = block.get_hash()
        assert hash2 == "029df394f68d2e6c61daf1c0c66afad0"
        assert hash1 != hash2

    def test_block_get_hash(self):
        block = Block()
        block._set_hash("")
        hash1 = block.get_hash()
        assert hash1 == "dae14fc4779a593026f01ebaade73e3a"
