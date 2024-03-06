import pytest
from agogos._core import _Base


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
