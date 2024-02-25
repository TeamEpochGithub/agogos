from agogos.refiner import Refiner
import pytest

class TestRefiner:

    def test_refiner(self):
        
        class refinerInstance(Refiner):
            def predict(self, predictions):
                return predictions
            
        refiner = refinerInstance()

        assert refiner.predict([1, 2, 3]) == [1, 2, 3]

    def test_refiner_abstract(self):
        refiner = Refiner()
        with pytest.raises(NotImplementedError):
            refiner.predict([1, 2, 3])

    def test_refiner_hash(self):
        refiner = Refiner()
        assert refiner.get_hash() == 'eef5b1817f60fa938f5fe9ee0899cc82'