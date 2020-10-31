from abc import ABC, abstractmethod

class BAARD(ABC):
    """
    Base class for BAARD
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    # @abstractmethod
    # def detect(self, X):
    #     """Detect adversarial examples."""
    #     raise NotImplementedError

    # @abstractmethod
    # def embedding(self, X):
    #     raise NotImplementedError