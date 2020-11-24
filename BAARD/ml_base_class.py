from abc import ABC, abstractmethod
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)


class BAARD(ABC):
    """
    Base class for BAARD
    """
    def __init__(self, model=None):
        self.model = model
        # raise error if model is not an object
        if model == None:
            raise TypeError("Model parameter has been incorrectly passed")

    def detect(self, X):
        """Detect adversarial examples."""
        pred = self.predict(X)
        encoded_adv = self.encoding(X) # returns X right now
        passed = self.def_stage1_(encoded_adv, pred, passed)
        passed = self.def_stage2_(encoded_adv, pred, passed)
        passed = self.def_stage2_(encoded_adv, pred, passed)

    def fit(self):
        """ Fit defense """
        pass

    def def_stage1_(self, X, pred_adv, passed):
        """
        A bounding box which uses [min, max] from training set
        """
        return passed

    def def_stage2_(self, X, pred_adv, passed):
        """
        Filtering the inputs based on in-class k nearest neighbours.
        """
        return passed

    def def_stage3_(self, X, pred_adv, passed):
        """
        Checking the class distribution of k nearest neighbours without predicting
        the inputs. Compute the likelihood using one-against-all approach.
        """
        return passed

    def _log_time_start(self):
        self._since = time.time()

    def _log_time_end(self, title=None):
        time_elapsed = time.time() - self._since
        title = ' [' + title + ']' if title else ''
        logger.debug(
            'Time to complete%s: %dm %.3fs',
            title, int(time_elapsed // 60), time_elapsed % 60)

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def encoding(self, X):
        return X