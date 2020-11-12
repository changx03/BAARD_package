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
        n = len(X)
        passed = np.ones(n, dtype=np.int8)
        encoded_adv = self.encoding(X)

        passed = self.def_stage1_(encoded_adv, pred, passed)
        blocked = len(passed[passed == 0])
        logger.debug('Stage 1: blocked %d inputs', blocked)
        # self.blocked_by_stages[0] = blocked

    def def_stage1_(self, adv, pred_adv, passed):
        """
        A bounding box which uses [min, max] from traning set
        """
        if len(np.where(passed == 1)[0]) == 0:
            return passed

        for i in range(self.num_classes):
            indices = np.where(pred_adv == i)[0]
            x = adv[indices]
            i_min = self._x_min[i]
            i_max = self._x_max[i]
            blocked_indices = np.where(
                np.all(np.logical_or(x < i_min, x > i_max), axis=1)
            )[0]
            if len(blocked_indices) > 0:
                passed[blocked_indices] = 0
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