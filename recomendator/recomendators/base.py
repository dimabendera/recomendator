import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator as BE

from ..configs import Config
from ..storages import Local
from ..metrics import sim_distance

class BaseEstimator(BE):
    """

    """
    def __init__(self, storage=Local, metric=sim_distance):
        """

        """
        self.storage = storage()
        self.metric = metric

    def fit(self, X, y, coefs=None, timestamps=None):
        """

        """
        self._fit_checkers(X, y, coefs, timestamps)

        sDict = {}
        i = 0
        for x in X:
            if x not in sDict.keys():
                sDict[x] = {}
            sDict[x][y[i]] = 1 * coefs[i]
            i += 1

        self.storage.store(sDict)

    def _fit_checkers(self, X, y, coefs, timestamps):
        """

        """
        assert type(X) in Config.ALLOWED_ARRAY_TYPES
        assert type(y) in Config.ALLOWED_ARRAY_TYPES
        assert len(X) == len(y)
        if type(coefs) != type(None):
            assert (type(coefs) in Config.ALLOWED_ARRAY_TYPES)
            assert len(X) == len(coefs)
        else:
            coefs = np.ones(len(X))
        if timestamps:
            assert (type(timestamps) in Config.ALLOWED_ARRAY_TYPES)
            assert len(X) == len(timestamps)

    def predict(self, X, y=None):
        """

        """
        pass