import pandas as pd
import numpy as np

class Config():
    ALLOWED_ARRAY_TYPES = [list, tuple, pd.core.series.Series, np.ndarray]
