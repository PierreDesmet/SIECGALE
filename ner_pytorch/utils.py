from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

import torch

if get_ipython() is not None:
    get_ipython().magic('matplotlib inline')
pd.set_option('max_columns', 150)
np.set_printoptions(suppress=True)
InteractiveShell.ast_node_interactivity = 'all'


def shift_tensor(t: torch.tensor):
    """
    Shift a 2-dim tensor just like pandas would shift a df.
    It is useful to get a sens of what the loss of a model could be, given 
    some desired (simulated) accuracy.
    :params t: is supposed to be the true
    Usage:
    >>> t = torch.tensor([[1, 0, 0],
                          [0, 1, 0]])
    >>> shift_tensor(t)
    tensor([[0, 1, 0],
            [0, 0, 1]])
    Note this function is meant to be used with ohe-like tensors.
    """
    t_all_wrong_preds = t[:, [-1] + list(range(t.ndim))]
    return t_all_wrong_preds