from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

if get_ipython() is not None:
    get_ipython().magic('matplotlib inline')
pd.set_option('max_columns', 150)
np.set_printoptions(suppress=True)
InteractiveShell.ast_node_interactivity = 'all'

