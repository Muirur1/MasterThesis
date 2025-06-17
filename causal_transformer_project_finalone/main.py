import sys

import pandas as pd
import numpy as np
import matplotlib
import seaborn
import statsmodels
import sklearn
import missingno
import scipy
import torch
import tqdm
import hydra
import omegaconf
import optuna


# Write to requirements.txt
with open("requirements.txt", "w") as f:
    f.write(f"Python version: {sys.version.split()[0]}\n")
    f.write(f"Anaconda Navigator version: 2.6.6\n")
    f.write(f"pandas=={pd.__version__}\n")
    f.write(f"numpy=={np.__version__}\n")
    f.write(f"matplotlib=={matplotlib.__version__}\n")
    f.write(f"seaborn=={seaborn.__version__}\n")
    f.write(f"statsmodels=={statsmodels.__version__}\n")
    f.write(f"scikit-learn=={sklearn.__version__}\n")
    f.write(f"missingno=={missingno.__version__}\n")
    f.write(f"scipy=={scipy.__version__}\n")
    f.write(f"torch=={torch.__version__}\n")
    f.write(f"tqdm=={tqdm.__version__}\n")
    f.write(f"hydra-core=={hydra.__version__}\n")
    f.write(f"omegaconf=={omegaconf.__version__}\n")
    f.write(f"optuna=={optuna.__version__}\n")
