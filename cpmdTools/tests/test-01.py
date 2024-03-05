#!/usr/local/miniconda/envs/aiida/bin/python3

from pyama.cpmdTools.utils import CPMDData
# from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData

# import numpy as np
import pandas as pd

df = pd.DataFrame({'col0':[1, 2, 3], 'col2':[4, 5, 6]})

df.assign(newcol=[7,8,9])
print(df)


