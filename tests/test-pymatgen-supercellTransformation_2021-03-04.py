# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:38:00 2021

@author: cadarp02
"""

from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
import os

fileName = r'D:\cadarp02\Documents\IRCER\AsTe3\USPEX-models_2021-02-08\AsTe3_Z-6_lowestE-001_ID-688.cif'
outFileName = os.path.join(os.path.dirname(fileName),'test.cif')

struct1 = Structure.from_file(fileName)
transform = SupercellTransformation( ((1,1,0),(1,-1,0),(0,0,2)) )
# -> yields almost-orthorhomic cell

# transform = SupercellTransformation( ((-1,2,0),(0,0,2),(1,0,0)) )
# Attempt to aligne layers with (a,b) plane. Not very convincing.

struct2 = transform.apply_transformation(struct1)


struct2.to('cif',outFileName)

