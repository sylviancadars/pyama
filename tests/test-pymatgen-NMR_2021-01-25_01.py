# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:33:45 2021

@author: cadarp02
"""

from pymatgen.core.structure import IStructure
from pymatgen.io.vasp import Outcar
import os.path
import numpy as np
import matplotlib.pyplot as plt

dirName = r'D:\cadarp02\Documents\Programming\examples\VASP\NMR_TeO2-glass_270at_randSnap\NMR_StdPot_KSPA-0.30_nodes-6_tpn-27'
systemDescription = 'TeO2-glass, 270-atom random Snapshot'

struct = IStructure.from_file(os.path.join(dirName,'CONTCAR'))
listOfElmts = struct.symbol_set

# Reading NMR chemical shielding as printed in OUTCAR file
myOutcar = Outcar(os.path.join(dirName,'OUTCAR'))
myOutcar.read_chemical_shielding()
nmr_cs = np.asarray(myOutcar.data['chemical_shielding']['valence_and_core'])
legend=[]
for elmt in listOfElmts :
    # Extract Te shieldings
    indices = np.asarray(struct.indices_from_symbol(elmt))

    # Initializing plots
    fig = plt.figure(systemDescription + ' - ' + elmt + ' NMR shielding parameters')
    ax1 = fig.add_subplot(3, 1, 1)
    legend.append(elmt)
    ax1.scatter(np.arange(len(nmr_cs[indices,0])),nmr_cs[indices,0])
    ax1.legend(legend)
    ax1.set(xlabel='Site index',
            ylabel=elmt+' isotropic shielding (ppm)',
            title = systemDescription + ' - ' + elmt + ' isotropic shielding')
    
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(np.arange(len(nmr_cs[indices,1])),nmr_cs[indices,1])
    ax2.set(xlabel='Site index',
            ylabel=elmt+' span (ppm)',
            title = systemDescription + ' - ' + elmt + ' CS span')
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.scatter(np.arange(len(nmr_cs[indices,2])),nmr_cs[indices,2])
    ax3.set(xlabel='Site index',
            ylabel=elmt+' skew',
            title = systemDescription + ' - ' + elmt + ' CS skew')
    