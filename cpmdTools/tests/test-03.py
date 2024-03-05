#!/usr/local/miniconda/envs/aiida/bin/python3

from pyama.cpmdTools.utils import CPMDData
from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData
# from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData

# import numpy as np
import sys
import matplotlib.pyplot as plt

dir_name = '/data/CPMD/As2Te3-beta/MD_SYM-4_EKINC-0.04_EFREQ-1600'
verbosity=2

cpmdd = CPMDData(dir_name, verbosity=verbosity,
                     system_description=dir_name, r_max=10, sigma=0.005,
                     exr_pdf_fNy=10)
cpmdd.set_simul_time(initial_simul_time=11500)
cpmdd.parse_output_file('output.out0')
print('Unit cell parameters (in \u212b):\n',
      cpmdd.initial_structure.lattice.lengths,
      cpmdd.initial_structure.lattice.angles)
cpmdd.parse_trajec_xyz_file(extract_every=100)

dmd = distanceMatrixData(Rmax=20, sigma=0.01)
partials, types, fig1, ax1 = dmd.calculate_all_partial_RDFs(
    cpmdd.initial_structure, showPlot=True)

partials, types, fig2, ax2 = dmd.calculate_all_partial_RDFs(
    cpmdd.trajectory[-1]['structure'], showPlot=True)

plt.show()


