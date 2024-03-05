# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:44:08 2023

@author: cadarp02
"""
from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData
from matplotlib import pyplot as plt

root_dir = '../../examples/Si_MD_VASP_multirun'
tad = TrajAnalyzesData(use_xdatcar_file=False, keep_one_every=1, verbosity=2)
tad.set_vasp_data_dir_from_root_dir(root_dir)
tad.set_trajectory_from_vasp()

# fig1, ax1 = tad.plot_thermodynamics_and_displacements()
fig2, ax2 = tad.plot_all_fractional_coordinates()
ax2[0, 0].set(title='Fractional coordinates before pbc correction')

print(tad.trajectory.frac_coords.shape)
tad.correct_pbc_problems()

# fig3, ax3, fig4, ax4 = tad.track_displacements_above_threshold(1.0)
fig3, ax3 = tad.plot_all_fractional_coordinates()
ax3[0, 0].set(title='Fractional coordinates after pbc correction')

plt.show()

