# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:44:08 2023

@author: cadarp02
"""

from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData

root_dir = '../../examples/Si_MD_VASP_multirun'
tad = TrajAnalyzesData(use_xdatcar_file=True, keep_one_every=10, verbosity=2)
tad.set_vasp_data_dir_from_root_dir(root_dir) 
tad.set_trajectory_from_vasp()

fig1, ax1, fig2, ax2 = tad.track_displacements_above_threshold(2.0)

