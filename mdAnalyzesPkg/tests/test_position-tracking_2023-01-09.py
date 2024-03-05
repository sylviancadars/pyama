from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData
from matplotlib import pyplot as plt

root_dir = '/data/VASP/AsTe3/MD_1031912-sc212/500K_SMASS-0.14'
# root_dir = '../../examples/Si_MD_VASP_multirun'

tad = TrajAnalyzesData(description='Si continuation test from XDATCAR files',
                       verbosity=3)
tad.set_vasp_data_dir_from_root_dir(root_dir, 'XDATCAR')
tad.keep_one_every = 5
tad.set_trajectory_from_vasp(use_xdatcar_file=True)
fig, ax = tad.plot_thermodynamics_and_displacements()

# Detect As atomic positions moving out-of-plane by a certain theshold
# Plot the trajectory of every of them in a 3D graph using a parametric curve

fig1, ax1, fig2, ax2 = tad.track_displacements_above_threshold(threshold=3.0,
                                                  tracking_axis=None)
plt.show()

