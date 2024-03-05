from matplotlib import pyplot as plt
from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData
import numpy as np
from pymatgen import core

dir_name='/data/VASP/AsTe3/MD_1031912-sc212/500K_SMASS-0.14/407176'

print('pymatgen core version: ', core.__version__)

tad = TrajAnalyzesData(vasp_data_dir=dir_name, keep_one_every=100,
                       initial_simul_time=0, verbosity=2)
tad.set_trajectory_from_vasp()

traj = tad.trajectory

fig, ax = tad.plot_thermodynamics_and_displacements()

traj_indexes = tad.pick_evenly_spaced_traj_indexes(max_nb_of_frames=10)
time, distances, fig2, ax2 = tad.calculate_structure_evolution(traj_indexes,
                                                               sigma=0.2, r_max=12.0)

plt.show()

