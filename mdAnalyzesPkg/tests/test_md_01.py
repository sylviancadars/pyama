from matplotlib import pyplot as plt
from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData
import numpy as np
from pymatgen import core

dir_name='/data/VASP/AsTe3/MD_1031912-sc212/500K_SMASS-0.14/407176'

print('pymatgen core version: ', core.__version__)

tad = TrajAnalyzesData(vasp_data_dir=dir_name, keep_one_every=100,
                       initial_simul_time=0, verbosity=0)
tad.set_trajectory_from_vasp()

traj = tad.trajectory
if isinstance(traj.lattice, np.ndarray):
    shape_or_length_str = ' of shape {}'.format(traj.lattice.shape)
else:
    shape_or_length_str = ' of length {}'.format(len(traj.lattice))
print('traj.lattice: {}{}'.format(type(traj.lattice), shape_or_length_str))
print('traj.constant_lattice = ', traj.constant_lattice)


"""
for k, v in traj.as_dict().items():
    shape_or_length_str = ''
    if isinstance(v, np.ndarray):
        shape_or_length_str = f' of shape {v.shape}'
    elif isinstance(v, list):
        shape_or_length_str = f' of length {len(v)}'
    print(f'{k}: {type(v)}{shape_or_length_str}')

fig, ax = tad.plot_thermodynamics_and_displacements()
plt.show()
"""
