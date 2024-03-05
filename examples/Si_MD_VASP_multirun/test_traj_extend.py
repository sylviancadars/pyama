import os
from pymatgen import core
from pymatgen.io.vasp import Vasprun
import numpy as np

print('pymatgen core version {}'.format(core.__version__))

root_dir = os.getcwd()
dir_names = [os.path.join(root_dir, dir_name) for dir_name in
             [d for d in os.listdir(root_dir) if os.path.isdir(d)]]

trajectories = []
for dir_name in dir_names:
    file_name = os.path.join(dir_name, 'vasprun.xml')
    vr = Vasprun(file_name)
    traj = vr.get_trajectory()
    trajectories.append(traj)
    print('trajectories[{}] with length {} created from file {}.'.format(
        len(trajectories)-1, len(traj), file_name))

extended_traj = trajectories[0]
for traj in trajectories[1:]:
    extended_traj.extend(traj)

print('extended_traj of length {}.'.format(len(extended_traj)))
print('extended_traj.constant_lattice = ', extended_traj.constant_lattice)
print('type(extended_traj.lattice) = ', type(extended_traj.lattice))
print('len(extended_traj.lattice) = ', len(extended_traj.lattice))
