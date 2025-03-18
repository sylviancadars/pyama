from pymatgen.core.lattice import Lattice
from pyama.utils import cyclic_permutations, get_lattice_system_with_unique_param

cell_lengths_and_angles = (5.0, 5.0, 5.0, 90., 90., 100.)
print("cell_lengths_and_angles = ", cell_lengths_and_angles)
lattice_system_dict = get_lattice_system_with_unique_param(
    cell_lengths_and_angles, return_as_dict=True, verbosity=2)

print('lattice_system_dict = ', lattice_system_dict)

original_lattice = Lattice.from_parameters(*cell_lengths_and_angles)
new_lattice =

