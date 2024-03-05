from pyama.utils import visualize_structure
from pymatgen.core.structure import IStructure

file_name='/home/cadarp02/CrystalStructures/As2Te3/As2Te3-alpha_2022.cif'
visualize_structure(file_name, viewer='ase')

struct = IStructure.from_file(file_name)
visualize_structure(struct, viewer='vesta')

