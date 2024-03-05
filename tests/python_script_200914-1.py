from pymatgen import Lattice, Structure
from pymatgen.ext.cod import COD

formula = 'TeO2'
codStructuresDict = COD.get_structure_by_formula(formula)
print(codStructuresDict)

"""
# Load structure from CIF file
InputFileName = "D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif"
structure = Structure.from_file(InputFileName)
# Setting oxidation states to default
structure.add_oxidation_state_by_guess()
# TO DO : Setting user-defined oxidation states (e.g. with add_oxidation_state_by_site method)
"""
