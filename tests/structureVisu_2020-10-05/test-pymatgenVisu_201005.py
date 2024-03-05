
#**************** FUNCTIONS ****************


#************** MAIN SECTION ***************

#********************* TeO2-alpha ***********************
inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif'
#********************* beta-As2Te3 ***********************
#inputFileName = r'D:\cadarp02\Documents\CristalStructures\beta-As2Te3_2015_InorgChem-ic5b01676.cif'

from pymatgen import Structure
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.vis.structure_chemview import quick_view

structure = Structure.from_file(inputFileName)
# structure.make_supercell([2,1,1],True)

structVis = StructureVis()
structVis.set_structure(structure, reset_camera=True, to_unit_cell=True)
structVis.show()

# The command below uses chemview, which does not seem to be available for Windows
chemViewObj = quick_view(structure=structure)
