from pymatgen.core.structure import Structure

from pyama.diffractionPkg.nanopdf import nanopdfData
from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData

import matplotlib.pyplot as plt

file_name = '/home/cadarp02/CrystalStructures/As2Te3/As2Te3-beta_R-3mH_2015_icsd-196147.cif'


sigma=0.02
nd = nanopdfData(R_max=12, fNy = 10, sigma=sigma, verbosity=2)
structure = Structure.from_file(file_name)
dmd = distanceMatrixData(R=nd.R, sigma=nd.sigma)
partials, types = dmd.calculate_all_reduced_partial_RDFs(structure)
# Define a reference structure (here the same) to set typenames and typeindexes
nd.set_reference_structure(structure)
nd.calculate_final_exact_pdf_from_partials(partials, types)
fig_1, ax_1 = nd.plot_partials()
ax_1.set_title('Total pdf calculated from partials')

nd = nanopdfData(R_max=12, fNy = 10, sigma=sigma, verbosity=2)
nd.calculate_final_exact_pdf_from_structure(structure)
fig_2, ax_2 = nd.plot_partials()
ax_2.set_title('Total pdf and partials calculated from structure')

plt.show()


