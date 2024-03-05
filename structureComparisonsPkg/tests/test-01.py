from structureComparisonsPkg.distanceTools import distanceMatrixData
from pymatgen.core.structure import Structure
from pyama.nanopdf import nanopdf
import matplotlib.pyplot as plt

dmd = distanceMatrixData()

file_name = '/home/cadarp02/CrystalStructures/As2Te3/As2Te3-beta_R-3mH_2015_icsd-196147.cif'
struct = Structure.from_file(file_name)

"""
dmd.calculate_partial_RDF(struct,'As','Te', showPlot=True)
dmd.calculate_partial_RDF(struct,'As','As', showPlot=True)
dmd.calculate_partial_RDF(struct,'Te','Te', showPlot=True)
"""

"""
dmd.calculate_all_partial_RDFs(struct, showPlot=True)

(r, pdf) = nanopdf.calculate_final_exact_pdf(struct)
fig = nanopdf.plot_exact_pdf(r, pdf)
plt.show()
"""

struct1 = struct.copy()
struct1.perturb(0.1)
distance = dmd.calculate_cosine_distance(struct, struct1)
print('distance between structure and its perturbation = ', distance)

distance = dmd.calculate_cosine_distance_old(struct, struct1)
print('distance calculated with old method = ', distance)

