from structureComparisonsPkg.distanceTools import distanceMatrixData
from structureComparisonsPkg.distanceTools import get_distance_matrix_from_soap_rematchkernel
from pymatgen.core.structure import Structure
from mp_api.client import MPRester

import matplotlib.pyplot as plt
import os
import numpy as np

mp_id = 'mp-2895'
nb_of_structures = 10

mprester = MPRester(os.environ.get('MP_API_KEY'))

dmd = distanceMatrixData()

ref_structure = mprester.get_structure_by_material_id(mp_id) # Mg2SiO4, 28 atoms
structures = [ref_structure]

perturbations = np.linspace(0.05, 2.0, nb_of_structures - 1)
names = ['ref'] + ['{:.2f}-A perturbation'.format(perturbation)
                   for perturbation in perturbations]

for i, perturbation in enumerate(perturbations):
    perturbed_structure = ref_structure.copy()
    perturbed_structure.perturb(perturbation)
    structures.append(perturbed_structure)


dmd.calculate_distance_matrix(structures)
dm_0 = dmd.Dmatrix
fig_0, ax_0 = dmd.plot_distance_matrix(tickLabels=names)
ax_0.set_title('{} Valle-Oganov cosine distance'.format(
    ref_structure.composition.reduced_formula))

dm_1, fig_1, ax_1 = get_distance_matrix_from_soap_rematchkernel(structures, show_plot=True,
    structure_names=names)

ax_1.set_title('{} REMatch-SOAP distance vs random perturbation'.format(
    ref_structure.composition.reduced_formula))
fig_2, ax_2 = plt.subplots()
ax_2.plot(dm_0.flatten(), dm_1.flatten(), 'ko')
ax_2.set(title='Valle-Oganov vs average-soap cosine distances',
         xlabel='Valle and Oganov cosine distance',
         ylabel='REMatch-SOAP distance')

plt.show()

