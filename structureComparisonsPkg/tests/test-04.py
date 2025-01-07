from structureComparisonsPkg.distanceTools import distanceMatrixData
from structureComparisonsPkg.atomicEnvComparisons import AtomEnvComparisonsData
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester

import matplotlib.pyplot as plt
import os
import numpy as np

mp_id = 'mp-2895'

mprester = MPRester(os.environ.get('MP_API_KEY'))

dmd = distanceMatrixData()

struct_1 = mprester.get_structure_by_material_id(mp_id) # Mg2SiO4, 28 atoms
struct_2 = struct_1.copy()
struct_2.perturb(0.02)

distance = dmd.calculate_cosine_distance(struct_1, struct_2)
print('distance = ', distance)

"""
aaa = AseAtomsAdaptor
atoms_1 = aaa.get_structure(struct_1)
atoms_2 = aaa.get_structure(struct_2)
"""

aecd = AtomEnvComparisonsData(verbosity=2)
full_sim = aecd.get_similary_maps_by_type(struct_1, struct_2)
fig, ax = aecd.plot_similarity_maps(full_sim)

print('full_sim = ', full_sim)

similarities = aecd.get_similarities_between_matching_sites(struct_1, struct_2)

print('similarities = ', similarities)

plt.show()

