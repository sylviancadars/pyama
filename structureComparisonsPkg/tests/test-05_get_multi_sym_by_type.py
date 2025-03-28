from pyama.structureComparisonsPkg.atomicEnvComparisons import AtomEnvComparisonsData
from ase.io import read

systems = []
atoms_1 = read('TeO2-alpha_P41212_1968_COD-1537586.cif')
atoms_1.rattle(stdev=0.1)
systems.append(atoms_1)
systems.append(read('TeO2-gamma_P212121_2000_COD-1520934.cif'))


aecd = AtomEnvComparisonsData(verbosity=2)
multisyst_sim_by_type = aecd.get_multisystem_similary_maps_by_type(systems)

fig, ax = aecd.plot_mutisyst_similarity_maps(multisyst_sim_by_type)

