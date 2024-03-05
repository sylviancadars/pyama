from pyama.uspexAnalysesPkg.uspexDataManipulation import uspexStructuresData
from matplotlib import pyplot as plt


usd = uspexStructuresData(fileOrDirectoryName='/data/USPEX/SiCN/Si12C10N10H8_v10.4/results1/',
                          verbosity=3)
usd.get_distance(usd.IDs[0], usd.IDs[-1])
selected_ids, seed, fig, ax = usd.select_distinct_lowest_energy_structures(5, max_calc_per_structure=5)

plt.show()

