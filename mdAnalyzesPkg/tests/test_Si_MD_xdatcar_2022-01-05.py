from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData
from matplotlib import pyplot as plt

tad = TrajAnalyzesData(description='Si continuation test from XDATCAR files')
tad.set_vasp_data_dir_from_root_dir('/home/cadarp02/python/pyama/examples/Si_MD_VASP_multirun', 'XDATCAR')
tad.keep_one_every = 5
tad.set_trajectory_from_vasp(use_xdatcar_file=True)
fig, ax = tad.plot_thermodynamics_and_displacements()
plt.show()


