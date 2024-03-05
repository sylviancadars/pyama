from matplotlib import pyplot as plt
from pyama.mdAnalyzesPkg.utils import TrajAnalyzesData
import os

"""
/data/VASP/AsTe3/Te-based_sc-181_2022-06-01_MD/best-refined-supercell-001_726177
(aiida-2.0) cadarp02@escs-215-02:/data/VASP/AsTe3/Te-based_sc-181_2022-06-01_MD/best-refined-supercell-001_726177$ ls
1846014  428875  790103
"""

root_dir = os.getcwd()
dir_names=[os.path.join(root_dir, dir_name) for dir_name in
           [d for d in os.listdir(root_dir) if os.path.isdir(d)]]
# print(dir_names)

tad = TrajAnalyzesData(vasp_data_dir=dir_names, keep_one_every=1,
                       initial_simul_time=0, verbosity=2)
tad.set_trajectory_from_vasp()
fig0, ax0 = tad.plot_thermodynamics_and_displacements()

r, pdf, fig1, ax1 = tad.get_single_frame_pdf(50, show_plot=True)

traj_indexes = tad.pick_evenly_spaced_traj_indexes(max_nb_of_frames=10)
pdf_data, fig2, ax2 = tad.get_multi_frames_pdf(traj_indexes, show_plot=True)


plt.show()

print(tad.vasp_data_dir)

