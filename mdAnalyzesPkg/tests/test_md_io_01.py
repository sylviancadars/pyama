from pyama.mdAnalyzesPkg.md_io import get_trajectory_from_vasp_run

dir_name='/data/VASP/AsTe3/MD_1031912-sc212/500K_SMASS-0.14/407176'

result = get_trajectory_from_vasp_run(dir_name, verbosity=2, ionic_step_skip=50)

"""
print(result)
"""
print(type(result[0]))
print(type(result[0][0]))
traj = result[0]
traj.to_positions()
print('After applying to_positions() method:\ntype(traj[0]) = ', type(traj[0]))

