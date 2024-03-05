from pyama.mdAnalyzesPkg.md_io import get_frac_coords_and_velocities_from_xdatcar, \
    export_poscar_file_with_velocity
import os

# dir_name = '../../examples/Si_MD_VASP_multirun/622216'
dir_name='/data/VASP/AsTe3/Te-based_sc-483_MD/best-refined-413-supercell-001_729364/MD_900K/805966_inerrupted/'

frac_coords, velocities = get_frac_coords_and_velocities_from_xdatcar(dir_name,
    99, verbosity=2)

poscar_file = '/home/cadarp02/tmp/POSCAR'
export_poscar_file_with_velocity(os.path.join(dir_name, 'XDATCAR'), 99,
                                 poscar_file_name=poscar_file)

