from pyama.mdAnalyzesPkg.md_io import create_xdatcar_condensed_copy
import os
import time

dir_name = '/data/VASP/AsTe3/Te-based_sc-483_MD/best-refined-413-supercell-001_729364/MD_900K/805966_inerrupted'

tic = time.perf_counter()
create_xdatcar_condensed_copy(dir_name, new_xdatcar_name='/home/cadarp02/tmp/XDATCAR_reduced',
                              ionic_step_offset=0, keep_one_every=100)
tac = time.perf_counter()
print('Process took {:.3f} seconds.'.format(tac-tic))

