#!/usr/local/miniconda/envs/aiida/bin/python3

from pyama.cpmdTools.utils import CPMDData
# from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData

# import numpy as np
import os
import pandas as pd
import click
import sys
import matplotlib.pyplot as plt


# Set label of the group containing the structures on which analyses should be
# performed.
@click.command('cli')
@click.option('-v', '--verbosity', default=1, type=int,
              help='Verbosity level. Default is 1')
@click.option('-d', '--dir_name', default='.', type=str,
              help='Directory name. Default is .')
def cli(verbosity, dir_name):
    """
    Program for the generation of a ceramic network.
    """
    verbosity_error_msg = 'Verbosity should be a positive integer'
    if not isinstance(verbosity, int):
        sys.exit(verbosity_error_msg)
    elif verbosity < 0:
        sys.exit(verbosity_error_msg)

    if verbosity >= 1:
        print('Running script {} in directory {}'.format(
              os.path.basename(__file__), os.path.abspath(dir_name)))

    main(verbosity, dir_name)


def main(verbosity, dir_name):
    cpmdd = CPMDData(dir_name, verbosity=verbosity,
                     system_description=dir_name, r_max=12, sigma=0.005,
                     exr_pdf_fNy=10)
    cpmdd.set_simul_time(initial_simul_time=11500)
    cpmdd.parse_output_file('output.out0')
    print('Unit cell parameters (in \u212b):\n',
          cpmdd.initial_structure.lattice.lengths,
          cpmdd.initial_structure.lattice.angles)

    print(cpmdd.energies_df[0:20])
    # Pick evenly-spaced ionic iterations
    picked_ionic_iterations = cpmdd.pick_evenly_spaced_frames(
        max_nb_of_frames=10, time_interval=None, ionic_iter_interval=5000)
    print('picked_ionic_iterations = ', picked_ionic_iterations)

    # Extract structures associated with picked ionc_iterations
    cpmdd.parse_trajec_xyz_file('TRAJEC.xyz',
        select_by_ionic_iteration=picked_ionic_iterations)

    # Plot
    # cpmdd.set_exact_xray_pdf_parameters(r_max=12.0, sigma=0.0001)
    # fig4, ax4 = cpmdd.plot_multi_frames_pdf()

    traj_indexes = cpmdd.get_traj_indexes_from_ionic_iterations(picked_ionic_iterations)
    partials, types = cpmdd.get_averaged_partials(traj_indexes)
    pdf, r = cpmdd.get_exact_xrd_pdf_from_partials(partials, types,
        export_to_file='averaged_pdf.txt', include_partials=True)

    """
    fig1, ax1 = cpmdd.plot_structure_evolution(reference_struct_or_index=0,
                                               initial_index=0, step_subdiv=1,
                                               r_max=8, sigma=0.005)
    """
    fig2, ax2 = cpmdd.plot_all_energies()

    fig3, ax3 = plt.subplots()
    ax3.plot(r, pdf)
    ax3.set(title='Averaged pdf')

    export_file_name = cpmdd.export_selected_configurations(traj_indexes,
                                                            'exported_configs.xyz')

    plt.show()

if __name__ == '__main__':
    cli()


