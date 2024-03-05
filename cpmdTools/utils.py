#!/usr/local/miniconda/envs/aiida/bin/python3

from pymatgen.core.units import LengthArray
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import Site
from pymatgen.core.structure import Structure
from pymatgen.io.xyz import XYZ

from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData
from pyama.diffractionPkg.nanopdf import nanopdfData
from pyama.utils import visualize_structure

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import time

def is_number(s):
    try:
        float(s)  # for int, long, float and complex
    except ValueError:
        return False

    return True


class CPMDData():
    """
    ADD DESCRIPTION
    """
    def __init__(self, dir_name, system_description='',
                 verbosity=1, print_to_console=True, print_to_file=False,
                 sigma=0.01, r_max=10, seed=None, print_performance=False,
                 exr_pdf_wavelength=0.709318, exr_pdf_Qmax=17.0,
                 exr_pdf_fNy=5):
        """
        Initialization
        """
        self.dir_name = dir_name
        self.verbosity = verbosity
        if isinstance(seed, int):
            self.seed = seed
        else:
            self.seed = np.random.randint(10000)
        # Intialize numpy random number generator (RNG) to ensure reproducibility
        if self.verbosity >= 1:
            print(('Initialization of random number generator with seed: {}'
                   ).format(self.seed))
        self.rng = np.random.default_rng(self.seed)
        self.print_to_console = print_to_console
        self.print_to_file = print_to_file
        self._output_text = []   # list of strings to be ultimately written in file
        self.energies_short_names = ['NFI', 'EKINC', 'TEMPP', 'EKS',
                                     'ECLASSIC', 'EHAM', 'DIS', 'TCPU']
        self.energies_long_names = ['Frame number',
                                    'Kinetic energy of electrons (a.u.)',
                                    'Ions temperature (K)',
                                    'Kohn-Sham energy (a.u.)',
                                    'Classic energy (a.u.)',
                                    'Hamioltonian energy (a.u.)',
                                    'MSD of ions wrt init. pos.(a.u.^2)',
                                    'Time step CPU time (s)']
        self.energies_units = ['', 'a.u.', 'K', 'a.u.', 'a.u.', 'a.u.', 'a.u.'
                               's']
        self.initial_simul_time = 0
        self.read_energies_file(dir_name)
        self.set_time_step(5)
        self.system_description = system_description
        self.initial_structure = None
        self.trajectory = None
        self.sigma = sigma  # Gaussian smearing factor in Angstroms for pdf
        self.r_max = r_max  # max radius in Agstroms for pdf
        self.r = None
        # Set exact x-ray pdf parameters to default
        self.set_exact_xray_pdf_parameters(r_max=self.r_max, sigma=self.sigma,
            wavelength=exr_pdf_wavelength, Qmax=exr_pdf_Qmax, fNy=exr_pdf_fNy)
        self.print_performance = print_performance

    def __repr__(self):
        """
        Printable representation
        """
        temp = vars(self)
        mystr = ''
        for item in temp:
            mystr += '{}: {}\n'.format(item, temp[item])

        return mystr

    def print(self, string_to_print, verb_th=1):
        """
        print to screen and/or file given verbosity threshold
        """
        if self.verbosity >= verb_th:
            if self.print_to_console:
                print(string_to_print)
            if self.print_to_file:
                self._output_text.append(string_to_print)
            # TODO: add a possibility to print/write a list line-by-line

    def get_energies_long_name(self, short_name_or_index):
        if isinstance(short_name_or_index, int):
            return self.energies_long_names[short_name_or_index]
        elif isinstance(short_name_or_index, str):
            return self.energies_long_names[self.energies_short_names.index(
                short_name_or_index)]

    def read_energies_file(self, energies_file_or_dir_name=None):
        """
        Read CPMD ENERGIES file as a pandas dataframe
        """
        if energies_file_or_dir_name is None:
            energies_file_name = os.path.join(os.getcwd(), 'ENERGIES')
        elif os.path.isdir(energies_file_or_dir_name):
            energies_file_name = os.path.join(energies_file_or_dir_name,
                                              'ENERGIES')
        elif os.path.isfile(energies_file_or_dir_name):
            energies_file_name = os.path.abspath(energies_file_name)

        with open(energies_file_name, 'r') as file:
            data = file.read().split('\n')
        # Separate columns. "if i.split()" removes empty lines
        energies_data = []
        for line in data:
            cols = line.split()
            if cols:
                energies_data.append([int(cols[0])] +
                                     [float(col) for col in cols[1:]])

        # energies_data = [i.split() for i in data if i.split()]
        self.energies_df = pd.DataFrame(energies_data,
                                        columns=self.energies_short_names)
        self.energies_df['NFI'].astype(int)

    def parse_output_file(self, output_file_name):
        """
        Read information from output file
        """
        if os.path.split(output_file_name)[0] == '':
            output_file_name = os.path.join(self.dir_name,
                                            output_file_name)

        # TODO: add exeption if file cannot be opened ?
        with open(output_file_name, 'r') as file:
            lines = file.read().split('\n')

        # Initializations
        cell = LengthArray(np.zeros((3, 3)), 'ang')
        sites = []

        is_cell_parsed = False
        is_atoms_parsed = False
        read_atoms = False
        is_ntraj_parsed = False  # TRAJEC.xyz IS SAVED ON FILE EVERY <NTRAJ>
        # Parse cell vectors
        for line_index, line in enumerate(lines):
            if not is_ntraj_parsed:
                patterns = ['TRAJEC.xyz IS SAVED ON FILE EVERY',
                    'TRAJECTORIES ARE SAVED ON FILE EVERY']
                for pattern in patterns:
                    if pattern in line:
                        ntraj = lines[line_index][len(pattern)+1:].split()[0]
                        self.traj_extracted_every = int(ntraj)
                        self.print(('ntraj (struct in trajectory every) = {}'
                                    ).format(self.traj_extracted_every),
                                   verb_th=2)
                        is_ntraj_parsed = True

            # Parse cell vectors
            if not is_cell_parsed:
                for i in range(3):
                    pattern = 'LATTICE VECTOR A'+str(i+1)
                    if pattern in line:
                        unit = line[len(pattern):].split('(')[-1].split(')')[0].lower()
                        self.print('Lattive vectors unit: {}'.format(unit),
                                   verb_th=3)
                        cell[i] = LengthArray([float(s) for s in
                                               line[len(pattern):].split() if
                                               is_number(s)], unit).to('ang')
                        self.print('Cell vector {} = {}'.format(i, cell[i]),
                                   verb_th=2)
                        if i >= 2:
                            is_cell_parsed = True

            if not is_atoms_parsed:
                # Parse atoms
                pattern = '* ATOMS *'
                if pattern in line:
                    sites = []
                    read_atoms = True
                    line_counter = 0
                    self.print('Reading ATOMS section from output file.',
                               verb_th=2)
                    continue
                if read_atoms and line_counter == 0:
                    unit = line.split('(')[-1].split(')')[0].lower()
                    self.print('Atomic coordinates units: {}'.format(unit),
                               verb_th=3)
                    line_counter += 1
                    continue
                if read_atoms and line_counter > 0:
                    if '************************' in line:
                        self.print('End of ATOMS section reached.', verb_th=2)
                        read_atoms = False
                        is_atoms_parsed = True
                    else:
                        _ = line.split()
                        sites.append(Site(_[1],
                                     LengthArray([float(i) for i in _[2:5]],
                                                 unit).to('ang')))
                        self.print('Adding site {}: {}'.format(_[0], sites[-1]),
                                   verb_th=3)
                        line_counter += 1

        self.print('Cell = {}'.format(cell), verb_th=1)
        # TODO: define lattice from cell
        self.initial_structure = Structure(Lattice(cell),
                                           [site.species for site in sites],
                                           [site.coords for site in sites],
                                           coords_are_cartesian=True)
        self.print('Initial struture read from output file:\n{}'.format(
                   self.initial_structure), verb_th=2)

        # Add 'has_extracted_truct' column to energies_df
        self.set_has_extracted_struct()

    def get_first_ionic_iteration(self, trajectory_file_name=None):
        """
        Retrieve the first extracted ionic iteration number from TRAJEC.xyz or TRAJECTORY file
        """

        if trajectory_file_name is None:
            trajectory_file_name = os.path.join(self.dir_name, 'TRAJEC.xyz')
        elif os.path.split(trajectory_file_name)[0] == '':
            trajectory_file_name = os.path.join(self.dir_name,
                                                trajectory_file_name)
        if os.path.basename(trajectory_file_name) == 'TRAJEC.xyz':
            line_to_read = 1
        elif os.path.basename(trajectory_file_name) == 'TRAJECTORY':
            line_to_read = 0
        elif os.path.basename(trajectory_file_name) == 'TMP':
            line_to_read = 0
        else:
            self.print(('Method get_first_ionic_iteration requires TRAJEC.xyz '
                        'or TRAJECTORY file name'), verb_th=2)
            return None
        with open(trajectory_file_name) as myfile:
            try:
                line = [next(myfile) for line_index in range(line_to_read+1)][-1].rstrip()
            except StopIteration as e:
                self.print(('StopIteration error in get_first_ionic_iteration. '
                            'File {} might be empty.').format(trajectory_file_name))
                return None
        if 'STEP:' in line:
            first_ionic_iteration = int(line.split()[-1])
        else:
            first_ionic_iteration = int(line.split()[0])
        self.print(('First extracted ionic iteration number obtained from '
                    'file {}: {}.').format(trajectory_file_name,
                                           first_ionic_iteration))

    def set_has_extracted_struct(self):
        """
        Add a column 'has_extracted_struct' in energies_df

        Based on traj_extracted_every property and first NFI (ionic iteration
        or frame index) value in ENNERGIES file.
        """
        self.print('Running function set_has_extracted_struct', verb_th=3)

        # Add a 'has_extracted_struct' column to energies_df and initialize
        values = [frame_index % self.traj_extracted_every == 1
                  for frame_index in self.energies_df['NFI'].tolist()]
        self.energies_df['has_extracted_struct']=values

    def parse_trajec_xyz_file(self, trajectory_file_name=None,
                              append_trajectory=False,
                              include_energy_data=False,
                              extract_every=1, select_by_index_in_file=None,
                              select_by_ionic_iteration=None,
                              initial_simulation_time=None):
        """
        Extract sequence of pymatgen structures from TRAJEC.xyz file

        Currently works for fixed-cell trajectories (lattice read from
        initial_structure)

        Args:
            trajectory_file_name:
            extract_every:
                extract a structure only every extract_every ionic steps
                (1-based indexing). Only used if select_by_index_in_file and
                select_by_cmpd_index are None.
            select_by_index_in_file: int, list, tuple or None (default is None)
                if None, extract_every will be used.
                warning: 1-based ionic iteration indexes
            select_by_ionic_iteration: int, list, tuple or None (default is None)
                list of cmpd indexes as they appear in ENERGIES or in the
                comment lines of the TRAJEC.xyz file.

        Returns:
            A list of dicts of the form [{'ionic_iteration': ...,
            'structure': ...}, ....]
        """
        if trajectory_file_name is None:
            trajectory_file_name = os.path.join(self.dir_name, 'TRAJEC.xyz')
        elif os.path.split(trajectory_file_name)[0] == '':
            trajectory_file_name = os.path.join(self.dir_name,
                                                trajectory_file_name)

        # Initialize has_extracted_struct column in energies_df if not present
        if 'has_extracted_struct' not in self.energies_df.columns:
            self.set_has_extracted_struct()

        if initial_simulation_time is not None:
            self.set_simul_time(initial_simulation_time)

        # Convert single-element index selections in lists
        if isinstance(select_by_ionic_iteration, int):
            select_by_ionic_iteration = [select_by_ionic_iteration]
        if isinstance(select_by_index_in_file, int):
            select_by_index_in_file = [select_by_index_in_file]

        nb_of_atoms = None
        if self.initial_structure is not None:
            nb_of_atoms = self.initial_structure.num_sites

        with open(trajectory_file_name, 'r') as file:
            lines = file.read().split('\n')

        stored_index = 0
        traj = []

        nb_of_atoms = int(lines[0].split()[0])
        if self.initial_structure is not None:
            if nb_of_atoms != self.initial_structure.num_sites:
                self.print(('WARNING: nb of atoms ({}) in TRAJEC.xyz file is '
                            'different from nb of sites in initial_structure: '
                            '{}').format(nb_of_atoms,
                                         self.initial_structure.num_sites))
        line_index = 0
        is_simul_time_positive = False
        cur_energies_row_index=0
        while line_index < len(lines):
            # Parse lines by blocks of nb_of_atoms+2 lines
            curr_iter_lines = lines[line_index:(line_index+nb_of_atoms+2)]
            if len(curr_iter_lines) < nb_of_atoms+2 and (len(curr_iter_lines)
                                                         > 0):
                self.print(('WARNING: incomplete xyz detected after {} ionic '
                           'iterations in TRAJEC.xyz file.').format(
                           stored_index))
                break
            stored_index += 1  # 1-based index
            ionic_iteration = int(curr_iter_lines[1].split()[1])

            """
            while cur_energies_row_index < self.energies_df.shape[0]:
                if (self.energies_df['NFI'][cur_energies_row_index] <
                    ionic_iteration):
                    cur_energies_row_index += 1
                elif (self.energies_df['NFI'][cur_energies_row_index] ==
                      ionic_iteration):
                    self.energies_df['has_extracted_struct'][
                        cur_energies_row_index] = True
                    self.print('Row {} of energies_df set to: {}'.format(
                               cur_energies_row_index, self.energies_df.iloc[
                                   cur_energies_row_index,:].tolist()),
                               verb_th=3)
                    break
                else:
                    break
                    self.print(('WARNING: cur_energies_row_index ({}) > '
                                'ionic_iteration ({})').format(
                                    cur_energies_row_index, ionic_iteration))
            """

            # Set is_simul_time_positive as soon as initial_simulation_time is reached
            if not is_simul_time_positive:
                simul_time = self.get_simul_times_from_ionic_iterations(
                    ionic_iteration, convert_to_single=True)
                if simul_time >= 0:
                    is_simul_time_positive = True
                self.print('ionic_iteration {}, simulation time {:.4f} ps.'.format(
                           ionic_iteration, 0.001*simul_time), verb_th=3)
            if is_simul_time_positive and (
               (
                    isinstance(select_by_index_in_file, (list, tuple)) and
                    stored_index in select_by_index_in_file
               ) or (
                    isinstance(select_by_ionic_iteration, (list, tuple)) and
                    ionic_iteration in select_by_ionic_iteration
               ) or (
                    select_by_index_in_file is None and
                    select_by_ionic_iteration is None and
                    (stored_index-1) % extract_every == 0)):

                self.print(('Reading structure from iteration {} in TRAJEC.xyz'
                            ' file (CPMD step index {}, simul. time {:.3f} ns.)'
                            ).format(stored_index, ionic_iteration,
                                     1000*simul_time), verb_th=3)
                struct = self.initial_structure.copy()
                for site_index in range(nb_of_atoms):
                    struct.sites[site_index].coords = [float(i) for i in
                        curr_iter_lines[site_index+2].split()[1:]]
                traj.append({'stored_index': stored_index,
                             'ionic_iteration': ionic_iteration,
                             'structure': struct})

            line_index += nb_of_atoms + 2

        if self.trajectory is None or not append_trajectory:
            self.trajectory = traj
            self.print(('Trajectory of {} ionic steps stored in trajectory '
                        'property.').format(len(traj)))
        elif append_trajectory:
            self.trajectory += traj
            self.print(('Trajectory of {} ionic steps appended to trajectory '
                       'property, leading to a total of {} steps.').format(
                        len(traj), len(self.trajectory)))


    def set_time_step(self, time_step, unit='au'):
        """
        TODO: read time step from input file
        """
        if unit.lower() in ['au', 'a.u.']:
            self.time_step_fs = time_step * 0.0241888428
        elif unit.lower() in ['fs', 'femtoseconds', 'femtosecond']:
            self.time_step_fs = time_step
        elif unit.lower() in ['ns', 'nanoseconds', 'nanosecond']:
            self.time_step_fs = time_step * 1000

    def set_simul_time(self, initial_simul_time=None):
        if initial_simul_time is not None:
            self.initial_simul_time = initial_simul_time
        self.simul_time = self.time_step_fs * np.arange(
            self.energies_df.shape[0]) - self.initial_simul_time


    def get_simul_times_from_ionic_iterations(self, ionic_iterations,
                                               convert_to_single=True):
        """
        calculate an array of (or a single) simulation time(s) from step numbers
        """
        if isinstance(ionic_iterations, int):
            ionic_iterations = [ionic_iterations]
        simul_times = np.asarray(
            [self.time_step_fs*(i-self.energies_df.loc[0, 'NFI']) -
             self.initial_simul_time for i in ionic_iterations])
        if len(simul_times) == 1 and convert_to_single:
            simul_times = float(simul_times)
        return np.asarray(simul_times)


    def plot_all_energies(self, initial_simul_time=None, step_subdiv=10):
        """
        Plot data in ENERGIES file.

        Args:
            initial_frame_index: int or None
                Start plot from a certain index (0-based indexing in ENERGIES
                file)
                step_subdiv: int (Default: 10)
                plot data points every step_subdiv steps.
        """
        self.set_simul_time(initial_simul_time)

        # Create Plot
        nrows = 2
        ncols = 2
        fig, axes = plt.subplots(nrows, ncols, sharex=True)

        # find initial_frame_index
        initial_frame_index = 0
        while self.simul_time[initial_frame_index] < 0:
            initial_frame_index += 1

        x = np.asarray(self.simul_time[initial_frame_index::step_subdiv])
        self.print('x = {}...{} ({} elements)'.format(x[0], x[-1], len(x)),
                   verb_th=2)

        indexes = np.arange(initial_frame_index, self.energies_df.shape[0],
                            step_subdiv)
        data = self.energies_df.iloc[indexes, :]

        def make_plot(data_names, plot_index):
            if isinstance(data_names, str):
                data_names = [data_names]
            for data_name in data_names:
                y = data.loc[:, data_name].to_numpy()
                axes[plot_index // ncols, plot_index % ncols].plot(x, y)

            if len(data_names) == 1:
                axes[plot_index // ncols, plot_index % ncols].set_ylabel(
                    self.get_energies_long_name(data_name))
            elif len(data_names) > 1:
                axes[plot_index // ncols, plot_index % ncols].legend(
                    [self.get_energies_long_name(n) for n in data_names])

        make_plot('TEMPP', 0)
        make_plot('EKINC', 1)
        make_plot(['EKS', 'ECLASSIC', 'EHAM'], 2)
        make_plot('DIS', 3)

        for ncol in range(ncols):
            axes[nrows-1, ncol].set_xlabel('Simulation time (fs)')

        axes[0, 0].set_title(self.system_description)

        return fig, axes

    def plot_structure_evolution(self, reference_struct_or_index=None,
                                 initial_index=0,
                                 use_trajectory_index=True,
                                 use_stored_index=True,
                                 use_ionic_iteration=False,
                                 step_subdiv=1, r_max=10.0, sigma = 0.005,
                                 show_all_partials=False,
                                 visualize_structures=False):
        """
        Plot the evolution of the distance between structures and a reference

        The function uses the cosine distance between structures as defined in
        Oganov, Artem R., et Mario Valle, J. Chem. Phys. 2009, 130 (10), 104504,
        DOI : 10.1063/1.3079326.

        Args:
            - show_all_partials: bool (default is False)
                Only for debugging purposes. The program will stop until user
                closes figures

        Returns:
            figure, axes
        """
        if self.trajectory is None:
            sys.exit(('Error in plot_structure_evolution: first set a '
                      'trajectory with parse_trajec_xys_file.'))

        if isinstance(reference_struct_or_index, Structure):
            ref_structure = reference_struct_or_index
            init_struct_index = 0
        elif isinstance(reference_struct_or_index, int):
            if use_trajectory_index:
                ref_struct_index = reference_struct_or_index
                init_struct_index = initial_index
            else:
                if use_ionic_iteration:
                    index_name = 'ionic_iteration'
                elif use_stored_index:
                    index_name = 'stored_index'
                ref_struct_index = [frame[index_name] for frame in
                                    self.trajectory].index(
                                        reference_struct_or_index)
                init_struct_index = [frame[index_name] for frame in
                                     self.trajectory].index(initial_index)
            ref_structure = self.trajectory[ref_struct_index]['structure']
            self.print(('Using reference structure with index {} (0-based) in '
                        'self.trajectory property: ionic iteration {}, '
                        'stored_index {}, simulation time: {:.4f} ps.'
                        ).format(ref_struct_index,
                       self.trajectory[ref_struct_index]['ionic_iteration'],
                       self.trajectory[ref_struct_index]['stored_index'],
                       self.get_simul_times_from_ionic_iterations(
                           self.trajectory[ref_struct_index]['ionic_iteration'])),
                       verb_th=2)

        dmd = distanceMatrixData(Rmax=r_max, sigma=sigma,
                                 print_performance=self.print_performance)
        dist_data = {'ionic_iterations': [], 'stored_indexes': [],
                     'indexes_in_traj': [], 'dist': []}

        # DEBUGGING:
        self.print(('\n' + 50*'*' + '\nplot_structure_evolution: Reference ' +
                    'structure \n' + 50*'*' + '\n{}\n' + 50*'*' + '\n'
                    ).format(ref_structure), verb_th=3)
        if visualize_structures:
            visualize_structure(ref_structure)

        for index in range(init_struct_index, len(self.trajectory),
                           step_subdiv):
            self.print(('plot_structure_evolution: calculating distance '
                        'reference structure and structure index {} in '
                        'trajectory.').format(index), verb_th=2)
            self.print(('\n' + 50*'*' + '\nplot_structure_evolution: '
                        'traj structure {}\n' + 50*'*' + '\n{}\n' + 50*'*' + '\n'
                        ).format(index, self.trajectory[index]['structure']),
                       verb_th=3)
            if visualize_structures:
                visualize_structure(self.trajectory[index]['structure'])

            dist_data['dist'].append(dmd.calculate_cosine_distance(
                ref_structure, self.trajectory[index]['structure'],
                showPlot=show_all_partials))
            dist_data['ionic_iterations'].append(self.trajectory[index][
                                                  'ionic_iteration'])
            dist_data['stored_indexes'].append(self.trajectory[index][
                                                  'stored_index'])
            dist_data['indexes_in_traj'].append(index)
        dist_data['simulation_time'] = \
            self.get_simul_times_from_ionic_iterations(
                dist_data['ionic_iterations'])

        fig, axes = plt.subplots(1, 1)
        axes.plot(dist_data['simulation_time'], dist_data['dist'], 'o')
        axes.set_xlabel('Simulation time (fs)')
        axes.set_ylabel('Cosine distance to reference structure')
        axes.set_title('{}: Structure evolution'.format(self.system_description))

        return fig, axes


    def get_traj_index_from_ionic_iteration(self, ionic_iteration):
        ionic_iterations = [self.trajectory[i]['ionic_iteration'] for i in
                             range(len(self.trajectory))]
        return ionic_iterations.index(ionic_iteration)

    def get_traj_indexes_from_ionic_iterations(self, ionic_iterations,
                                               convert_to_single=False):
        if isinstance(ionic_iterations, int):
            ionic_iterations = [ionic_iterations]
        all_ionic_iterations = [self.trajectory[i]['ionic_iteration'] for i in
                                range(len(self.trajectory))]
        if self.verbosity >= 3:
            # use numpy to benefit from condensed printout
            print('all_ionic_iterations = {}'.format(
                np.asarray(all_ionic_iterations)))
        traj_indexes = [all_ionic_iterations.index(ionic_iteration) for
                        ionic_iteration in ionic_iterations]
        if convert_to_single and len(traj_indexes) == 1:
            (traj_indexes) = traj_indexes
        return traj_indexes


    def set_exact_xray_pdf_parameters(self, r_max=10.0, sigma=0.01,
                                      wavelength=0.559422, Qmax=17.0,
                                      fNy=5, verbosity=0):
        """

        wavelength:
           In angtroms
           0.709318 for Mo K_alpha_1
           0.559422 for Ag K_apha_1
           0.563813 for Ag K_apha_2
        """

        self.r_max = r_max
        self.sigma = sigma
        self.exr_pdf_wavelength = wavelength
        self.exr_pdf_Qmax = Qmax
        self.exr_pdf_fNy = fNy
        self._nanopdf_data = nanopdfData(R_max=self.r_max, sigma=self.sigma,
                                         lambda1=self.exr_pdf_wavelength,
                                         Qmax=self.exr_pdf_Qmax,
                                         fNy=self.exr_pdf_fNy,
                                         verbosity=verbosity)
        self.print('Exact X-ray pdf parameters set to:', verb_th=1)
        self.print('r_max = {} \u212B'.format(self.r_max), verb_th=1)
        self.print('sigma = {} \u212B (Gaussian smearing)'.format(self.sigma),
                   verb_th=1)
        self.print('wavelength = {} \u212B [CONFIRM UNIT]'.format(self.exr_pdf_wavelength),
                   verb_th=1)
        self.print('Qmax = {} \u212B^-1'.format(self.exr_pdf_Qmax), verb_th=1)
        self.print('fNy = {}'.format(self.exr_pdf_fNy), verb_th=1)


    def set_r_from_nanopdf(self):
        """
        Set values of radial distances r vector as defined in nanopdf program

        The internal definition of the r step value used in nanopdf avoids the
        need for interpolation in the calculation of the exact X-ray total
        scattering function as defined in Masson O. and Thomas P., Journal of
        Applied Crystallography 46 (2), 461‑65, 2013.
        https://doi.org/10.1107/S0021889812051357.

        dr = (pi/Qmax) / fNy  # Nyquist step / fNy
        r = numpy.arange(dr, R_max, dr)

        Note does not start at 0 Angstroms.
        """
        self.r = self._nanopdf_data.R
        self.r_max = self.r[-1]

    def update_r(self, r, is_ascending=True):
        """
        update array of of radial distances self.r

        r should be sorted in ascending order.

        Args:
            r: numpy array, tuple or list
            is_ascending: bool (default is false)
                Set to false if specified r values are not in ascending order

        Returns:

        """
        self.r = np.asarray(r)
        if is_ascending:
            r_max = self.r[-1]
        else:
            r_max = max(self.r)
        if self.r_max != r_max:
            self.r_max = r_max

    def plot_single_frame_pdf(self, traj_index, update_r=True):
        """
        Plot exact PDF for one frame

        TODO: offer the possibility to plot neutron rather than exact X-ray
        total scattering

        Args:
            traj_index:
                index of frame in trajectory property
            update_r:
                whether self.r should be updated based on R vector used in
                by nanopdf.
        Returns:
            fig: matplotlib figure handle
            ax: matplotlib axes handle
        """

        self.print(('Calculating exact PDF for structure index {} in '
                    'trajectory, ionic_iteration {}, stored_index {}.'
                    ).format(traj_index,
                             self.trajectory[traj_index]['ionic_iteration'],
                             self.trajectory[traj_index]['stored_index']),
                   verb_th=2)
        self._nanopdf_data.calculate_final_exact_pdf_from_structure(
            self.trajectory[traj_index]['structure'])
        (r, pdf) = (self._nanopdf_data.R, self._nanopdf_data.exactPDF)
        if update_r: # update array of of radial distances self.r
            # update array of of radial distances self.r
            self.update_r(self._nanopdf_data.R)
        fig, ax = self._nanopdf_data.plot_exact_pdf()

        return fig, ax

    def get_multi_frames_pdf(self, traj_indexes, update_r=True):
        """
        Calculate the exact x-ray total scattering independently for a selection of frames

        To obtain an averaged PDF it is preferable to extract and directly take the
        average of partial radial distribution functions.

        Args:
            traj_indexes: list or tuple
                list of indexes of frames in the stored trajectory property.

        Returns:
            pdf_data: dict consisting of
                'traj_indexes': list of indexes in trajectory property
                'ionic_iterations': list of ionic ietration numbers (NFI, i.e.
                    first column in ENERGIES file),
                 'stored_indexes': list of indexes in TRAJEC.xyz file,
                'r': vector of r values in Angstroms
                'pdfs': numpy array of shape (nb_of_traj_indexes, len(r))
        """
        if isinstance(traj_indexes, int):
            traj_indexes = [traj_indexes]
        pdf_data={'traj_indexes': [], 'ionic_iterations': [],
                  'stored_indexes': []}
        for traj_index in traj_indexes:
            self.print(('Calculating pdf for traj_index {}, ionic_iteration {}'
                        ).format(traj_index, self.trajectory[traj_index][
                            'ionic_iteration']), verb_th=2)
            self._nanopdf_data.calculate_final_exact_pdf_from_structure(
                self.trajectory[traj_index]['structure'])
            (r, pdf) = (self._nanopdf_data.R, self._nanopdf_data.exactPDF)

            pdf_data['traj_indexes'].append(traj_index)
            pdf_data['ionic_iterations'].append(self.trajectory[traj_index][
                'ionic_iteration'])
            pdf_data['stored_indexes'].append(self.trajectory[traj_index][
                'stored_index'])
            if 'pdfs' not in pdf_data.keys():
                pdf_data['pdfs'] = np.reshape(pdf, (1,-1))
            else:
                pdf_data['pdfs'] = np.append(pdf_data['pdfs'],
                                             np.reshape(pdf, (1,-1)), axis=0)
            if 'r' not in pdf_data.keys():
                pdf_data['r'] = r

        return pdf_data

    def plot_multi_frames_pdf(self, traj_indexes=None, y_shift=2.0):
        if traj_indexes is None:
            traj_indexes = list(range(len(self.trajectory)))
        self.print('Running plot_multi_frames with traj_indexes: {}'.format(
                   traj_indexes), verb_th=2)
        pdf_data = self.get_multi_frames_pdf(traj_indexes)
        # DEBUGGING:
        print('plot_multi_frames_pdf: traj_indexes = ', traj_indexes)
        print('pdf_data[\'ionic_iterations\'] = ', pdf_data['ionic_iterations'])
        fig, ax = fig, ax = plt.subplots()
        legend = []
        tot_y_shift = 0
        for i, traj_index in enumerate(traj_indexes):
            # DEBUGGING:
            print('i = {}, traj_index = {}'.format(i, traj_index))
            self.print(('Plotting pdf for traj_index {}, ionic_iteration {}'
                        ).format(traj_index, pdf_data['ionic_iterations'][i]),
                       verb_th=2)
            ax.plot(pdf_data['r'], tot_y_shift + pdf_data['pdfs'][i])
            tot_y_shift += y_shift
            legend.append('t = {:.3f} ns (ionic step {})'.format(
                0.001 * self.get_simul_times_from_ionic_iterations(
                    pdf_data['ionic_iterations'][i]),
                pdf_data['ionic_iterations'][i]))

        ax.set_title('{} - reduced total X-ray PDF'.format(
                     self.system_description))
        ax.set_xlabel('r (\u212B)')
        ax.set_ylabel('$\mathregular{g^{x-ray}(r)}$ (AU)')
        ax.legend(legend)

        return fig, ax

    def get_ionic_iterations_from_traj_indexes(self, traj_indexes,
                                               convert_to_single=True):
        """
        Get ionic iteration indexes associated with indexes in self.trajectory

        Args:
            traj_indexes: list or int
                (list of) trajectory index(es)
            convert_to_single: bool (default is True)
                whether to convert a list conatining a single ionic iteration
                to a sinle int.

        Returns:
            list or int
                single (if convert_to_single is True) or list of ionic
                iteration number(s) corresponding to traj_indexes
        """
        # Convert single traj_index to list
        if isinstance(traj_indexes, int):
            traj_indexes = [traj_indexes]
        ionic_iterations = [self.trajectory[traj_index]['ionic_iteration'] for
                            traj_index in traj_indexes]
        if convert_to_single and len(ionic_iterations) == 1:
            (ionic_iterations) = ionic_iterations
        return ionic_iterations


    def get_averaged_partials(self, traj_indexes, show_plot=False):
        """
        Get all averaged partials for a selection of frames stored in trajectory

        Args:
            indexes of frames in self.trajectory

        Returns:
            averaged_partials: numpy.darray of float
                a numpy array of dimension (len(types, len(types, len(r)) of
                partial radial distribution functions for all combinations of
                atomic types in types. The array is symmetric with respect to
                the first dimensions:
                    averaged_partials[i, j, :] = averaged_partials[j, i, :]
            types: a list of atomic types in the order used in
                averaged_partials
        """
        # Convert single trajectory index to list
        if isinstance(traj_indexes, int):
            traj_indexes = [traj_indexes]

        ionic_iterations = self.get_ionic_iterations_from_traj_indexes(
            traj_indexes)
        simul_times = self.get_simul_times_from_ionic_iterations(
            ionic_iterations)

        # Get vector of radial distances r compatible with nanopdf definition
        self.set_r_from_nanopdf()
        # Define a distanceMatrixData instance using this r vector
        dmd = distanceMatrixData(R=self.r, sigma=self.sigma)
        for i, traj_index in enumerate(traj_indexes):
            # Calculate all radial distributions functions
            self.print(('Calculating partials for configuration {} out {}: '
                        'ionic_iteration {}, t = {:.3f} ps.').format(i,
                        len(traj_indexes), ionic_iterations[i],
                        0.001*simul_times[i]), verb_th=1)
            partials, types = dmd.calculate_all_reduced_partial_RDFs(self.trajectory[
                traj_index]['structure'])
            if i == 0:
                # Add a 4th dimension corresponding to frame index
                all_partials = np.zeros(list(partials.shape) + [len(traj_indexes)])
            all_partials[:, :, :, i] = partials

        # DEBUGGING
        print('traj_indexes = ', traj_indexes)
        print('all_partials.shape = ', all_partials.shape)

        averaged_partials = np.mean(all_partials, axis=3)
        if not show_plot:
            return averaged_partials, types
        else:
            tot_y_range = 0
            y_range = 0.25*(max(partials[0, 0, :])-min(partials[0, 0, :]))
            lgd = []
            fig, ax = plt.subplots()
            for i in range(partials.shape[0]):
                for j in range(i+1):
                    plt.plot(self.r, partials[i, j, :]+tot_y_range)
                    tot_y_range += y_range
                    lgd.append('{}-{} partial'.format(types[i], types[j]))
            ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)',
                   title='Partials averaged from {} configurations.'.format(
                   len(traj_indexes)))
            ax.legend(lgd)
            return averaged_partials, types, fig, ax


    def get_exact_xrd_pdf_from_partials(self, partials, types, update_r=True,
                                        export_to_file=None,
                                        include_partials=True, show_plot=True,
                                        rel_y_shift=0.5, experiment_file=None,
                                        partials_rel_y_shift=0.0,
                                        experiment_title=None):
        """
        Get the exact X-ray total scattering from pre-calculated partials.

        The partials, r vector and should be calculated beforeahand with:
            self.set_r_from_nanopdf()
            dmd = distanceMatrixData(R=self.r, sigma=self.sigma)
            partials, types = dmd.calculate_all_RDFs(structure)

        or with self.get_averaged_partials()


        Args:
            partials: numpy darray
                3D numpy darray of shape (len(types, len(types), len(r))
            types: list
                List of types in the same ordering as in partials.
            export_to_file: str or None
                If not none,
            show_plot: bool (default is True)
                Whether to show a plot, in which case figure and axes handles
                are returned in addition to pdf and r.
            rel_y_shift: float (default is 0.5)
                Only used if include_partials. Y shift between curves in
                fraction of the pdf amplitude
            partials_rel_y_shift: float (default is 0.0)
                Only used if include_partials. Additional Y shift between
                total pdf, and experiment one the one hand and partials in the
                other.
                e.g., to overlay experiment and pdf and show all partials
                overlaid above, use rel_y_shift=0 and partials_rel_y_shift=1
            experiment_file: str or None (default is None)
                experimental g(r) as a 2-column file without header
            experiment_title: str or None  (default is None)
                Title of experimental data for legend

        Returns:
            pdf: numpy darray
            r: numpy darray
        if show_plot:
            fig: figure handle
            ax: axes handle
        """
        # Set a reference structure for the appropriate definition of types
        self._nanopdf_data.set_reference_structure(self.initial_structure)
        self._nanopdf_data.calculate_final_exact_pdf_from_partials(partials,
            types_in_partials=types)
        (r, pdf) =  (self._nanopdf_data.R, self._nanopdf_data.exactPDF)
        if update_r:
            self.update_r(r)
        if isinstance(export_to_file, str):
            self._nanopdf_data.export_exact_pdf(file_name=export_to_file,
                                                include_partials=include_partials)
            self.print('Total X-ray scattering saved to file: {}'.format(
                export_to_file), verb_th=1)
        if not show_plot:
            return pdf, r
        else:
            fig, ax = plt.subplots()
            ax.plot(r, pdf)
            lgd = ['Exact total X-ray PDF']
            y_shift = rel_y_shift*(max(pdf)-min(pdf))
            tot_y_shift = y_shift
            if experiment_file is not None:
                expt_data = np.genfromtxt(experiment_file)
                ax.plot(expt_data[:, 0], expt_data[:, 1] + y_shift, color='k')
                tot_y_shift += y_shift
                if isinstance(experiment_title, str):
                    lgd.append(experiment_title)
                else:
                    lgd.append('Experiment')
            if include_partials:
                partials_y_shift = partials_rel_y_shift*(max(pdf)-min(pdf))
                tot_y_shift += partials_y_shift
                for i in range(partials.shape[0]):
                    for j in range(i+1):
                        ax.plot(r, partials[i, j, :] + tot_y_shift)
                        tot_y_shift += y_shift
                        lgd.append('{}-{} partial'.format(types[i], types[j]))
                        ax.set_title(('{} - Total X-ray scattering and '
                                      'partials').format(
                                          self.initial_structure.formula))
            else:
                ax.set_title('{} - Total X-ray scattering'.format(
                    self.initial_structure.formula))
            ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)',
                   xlim=[0, self._nanopdf_data.R_max])
            ax.legend(lgd)
            return pdf, r, fig, ax


    def pick_random_frames(self, nb_of_frames, min_time_interval=None,
                           initial_simul_time=None):
        """
        pick_random_frames starting from initial_simul_time
        """
        if initial_simul_time is not None:
            self.set_simul_time(initial_simul_time)
        all_ionic_iterations = self.energies_df.loc[:, 'NFI'].tolist()
        ionic_iterations = [int(all_ionic_iterations[i]) for i, t in
                             enumerate(self.simul_times) if t > 0]
        picked_ionic_iterations = self.rng.choice(ionic_iterations,
            size=nb_of_frames, replace=False)
        return picked_ionic_iterations


    def pick_evenly_spaced_frames(self, max_nb_of_frames=20,
                                  time_interval=None, ionic_iter_interval=None,
                                  initial_simul_time=None):
        """
        ADD DESCRIPTION HERE

        Args:
            time_iterval: float or None (default is None)
            ionic_iter_interval: int or None (default is None)

        Returns:
            List of ionic_step_indexes

        TODO: account for TRAJECTORY SAMPLE  XYZ value

        """
        if initial_simul_time is not None:
            self.set_simul_time(initial_simul_time)
        extracted_ionic_iterations = [int(i) for i in
            self.energies_df.loc[self.energies_df['has_extracted_struct'] ,
                                 'NFI'].tolist()]
        self.print(('extracted_ionic_iterations of length {}: [{},...{}]'
                    ).format(len(extracted_ionic_iterations),
                             extracted_ionic_iterations[0],
                             extracted_ionic_iterations[-1]), verb_th=2)

        # initialize simul_time or cmpd_step_index thresholds to be
        # incremented according to intervals
        time_limit = 0
        # Initialize inex_limit to None to then initialize it only when
        # initial_simul_time has been reached
        index_limit = None

        picked_ionic_iterations = []
        simul_times = self.get_simul_times_from_ionic_iterations(
            extracted_ionic_iterations)
        for index, ionic_iteration in enumerate(extracted_ionic_iterations):
            t = simul_times[index]
            if len(picked_ionic_iterations) >= max_nb_of_frames:
                break
            if t < 0:
                continue
            if index_limit is None and t >= 0:
                index_limit = ionic_iteration
            elif time_interval is not None:
                if t >= time_limit:
                    picked_ionic_iterations.append(ionic_iteration)
                    time_limit += time_interval
            elif ionic_iter_interval is not None:
                if ionic_iteration >= index_limit:
                    picked_ionic_iterations.append(ionic_iteration)
                    index_limit += ionic_iter_interval

        if self.verbosity >= 2:
            picked_simul_times = self.get_simul_times_from_ionic_iterations(
                picked_ionic_iterations)
            self.print(('The following ionic iterations and simul. time have '
                        'been selected:'), verb_th=2)
            for (i, t) in zip(picked_ionic_iterations, picked_simul_times):
                self.print('{}: {:.3f} ps'.format(i, 0.001*t), verb_th=2)

        return picked_ionic_iterations


    def export_selected_configurations(self, traj_indexes, file_name,
                                       file_format='xyz'):

        if file_format.lower() == 'xyz':
            xyz = XYZ([self.trajectory[traj_index]['structure'] for traj_index
                       in traj_indexes])
            xyz.write_file(file_name)
            export_file_name = file_name
        else:
            self.print('Format {} not implemented.'.format(file_format),
                       verb_th=0)
            export_file_name = None

        if export_file_name is not None:
            self.print(('{} selected configurations exported to file: {}'
                        ).format(len(traj_indexes),
                                 os.path.abspath(export_file_name), verb_th=1))

        return export_file_name



