from pymatgen.core.lattice import Lattice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import glob
from pyama.utils import BaseDataClass, get_pymatgen_structure
from pyama.plotUtils import get_nb_of_subplot_rows, get_row_col_and_subplot_indexes
from pyama.mdAnalyzesPkg.md_io import get_trajectory_from_vasp_run, get_trajectory_from_xdatcar
from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData
from pyama.diffractionPkg.nanopdf import nanopdfData
import warnings

class TrajAnalyzesData(BaseDataClass):
    """
    A class to analayze trajectories from molecular dyanmics runs

    TO BE COMPLETED

    TODO: define simul_time as a numpy array (?)
          Check whether this may cause problems
    """
    def __init__(self, trajectory=None, vasp_data_dir=None , keep_one_every=1,
                 initial_simul_time=0, description='', verbosity=1,
                 print_to_console=True, print_to_file=False,
                 output_text_file='trajAnalyzesData_output.txt',
                 xray_pdf_r_max=10.0, xray_pdf_sigma=0.01,
                 xray_pdf_wavelength=0.559422, xray_pdf_q_max=17.0,
                 xray_pdf_fNy=5, use_xdatcar_file=True):
        super().__init__(verbosity=verbosity,
                         print_to_console=print_to_console,
                         print_to_file=print_to_file,
                         output_text_file=output_text_file)

        self.keep_one_every = keep_one_every
        self.initial_simul_time = initial_simul_time
        self.description = description
        self.use_xdatcar_file = use_xdatcar_file
        if trajectory is not None:
            self.trajectory = trajectory
        elif vasp_data_dir is not None:
            self.vasp_data_dir = vasp_data_dir
            self.set_trajectory_from_vasp()

        self._set_data_name_and_unit_mapping()
        self.displacements_data = None
        self.ionic_step_indexes = []
        self.original_files_info = []  # List of dict providing information on original data_dirctories,
                                       # corresponding ionic relaxation indexes therein, and
        self.set_xray_pdf_parameters(r_max=xray_pdf_r_max,
            sigma=xray_pdf_sigma, wavelength=xray_pdf_wavelength,
            q_max=xray_pdf_q_max, fNy=xray_pdf_fNy)

    def _set_data_name_and_unit_mapping(self):
        """
        Create a dictionary mapping data names and units

        The first key (short_name) refers to the data as labelled in
        the (program-dependent) parsed calculation files.

        These data may be associated with units, long_name and
        possibly label_name, a shorter name used for labels if
        long_name is too long.
        """
        self._data_name_and_unit_mapping = {
            'max_disp_norm': {
                'unit': '\u212B',
                'label_name': 'Max. displac. norm',
                'long_name': 'Norm of maximum displacement'},
            'max_cumul_disp_norm':{
                'unit': '\u212B',
                'label_name': 'Max. cumul. displac. norm',
                'long_name': 'Norm of maximum cumulated displacement'},
            'mean_cumul_disp_norm': {
                'unit': '\u212B',
                'label_name': 'Mean cumul. displac. norm',
                'long_name': 'Norm of mean cumulated displacement'},
            'RMSD': {
                'unit': '\u212B',
                'label_name': 'RMS displacement',
                'long_name': 'Root mean square displacement'},
            'MSD': {
                'unit': '\u212B^2',
                'label_name': 'MS displacement',
                'long_name': 'Mean square displacement'},
            # TODO: add thermodynamics_data mapping
            # ADD HERE ANY DATA THAT SHOULD BE ASSOCIATED WITH A UNIT AND LONG NAME
            'temperature': {
                'unit': 'K',
                'long_name': 'Temperature'},
            'e_fr_energy': {
                'unit': 'eV',
                'long_name': 'Free energy'},  # TODO: check explicit long name
            'e_0_energy': {
                'unit': 'eV',
                'long_name': 'Extrapolated energy'},  # TODO: check explicit long name
            'kinetic': {
                'unit': 'eV',
                'long_name': 'Kinetic energy'},  # TODO: check explicit long name
            'nosepot': {
                'unit': 'eV',
                'long_name': 'Nose potential energy'},  # TODO: check explicit long name
            'nosekinetic': {
                'unit': 'eV',
                'long_name': 'Nose kinetic energy'},  # TODO: check explicit long name
            'total': {
                'unit': 'eV',
                'long_name': 'Total energy'},
        }

    def get_data_name(self, data_short_name, use_label_name=False):
        data_name = data_short_name
        if data_short_name in self._data_name_and_unit_mapping.keys():
            key = 'label_name' if use_label_name else 'long_name'
            if key in self._data_name_and_unit_mapping[data_short_name].keys():
                data_name = self._data_name_and_unit_mapping[data_short_name][key]
            elif use_label_name and ('long_name' in
                                     self._data_name_and_unit_mapping[data_short_name].keys()):
                data_name = self._data_name_and_unit_mapping[data_short_name]['long_name']
        return data_name

    def get_data_unit(self, data_short_name):
        unit = 'A.U.'
        if data_short_name in self._data_name_and_unit_mapping.keys():
            if 'unit' in self._data_name_and_unit_mapping[data_short_name].keys():
                unit = self._data_name_and_unit_mapping[data_short_name]['unit']
        return unit

    def extend_thermodynamics_data(self, new_thermo_data):
        for key, val in self.thermodynamics_data.items():
            self.thermodynamics_data[key] += new_thermo_data[key]

    def sort_vasp_data_dir_by_mtime(self, reverse=False):

        if isinstance(self.vasp_data_dir, (tuple, list)):
            self.vasp_data_dir = [os.path.basename(i) for i in sorted(self.vasp_data_dir,
                                  key=os.path.getmtime, reverse=reverse)]
            self.print('vasp_data_dir property is now : {}'.format(self.vasp_data_dir))

    @staticmethod
    def get_data_dirs_from_recursive_search(root_dir, file_name, sort_by_mtime=True,
                                            reverse=False):
        """
        Recursively get all files with a certain name in a directory and sort by modif. time
        """

        file_names = glob.glob(os.path.join(os.path.abspath(root_dir), '**', file_name),
                               recursive=True)
        if sort_by_mtime:
            file_names = sorted(file_names, key=os.path.getmtime, reverse=reverse)
        return [os.path.split(file_name)[0] for file_name in file_names]

    def set_vasp_data_dir_from_root_dir(self, root_dir, file_name=None,
                                        sort_by_mtime=True, reverse=False):
        """
        Look for specified VASP output files in subfolders of root_dir and set
        vasp_data_dir property accordingly

        TO BE COMPLETED
        """
        if file_name is None:
            if self.use_xdatcar_file:
                file_name = 'XDATCAR'
            else:
                file_name = 'vasprun.xml'
        self.vasp_data_dir = self.get_data_dirs_from_recursive_search(root_dir,
            file_name, sort_by_mtime=sort_by_mtime, reverse=reverse)
        self.print(('vasp_data_dir property set to {} by a recursive search for {}'
                    ' files in root directory : {}').format(
                        self.vasp_data_dir, file_name, root_dir),
                   verb_th=1)

    def sort_vasp_data_dir_by_mtime(self, file_name='vasprun.xml', reverse=False):
        if isinstance(self.vasp_data_dir, (tuple, list)):
            self.vasp_data_dir = [os.path.split(full_name)[0] for full_name in sorted(
                [os.path.join(os.path.abspath(dir_name), file_name) for dir_name in
                 self.vasp_data_dir],
                key=os.path.getmtime,reverse=reverse)]
            self.print('vasp_data_dir sorted from modification time of files {} therein.'.format(
                       file_name), verb_th=1)

    def set_trajectory_from_vasp(self, vasp_data_dir=None, root_dir=None,
                                 extend_existing=False, sort_by_mtime=True,
                                 reverse=False, constant_lattice=True,
                                 use_xdatcar_file=None, **kwargs):
        """
        Set the trajectory property as a pymatgen Trajectory object from a VASP MD or geom opt run

        Takes the same keyword arguments as paymatgen.Vasprun
        with the exception of ionic_step_offset and ionic_step_skip:
            parse_dos (default: False), parse_eigen (default: False),
            parse_potcar_file(default: False), occu_tol (default: 1e-08),
            exception_on_bad_xml (default: False)

        Args:
            data_dir_name: str or None (default is None)
                Directory containing the VASP molecular dynamics (or geom opt) data.
                If None the vasp_data_dir propêrty will be used.

            use_xdatcar_file: bool or None (default is None)
                To set trajectory from VASP XDATCAR file instead of vasprun.xml.
                Use when the vasprun.xml file is corrupted (interrupted calculation)
                or for faster processing of heavy trajectory. XDATCAR will first be
                copied to a reduced version with only the ionic steps selected
                according to the keep_one_every property.
            constant_lattice: bool (default is True)
                Only used when use_xdatcar_file is True. Should be set to false
                for variable-cell MD runs (e.g. NPT ensemble)

        """
        if isinstance(use_xdatcar_file, bool):
            self.use_xdatcar_file = use_xdatcar_file
            self.print('VASP trajectory files will be read from XDATCAR file(s).')

        if vasp_data_dir is not None:
            self.vasp_data_dir = vasp_data_dir

        if self.vasp_data_dir is None:
            if root_dir is None:
                root_dir = os.getcwd()
            self.set_vasp_data_dir_from_root_dir(root_dir, sort_by_mtime=sort_by_mtime,                                     reverse=reverse)
        elif sort_by_mtime:
            self.sort_vasp_data_dir_by_mtime(reverse=reverse)

        print(50*'-', '\n', self.vasp_data_dir)

        if isinstance(self.vasp_data_dir, str):
            # load a single trajectory
            if self.use_xdatcar_file:
                input_file_name = os.path.join(self.vasp_data_dir, 'XDATCAR')
                if not os.path.isfile(input_file_name):
                    raise FileNotFoundError('File {} not found.'.format(input_file_name))
                (self.trajectory, ionic_step_indexes, nionic_steps,
                 self.thermodynamics_data, _) = get_trajectory_from_xdatcar(self.vasp_data_dir,
                    constant_lattice=constant_lattice, get_thermodynamics_data=True,
                    keep_one_every=self.keep_one_every, verbosity=self.verbosity)
            else:
                input_file_name = os.path.join(self.vasp_data_dir, 'vasprun.xml')
                if not os.path.isfile(input_file_name):
                    raise FileNotFoundError('File {} not found.'.format(input_file_name))
                (self.trajectory, ionic_step_indexes, nionic_steps,
                 self.thermodynamics_data, _)  = get_trajectory_from_vasp_run(
                    self.vasp_data_dir, ionic_step_skip=self.keep_one_every,
                    verbosity=self.verbosity, **kwargs)

            self.print('Setting trajectory from file {}.'.format(input_file_name),
                       verb_th=1)
            self.original_files_info = [{
                'data_dir': self.vasp_data_dir,
                'ionic_step_indexes': ionic_step_indexes,
                'nb_of_ionic_steps': nionic_steps,
            }]
            self.ionic_step_indexes = ionic_step_indexes

        elif isinstance(self.vasp_data_dir, (list, tuple)):
            # Concatenating multiple trajectories
            self.trajectory = None
            self.thermodynamics_data = {}
            ionic_step_offset = 0

            self.original_files_info = []
            for data_dir in self.vasp_data_dir:
                if self.use_xdatcar_file:
                    input_file_name = os.path.join(data_dir, 'XDATCAR')
                else:
                    input_file_name = os.path.join(data_dir, 'vasprun.xml')
                oszicar_file_name = os.path.join(data_dir, 'OSZICAR')
                if not os.path.isfile(input_file_name):
                    raise FileNotFoundError('File {} not found.'.format(input_file_name))
                if self.trajectory is None:
                    if self.use_xdatcar_file:
                        (self.trajectory, ionic_step_indexes, nionic_steps,
                        self.thermodynamics_data, _ ) = get_trajectory_from_xdatcar(data_dir,
                            constant_lattice=constant_lattice, get_thermodynamics_data=True,
                            keep_one_every=self.keep_one_every, verbosity=self.verbosity)
                    else:
                        (self.trajectory, ionic_step_indexes, nionic_steps,
                        self.thermodynamics_data, _ ) = get_trajectory_from_vasp_run(
                            data_dir, ionic_step_skip=self.keep_one_every, **kwargs)
                    self.print('Inititializing trajectory from file {}.'.format(
                        input_file_name), verb_th=1)
                    # Calculate ionic_step_offset for next file to ensure that
                    remaining_ionic_steps = nionic_steps - (ionic_step_indexes[-1] + 1)
                    if remaining_ionic_steps == 0:
                        ionic_step_offset = 0
                    else:
                        ionic_step_offset = self.keep_one_every - remaining_ionic_steps
                    self.ionic_step_indexes = ionic_step_indexes
                else:
                    if self.use_xdatcar_file:
                        traj, ionic_step_indexes, nionic_steps, thermo_data, _ = get_trajectory_from_xdatcar(
                            data_dir, constant_lattice=constant_lattice, get_thermodynamics_data=True,
                            keep_one_every=self.keep_one_every, verbosity=self.verbosity)
                    else:
                        traj, ionic_step_indexes, nionic_steps, thermo_data, _ = get_trajectory_from_vasp_run(
                            data_dir, ionic_step_skip=self.keep_one_every, ionic_step_offset=ionic_step_offset, verbosity=self.verbosity, 
                            **kwargs)
                    self.trajectory.extend(traj)
                    self.print('Trajectory appended from file {}.'.format(
                        input_file_name), verb_th=1)
                    self.extend_thermodynamics_data(thermo_data)
                    self.print('Thermodynamics_data appended from file {}.'.format(
                        oszicar_file_name), verb_th=1)
                self.original_files_info.append({
                        'data_dir': data_dir,
                        'ionic_step_indexes': ionic_step_indexes,
                        'nb_of_ionic_steps': ionic_step_indexes,
                    })
        self.set_simul_time()

    """
    def get_ionic_step_in_orig_file(index_in_trajectory):
        TO BE COMPLETED
    """

    def get_time_step(self):
        return self.trajectory.time_step

    def set_simul_time(self, initial_simul_time=None):
        """
        Set the simul_time property.

        Frames before initial_simul_time will be shifted to negative values.
        """
        if initial_simul_time is not None:
            self.initial_simul_time = initial_simul_time
        self.simul_time = self.get_time_step() * self.keep_one_every * np.arange(
            len(self.trajectory)) - self.initial_simul_time

    def set_displacements_data(self):
        """
        Set displacements_data property related to diplacements along the trajectory.

        displacements_data will be a dict of
        """
        _traj = self.trajectory.copy()
        if not _traj.coords_are_displacement:
            _traj.to_displacements()

        self.displacements_data = {}
        # Creating a 3D numpy array of displacements with shape [nb_steps, nb_atoms, 3]
        disp_3d = np.asarray([step for step in _traj])
        self.print('Using a 3D numpy darray of shape {} to track displacements.'.format(
            disp_3d.shape), verb_th=2)
        self.displacements_data['max_disp_norm'] = np.max(np.linalg.norm(disp_3d, axis=2),
                                                          axis=1)
        cumul_disp = np.cumsum(disp_3d, axis=0)
        self.displacements_data['max_cumul_disp_norm'] = np.max(
            np.linalg.norm(cumul_disp, axis=2), axis=1)
        self.displacements_data['mean_cumul_disp_norm'] = np.mean(
            np.linalg.norm(cumul_disp, axis=2), axis=1)
        self.displacements_data['RMSD'] = np.sqrt(np.mean(np.sum(np.square(disp_3d),
                                                                 axis=2), axis=1 ))
        self.displacements_data['MSD'] = np.mean(np.sum(np.square(disp_3d), axis=2),
                                                 axis=1 )

    def plot_thermodynamics_and_displacements(self, plot_negative_times=False):
        """
        Should be replaced in the end by a generic function
        """
        _data = self.thermodynamics_data.copy()
        if self.displacements_data is None:
            self.set_displacements_data()
        for k, v in self.displacements_data.items():
            _data[k] = v

        data_types = list(_data.keys())
        nb_of_cols = 2
        data_sets_per_subplot = 2
        nb_of_rows = get_nb_of_subplot_rows(len(data_types), data_sets_per_subplot,
                                            nb_of_cols)
        self.print('{} data types will be plotted in a {}x{} subplot array.'.format(
            len(data_types), nb_of_rows, nb_of_cols), verb_th=2)

        fig, axes = plt.subplots(nb_of_rows, nb_of_cols, sharex=True)

        def _make_main_plot(property_name, subplot_row_index=0,
                            subplot_col_index=0, property_unit='A.U.',
                            color='red'):
            axes[subplot_row_index, subplot_col_index].plot(self.simul_time,
                _data[property_name], color=color)
            axes[subplot_row_index, subplot_col_index].set_ylabel('{} ({})'.format(
                self.get_data_name(property_name, use_label_name=True), property_unit), color=color)
            axes[subplot_row_index, subplot_col_index].tick_params(axis ='y', labelcolor=color)


        def _add_twinx_plot(property_name, subplot_row_index=0,
                            subplot_col_index=0, property_unit='A.U.',
                            color='blue'):
            ax = axes[subplot_row_index, subplot_col_index].twinx()
            ax.plot(self.simul_time, _data[property_name], color=color)
            ax.set_ylabel('{} ({})'.format(self.get_data_name(property_name, use_label_name=True),
                property_unit), color=color)
            ax.tick_params(axis ='y', labelcolor=color)

        data_types = list(_data.keys())
        for type_index, data_type in enumerate(data_types):
            row_index, col_index, subplot_index = get_row_col_and_subplot_indexes(
                type_index, nb_of_rows, nb_of_cols, data_sets_per_subplot)
            self.print((f'data_type # {type_index}: {data_type} ;  subplot '
                        f'({row_index}, {col_index})'), verb_th=2)
            property_unit = self.get_data_unit(data_type)
            if (type_index % 2) == 0:
                _make_main_plot(data_types[type_index], row_index, col_index,
                                property_unit)
            else:
                _add_twinx_plot(data_types[type_index], row_index, col_index,
                                property_unit)

        # Set x axis labels (common to all suplots)
        for col_index in range(2):
            axes[-1, col_index].set_xlabel('Simulation time (fs)')
        if plot_negative_times:
            axes[0, 0].set_xlim([self.simul_time[0], self.simul_time[-1]])
        if not plot_negative_times:
            axes[0, 0].set_xlim([0, self.simul_time[-1]])

        # Show plot
        fig.tight_layout(w_pad=0.1)

        return fig, axes

    def get_first_time_positive_traj_index(self, initial_simul_time=None):
        """
        Get first trajectory index with simul_time >= 0 allowing the
        possibility to set self.initial_simul_time

        TODO: Add warning and set default value if initial_simul_time
        is out of allowed range.

        Args:
            initial_simul_time: float of None
        """
        if initial_simul_time is not None:
            self.set_simul_time(initial_simul_time)
        return self.get_first_traj_index_from_initial_time()


    def pick_evenly_spaced_traj_indexes(self, max_nb_of_frames=20,
        time_step=None, traj_index_step=None, initial_simul_time=None):

        initial_traj_index = self.get_first_time_positive_traj_index(initial_simul_time)

        # If time_step and traj_index_step are both None,
        # calculate a time_step according to
        if time_step is None and traj_index_step is None:
            time_step = self.simul_time[-1] / max_nb_of_frames
            self.print(('Setting time interval between picked trajectory indexes to '
                       '{:.3f} ps.').format(0.001 * time_step), verb_th=2)

        picked_traj_indexes = []
        pick_time = 0
        for traj_index in range(initial_traj_index, len(self.trajectory)):
            t = self.simul_time[traj_index]
            if len(picked_traj_indexes) >= max_nb_of_frames:
                break
            if t < 0:
                continue
            if time_step is not None:
                if t >= pick_time:
                    picked_traj_indexes.append(traj_index)
                    pick_time += time_step
                elif traj_index_step is not None:
                    picked_traj_indexes = range(traj_index, len(self.trajectory),
                        traj_index_step)[:max_nb_of_frames]
                    break

        self.print(('{} trajectory indexes ranging from {} ({:.3f} ps) to '
                    '{} ({:.3f} ps) have been selected.').format(len(picked_traj_indexes),
                    picked_traj_indexes[0], 0.001 * self.simul_time[picked_traj_indexes[0]],
                    picked_traj_indexes[-1], 0.001 * self.simul_time[picked_traj_indexes[-1]]),
                    verb_th=2)

        return picked_traj_indexes
        
        
    def pick_evenly_spaced_traj_indexes_in_time_range(self, time_range,
            time_step=None, traj_index_step=None):
        
        min_traj_index = self.get_first_traj_index_beyond_time_limit(time_range[0])
        max_traj_index = self.get_last_traj_index_up_to_time_limit(time_range[1])
        
        # If time_step and traj_index_step are both None,
        # calculate a time_step according to
        if time_step is None and traj_index_step is None:
            traj_index_step = 1
        elif isinstance(time_step, (float, int)):
            traj_time_step = self.simul_time[1] - self.simul_time[0]
            traj_index_step = int(np.around(time_step/traj_time_step))
            # DEBUGGING:
            print('traj_index_step = ', traj_index_step)
        elif isinstance(traj_index_step, int):
            if traj_index_step <= 0:
                raise ValueError('traj_index_step should be a positive integer.')
        else:
            raise TypeError('Use either an int for traj_index_step or a float or int for time_step')

        picked_time_step = self.simul_time[traj_index_step] - self.simul_time[0]
        
        picked_indexes = list(range(min_traj_index, max_traj_index, traj_index_step))
        
        t_min = self.simul_time[picked_indexes[0]]
        t_max = self.simul_time[picked_indexes[-1]]
        self.print(('{} trajectory indexes picked between {:.3f} and {:.3f} ps '
                    '(indexes {} to {}) with a time step of {:.3f} ps.'
                    ).format(len(picked_indexes), 0.001 * t_min, 0.001 * t_max, 
                    min_traj_index, max_traj_index, 0.001 * picked_time_step, 
                    verb_th=1))

        return picked_indexes
        

    def calculate_structure_evolution(self, traj_indexes, ref_traj_index=0,
                                      ref_structure = None, ref_struct_description=None,
                                      use_first_time_positive_as_ref=False,
                                      sigma=0.01, r_max=10.0, show_plot=True):
        """
        Calculate distance to a reference
        """
        if ref_structure is not None:
            ref_structure, ref_struct_description = get_pymatgen_structure(ref_structure,
                ref_struct_description=ref_struct_description)
        elif use_first_time_positive_as_ref:
            ref_traj_index = self.get_first_time_positive_traj_index()
            ref_structure = self.trajectory[ref_traj_index]
            ref_struct_description = 'Reference frame at t = {:.3f} ps'.format(
                1000 * self.simul_time(ref_traj_index))
        else:
            ref_structure = self.trajectory[ref_traj_index]
            ref_struct_description = 'Reference frame at t = {:.3f} ps'.format(
                1000 * self.simul_time[ref_traj_index])
        dmd = distanceMatrixData(Rmax=r_max, sigma=sigma)
        time =  [self.simul_time[traj_index] for traj_index in traj_indexes]
        # FIXME: Go back to positions otherwise trajectory[i] is a numpy.darray
        # rather than a pymatgen structure
        if self.trajectory.coords_are_displacement:
            coords_are_displacement_old = True
            self.trajectory.to_positions()
            self.print('Trajectory switched to positions to extract structures.', verb_th=2)
            self.print('Type(self.trajectory[0]) = {}'.format(type(self.trajectory[0])),
                       verb_th=2)
        else:
            coords_are_displacement_old = False
        distances = [dmd.calculate_cosine_distance(self.trajectory[traj_index],
                        ref_structure) for traj_index in traj_indexes]
        if coords_are_displacement_old:
            self.trajectory.to_displacements()
        if not show_plot:
            return time, distances
        else:
            fig, ax = plt.subplots(1, 1)
            ax.plot(time, distances, 'o')
            ax.set_xlabel('Simulation time (fs)')
            ax.set_ylabel(f'Cosine distance to {ref_struct_description}')
            ax.set_title('Structure evolution')
            # FIXME:
            lgd = ['\u03C3 = {} \u212B'.format(sigma)]
            ax.legend(lgd)
            return time, distances, fig, ax

    def set_xray_pdf_parameters(self, r_max=10.0, sigma=0.01,
                                wavelength=0.559422, q_max=17.0,
                                fNy=5, verbosity=0):
        """

        wavelength:
           In angtroms
           0.709318 for Mo K_alpha_1
           0.559422 for Ag K_apha_1
           0.563813 for Ag K_apha_2
        """

        self.xray_pdf_r_max = r_max
        self.xray_pdf_sigma = sigma
        self.xray_pdf_wavelength = wavelength
        self.xray_pdf_q_max = q_max
        self.xray_pdf_fNy = fNy
        self._nanopdf_data = nanopdfData(R_max=self.xray_pdf_r_max,
                                         sigma=self.xray_pdf_sigma,
                                         lambda1=self.xray_pdf_wavelength,
                                         Qmax=self.xray_pdf_q_max,
                                         fNy=self.xray_pdf_fNy,
                                         verbosity=verbosity)
        self.is_r_ascending = True
        self.print('Exact X-ray pdf parameters set to:', verb_th=1)
        self.print('r_max = {} \u212B'.format(self.xray_pdf_r_max), verb_th=1)
        self.print('sigma = {} \u212B (Gaussian smearing)'.format(self.xray_pdf_sigma),
                   verb_th=1)
        self.print('wavelength = {} \u212B'.format(self.xray_pdf_wavelength),
                   verb_th=1)
        self.print('Qmax = {} \u212B^-1'.format(self.xray_pdf_q_max), verb_th=1)
        self.print('fNy = {}'.format(self.xray_pdf_fNy), verb_th=1)

    def get_xray_pdf_r(self):
        return self._nanopdf_data.R

    def get_stored_xray_pdf(self):
        return self._nanopdf_data.exactPDF

    def update_xray_pdf_r_max(self):
        self.xray_pdf_r_max = self._nanopdf_data.R_max

    def get_single_frame_pdf(self, traj_index, show_plot=False, **kwargs):
        """
        Get exact x-ray total scattering function and plot, possibly with
        experiment

        Args:
            traj_index: int
                Frame index in trajectory property
            show_plot: bool (default: False)

            **kwargs:
                All keyword arguments used in nanopdfData.plot_exact_pdf

        Returns:
            r, pdf if show_plot is False
            r, pdf, fig, ax if show_plot is True
        """
        description = (('Exact PDF for structure index {} in trajectory'
                        ' (t = {:.3f} ps).').format(traj_index,
                       0.001 * self.simul_time[traj_index]))
        if 'title' not in kwargs.keys():
            kwargs['title'] = description
        self.print('Calculating: {}'.format(kwargs['title']), verb_th=2)
        self._nanopdf_data.calculate_final_exact_pdf_from_structure(
            self.trajectory[traj_index])
        if show_plot:
            fig, ax = self._nanopdf_data.plot_exact_pdf(**kwargs)
            return self.get_xray_pdf_r(), self._nanopdf_data.exactPDF, fig, ax
        else:
            return self.get_xray_pdf_r(), self._nanopdf_data.exactPDF

    def get_multi_frames_pdf(self, traj_indexes, show_plot=False, **kwargs):
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
                'r': vector of r values in Angstroms
                'pdfs': numpy array of shape (nb_of_traj_indexes, len(r))
        """
        if isinstance(traj_indexes, int):
            traj_indexes = [traj_indexes]
        if show_plot:
            if 'title' not in kwargs.keys():
                kwargs['title'] = '{} : exact X-ray PDF extracted from MD frames'.format(
                    self.description)
            if 'data_names' not in kwargs.keys():
                kwargs['data_names'] = ['Frame at t = {:.3f} ps'.format(
                    0.001 * self.simul_time[traj_index]) for traj_index in traj_indexes]

        pdf_data={'traj_indexes': []}
        for traj_index in traj_indexes:
            self.print('Calculating pdf for traj_index {}'.format(traj_index), verb_th=2)
            self._nanopdf_data.calculate_final_exact_pdf_from_structure(
                self.trajectory[traj_index])
            (r, pdf) = (self._nanopdf_data.R, self._nanopdf_data.exactPDF)

            pdf_data['traj_indexes'].append(traj_index)

            if 'pdfs' not in pdf_data.keys():
                pdf_data['pdfs'] = np.reshape(pdf, (1,-1))
            else:
                pdf_data['pdfs'] = np.append(pdf_data['pdfs'],
                                             np.reshape(pdf, (1,-1)), axis=0)
            if 'r' not in pdf_data.keys():
                pdf_data['r'] = r

        if show_plot:
            fig, ax = self._nanopdf_data.plot_multiple_exact_pdfs(
                [self.trajectory[traj_index] for traj_index in traj_indexes],
                **kwargs)
            return pdf_data, fig, ax
        else:
            return pdf_data

    def get_averaged_partials(self, traj_indexes, show_plot=False,
                              y_rel_shift=0.25):
        """
        Get all averaged partials for a selection of frames stored in trajectory

        Args:
            indexes of frames in self.trajectory

        TODO: a method to get averaged partials from a list of structures
        should be added to distanceMatrixData in the structureComparisons
        module.

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
        
        # Define a distanceMatrixData instance using this (???) r vector
        dmd = distanceMatrixData(R=self.get_xray_pdf_r(),
                                 sigma=self.xray_pdf_sigma)
        for i, traj_index in enumerate(traj_indexes):
            # Calculate all radial distributions functions
            self.print(('Calculating partials for frame {} out {} at t = '
                        '{:.3f} ps ').format(i, len(traj_indexes),
                        0.001*self.simul_time[traj_index]), verb_th=1)
            
            partials, types = dmd.calculate_all_reduced_partial_RDFs(
                self.trajectory[traj_index])
            if i == 0:
                # Add a 4th dimension corresponding to frame index
                all_partials = np.zeros(list(partials.shape) + [len(traj_indexes)])
            all_partials[:, :, :, i] = partials

        averaged_partials = np.mean(all_partials, axis=3)
        if not show_plot:
            return averaged_partials, types
        else:
            tot_y_shift = 0
            y_shift = y_rel_shift*(max(partials[0, 0, :])-min(partials[0, 0, :]))
            lgd = []
            fig, ax = plt.subplots()
            for i in range(partials.shape[0]):
                for j in range(i+1):
                    plt.plot(self.get_xray_pdf_r(),
                             partials[i, j, :] + tot_y_shift)
                    tot_y_shift += y_shift
                    lgd.append('{}-{} partial'.format(types[i], types[j]))
            ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)',
                   title='Partials averaged from {} configurations.'.format(
                   len(traj_indexes)))
            ax.legend(lgd)
            return averaged_partials, types, fig, ax


    def get_xray_pdf_from_partials(self, partials, types,
                                   export_to_file=None, include_partials=True,
                                   show_plot=True, rel_y_shift=0.5,
                                   experiment_file=None, partials_rel_y_shift=0.0,
                                   experiment_title=None):
        """
        Get the exact X-ray total scattering from pre-calculated partials.

        TODO: adapt to use as directly as possible the plotting functionalities
        of nanopdfData, using **kwargs
        Add a include_individuals=False functionality to include a superposition
        of all frames in the plot. WARN THE USER THAT THIS IS TIME CONSUMING.

        TODO: UPDATE DOCUMENTATION

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
        self._nanopdf_data.set_reference_structure(self.trajectory[0])
        self._nanopdf_data.calculate_final_exact_pdf_from_partials(partials,
            types_in_partials=types)
        (r, pdf) =  (self._nanopdf_data.R, self._nanopdf_data.exactPDF)

        if isinstance(export_to_file, str):
            self._nanopdf_data.export_exact_pdf(file_name=export_to_file,
                                                include_partials=include_partials)
            self.print('Total X-ray scattering saved to file: {}'.format(
                export_to_file), verb_th=1)
        if not show_plot:
            return r, pdf
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
                                          self.trajectory[0].formula))
            else:
                ax.set_title('{} - Total X-ray scattering'.format(
                    self.trajectory[0].formula))
            ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)',
                   xlim=[0, self._nanopdf_data.R_max])
            ax.legend(lgd)
            return r, pdf, fig, ax


    def get_trajectory_in_positions(self, get_copy=True):
        if get_copy:
            traj = self.trajectory.copy()
        else:
            traj = self.trajectory
        if traj.coords_are_displacement:
            traj.to_positions()
        return traj


    def get_total_fractional_displacements(self, reference_traj_index=0,
                                           reference_frac_coords=None):
        """
        Get an array of fractional positions relative a reference coordinates

        Args:
            reference_traj_index: int (default is 0)
                Trajectory index used as the reference initial position to calculate
                calculate the total displacements.Only used if reference_frac_coords
                is None.
            reference_frac_coords: None or numpy.ndarray (default is None)
                Reference fractional coordinates used to calculate total displacements.
                should be of shape (nb_of_atoms, 3).

        Returns:
            A numpy.ndarray of the same shape as self.trajectory.frac_coords, i.e.
            (nb_of_trajectory_frames, nb_of_atoms, 3)
        """
        traj = self.get_trajectory_in_positions(get_copy=True)
        if isinstance(reference_frac_coords, np.ndarray):
            if reference_frac_coords.shape == traj.base_positions.shape:
                ref_frac_coords = reference_frac_coords
            else:
                raise ValueError(('reference_frac_coords should be a numpy ndarray '
                                  'of shape {}.').format(traj.base_positions.shape))
        elif isinstance(reference_traj_index, int):
            if reference_traj_index >= 0 and reference_traj_index < len(traj):
                ref_frac_coords = traj.frac_coords[reference_traj_index]
            else:
                raise ValueError(('reference_traj_index should be a positive integer smaller '
                                  'than the trajectory length ({}).').format(len(traj)))
        # Broadcast ref_frac_coords to the shape of frac_coords and substract thereof.
        total_frac_displacements = traj.frac_coords - np.broadcast_to(
            np.expand_dims(ref_frac_coords, axis=0), traj.frac_coords.shape)

        return total_frac_displacements

    def get_cartesian_coords(self, frac_coords, lattice=None):
        if lattice is None:
            if not self.trajectory.constant_lattice:
                raise ValueError(('Trajectory does not have a constant_lattice. '
                                  ' lattice must be set.'))
            else:
                lattice = self.trajectory.lattice
        l = Lattice(lattice)
        cart_coords = l.get_cartesian_coords(frac_coords)

    def get_last_traj_index_up_to_time_limit(self, time_limit=None,
                                              include_equal=True):
        traj_index_limit = len(self.simul_time)
        if isinstance(time_limit, (int, float)):
            if time_limit >= self.simul_time[0] and time_limit < self.simul_time[-1]:
                if include_equal:
                    indexes = np.nonzero(self.simul_time <= time_limit)
                else:
                    indexes = np.nonzero(self.simul_time < time_limit)
                traj_index_limit = indexes[0][-1]
        return traj_index_limit
    
    def get_first_traj_index_beyond_time_limit(self, time_limit=0.0,
                                               return_if_none=None):
        """
        Args:
            time_limit: float (default is 0.0)
            return_if_none: int or None:
                Use to decide what value should be returned in case no
                simul_time value is above the specified time_limit.
                Use  to use the result in a range of indexes as in
                self.trajectory[0:traj_index].
        """
        traj_index = return_if_none
        indexes = np.nonzero(self.simul_time > time_limit)
        if len(indexes[0]):
            traj_index = indexes[0][0]
        return traj_index

    def get_first_traj_index_from_initial_time(self):
        indexes = np.nonzero(self.simul_time >= 0)
        initial_traj_index = indexes[0][0]
        return initial_traj_index

    def get_atom_selection_from_text(self, text=None, check=True):
        """
        Convet a text of space- or comma-separated values to a list of atom types or integers

        TODO: add wildcards
        """
        if text is None:
            atom_selection = {
                'select_by': 'type',
                'selection': list(self.trajectory.get_structure(0).symbol_set)
            }
        else:
            atom_selection = {'select_by': '', 'selection': []}
            if ',' in text:
                selections = ('').join(text.split(' ')).split(',')
            else:
                selections = text.split()
            try:
                atom_selection['selection'] = [int(i) for i in selections]
                atom_selection['select_by'] = 'index'
            except ValueError:
                # TODO: add a mixed selection type
                atom_selection['select_by'] = 'type'
                atom_selection['selection'] = selections
            if check:
                struct = self.trajectory.get_structure(0)
                if atom_selection['select_by'] == 'index' and (
                        max(atom_selection['selection']) > struct.num_sites or
                        max(atom_selection['selection']) < 0):
                    warnings.warn('Atom indexes should be between 0 and struct.num_sites.')
                elif atom_selection['select_by'] == 'type':
                    if set(atom_selection['selection']) not in set(struct.symbol_set):
                        warnings.warn(('Some of the selected types {} are not among '
                                       'trajectory types: {}.').format(set(atom_selection['selection']), set(struct.symbol_set)))
        return atom_selection

    @staticmethod
    def is_in_selection(index_or_type, atom_selection):
        """
        Test if index or type is in the atom_selection dictionary
        defined as in read_atom_selection_from_text
        """
        if index_or_type in atom_selection['selection']:
            return True

    def track_displacements_above_threshold(self, threshold, reference_traj_index=0,
                                            reference_frac_coords=None, 
                                            tracking_axis=None, atom_types=None, 
                                            atom_indexes=None, markersize=3,
                                            tracking_time_limit=None,
                                            show_3d_plot_grid=False, 
                                            reference_atoms_markersize=10,
                                            reference_atoms_alpha=0.25,
                                            figure_sizes=[8.0, 8.0]):
        """
        Track atom displacements above threshold in interactive plots

        IMPLEMENTATION IN PROGRESS

        WARNING: currently works only for constant_lattice trajectories
            (NVT ensemble)
        FIXME: PROBLEMS AT CELL BORDERS THAT SEEM TO ARISE AT TRAJECTORY CONCATENATION
              THIS NEEDS TO BE SOLVED AT THAT LEVEL. TRACK DISPLACEMENTS OF THE ORDER
              OF 1 IN FRACTIONAL COORDINATES.

        TODO: (difficult) Draw reference atoms a litle bit (?) beyond original cell
            Easier (but heavier): show a full 3x3x3 supercell and zoom in onto
            original cell

        TODO: show site labels on figures.
            Ideal would be to do it interactively (as cursor moves onto 
            corresponding lines).
        
        Args:
            threshold: float
                Threshold displacement in Angstrom if tracking_axis is None, in fraction
                of the considered cell vector if tracking_axis is an integer between 0 and 2.
            reference_traj_index: int (default is 0)
                Trajectory index used as the reference initial position to calculate
                calculate the total displacements.Only used if reference_frac_coords
                is None.
            reference_frac_coords: None or numpy.ndarray (default is None)
                Reference fractional coordinates used to calculate total displacements.
                should be of shape (nb_of_atoms, 3).
            tracking_axis: int or None (default is None)
                Track displacements along a certain direction.
                If None
            tracking_time_limit: float or None (default is None)
                Initial time limit to tracking of fs
            show_3d_plot_grid: bool (default is False)
                Whether 3D grid with axes and ticks should be shown on 
                3D plot of tracked atomic cartesian coordinates.
            figure_sizes: list or tuple (default: [8.0, 8.0])
                Figure sizes in inches. Note that the default for
                matplotlib figures is [6.4, 4.8].
        """
        total_frac_displacements = self.get_total_fractional_displacements(
            reference_traj_index=reference_traj_index, reference_frac_coords=reference_frac_coords)
        traj = self.get_trajectory_in_positions(get_copy=True)
        if not traj.constant_lattice:
            raise ValueError(('Trajectory does not have a constant_lattice. '
                              ' lattice must be set.'))
        else:
            lattice = Lattice(traj.lattice)
        total_cart_displacements = lattice.get_cartesian_coords(total_frac_displacements)
        max_displacement = np.max(np.linalg.norm(total_cart_displacements, axis=2))

        # Initialize
        tracked_coords = {}
        if not isinstance(tracking_time_limit, (float, int)):
            tracking_time_limit = self.simul_time[-1]
        elif tracking_time_limit < self.simul_time[0] or (
                tracking_time_limit >= self.simul_time[-1]):
            warnings.warn(('tracking_time_limit should be between {} and {}.'
                           'The value has been set to {}.').format(
                          self.simul_time[0], self.simul_time[-1], self.simul_time[-1]))
            tracking_time_limit = self.simul_time[-1]

        def update_tracked_coords(_threshold, time_limit=None):
            """
            track fractional and cartesian coordinates for atoms whose
            displacement goes above a threshold in internal or cartesian
            coordinates.
            """
            # Empty tracked coords without deleting it to keep global character
            for k in list(tracked_coords.keys()):
                tracked_coords.pop(k)

            traj_index_limit = self.get_last_traj_index_up_to_time_limit(
                time_limit=time_limit, include_equal=True)

            if tracking_axis is None:
                # Track displacements above threshold in Angstroms
                indexes = np.nonzero(np.linalg.norm(
                    total_cart_displacements[0:traj_index_limit + 1, :, :], axis=2)
                    > _threshold)
                tracked_atoms = list(set(indexes[1]))
                below_th_indexes = np.nonzero(np.linalg.norm(
                    total_cart_displacements[0:traj_index_limit + 1, tracked_atoms, :], axis=2)
                    <= _threshold)
            elif isinstance(tracking_axis, int):
                indexes = np.nonzero(total_frac_displacements[
                    0:traj_index_limit + 1, :, tracking_axis] > _threshold)
                tracked_atoms = list(set(indexes[1]))
                below_th_indexes = np.nonzero(total_frac_displacements[
                    0:traj_index_limit + 1, tracked_atoms, tracking_axis] <= _threshold)

            # indexes[0] give the frame indexes, Indexes[1] the atom indexes
            if len(indexes[1]) == 0:
                return
            for traj_index, atom_index in zip(indexes[0], indexes[1]):
                if atom_index not in tracked_coords.keys():
                    tracked_coords[atom_index] = {
                        'traj_indexes': [traj_index],
                        'frac_coords': [traj.frac_coords[traj_index,atom_index,:]],
                        'cart_coords': [lattice.get_cartesian_coords(traj.frac_coords[
                            traj_index,atom_index,:])],
                    }
                else:
                    tracked_coords[atom_index]['traj_indexes'].append(traj_index),
                    tracked_coords[atom_index]['frac_coords'].append(traj.frac_coords[
                        traj_index,atom_index,:])
                    tracked_coords[atom_index]['cart_coords'].append(
                        lattice.get_cartesian_coords(traj.frac_coords[
                            traj_index,atom_index,:]))
            # End of function update_tracked_coords()

        atom_selection = {'select_by': 'type',
                          'selection': list(traj.get_structure(0).symbol_set)}
        if atom_indexes is not None:
            # convert single element to list
            if isinstance(atom_indexes, int):
                atom_indexes = [atom_indexes]
            if max(atom_indexes) >= traj.get_structure(0).num_sites:
                warnings.warn('atom_indexes should be smaller than {}.'.format(
                    traj.get_structure(0).num_sites))
            else:
                atom_selection = {'select_by': 'index', 'selection': atom_indexes}
        elif atom_types is not None:
            # convert single element to list
            if isinstance(atom_types, str):
                atom_types = [atom_types]
            if set(atom_types) not in set(traj.get_structure(0).symbol_set):
                warnings.warn(('Some of the selected types {} are not among '
                    'trajectory types: {}.').format(set(atom_types),
                    set(traj.get_structure(0).symbol_set)))
            else:
                atom_selection = {'select_by': 'type', 'selection': atom_types}

        # Set a atom_types based on all available sites
        atom_types = list(traj.get_structure(0).symbol_set)

        # Define colors based on the default cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig_3d_traj = plt.figure(figsize=figure_sizes)
        ax_3d_traj = fig_3d_traj.add_subplot(projection='3d')

        fig_pos_vs_time, ax_pos_vs_time = plt.subplots(2, 2, sharex=True)
        fig_pos_vs_time.set_size_inches(figure_sizes)
        ax_pos_vs_time[1, 1].set_visible(False)

        # adjust the main plot to make room for the sliders
        fig_3d_traj.subplots_adjust(bottom=0.15)

        # Define sliders
        # TODO: leave room to add a button to switch between threshold types
        slider_thr_3d_traj_ax = fig_3d_traj.add_axes([0.55, 0.02, 0.35, 0.05])
        slider_thr_pos_vs_time_ax = fig_pos_vs_time.add_axes([0.70, 0.02, 0.25, 0.05])
        slider_simul_time_ax = fig_3d_traj.add_axes([0.55, 0.12, 0.35, 0.05])
        slider_timelim_3d_traj_ax = fig_3d_traj.add_axes([0.55, 0.07, 0.35, 0.05])
        slider_timelim_pos_vs_time_ax = fig_pos_vs_time.add_axes([0.70, 0.07, 0.25, 0.05])
        button_show_grid_ax = fig_3d_traj.add_axes([0.05, 0.12, 0.20, 0.04])
        button_show_cell_ax = fig_3d_traj.add_axes([0.05, 0.07, 0.20, 0.04])
        textbox_atom_selection_ax = fig_3d_traj.add_axes([0.20, 0.02, 0.15, 0.04])

        if tracking_axis is None:
            valmax=max_displacement
            thr_unit = '\u212B'
        elif isinstance(tracking_axis, int):
            valmax=np.max(total_frac_displacements[:, :, tracking_axis])
            if tracking_axis == 0:
                thr_unit = '1/a'
            elif tracking_axis == 1:
                thr_unit = '1/b'
            elif tracking_axis == 2:
                thr_unit = '1/c'
            else:
                raise ValueError('tracking_axis value should be 0, 1, 3 or None.')
        thr_slider_3d_traj = Slider(
            ax=slider_thr_3d_traj_ax,
            label='Threshold ({})'.format(thr_unit),
            valmin=0.0,
            valmax=valmax,
            valinit=threshold,
        )
        thr_slider_pos_vs_time = Slider(
            ax=slider_thr_pos_vs_time_ax,
            label='Threshold ({})'.format(thr_unit),
            valmin=0.0,
            valmax=valmax,
            valinit=threshold,
        )
        simul_time_slider = Slider(
            ax=slider_simul_time_ax,
            label='Simulation time (fs)',
            valmin=0,
            valmax=tracking_time_limit,
            valinit=tracking_time_limit,
            valstep=self.simul_time[1] - self.simul_time[0],
        )
        # Initialize sliders for the tracking time limit
        timelim_slider_3d_traj = Slider(
            ax=slider_timelim_3d_traj_ax,
            label='Tracking time limit (fs)',
            valmin=0,
            valmax=self.simul_time[-1],
            valinit=tracking_time_limit,
        )
        timelim_slider_pos_vs_time = Slider(
            ax=slider_timelim_pos_vs_time_ax,
            label='Tracking time limit (fs)',
            valmin=0,
            valmax=self.simul_time[-1],
            valinit=tracking_time_limit,
        )
        buttons_states = {
            'grid': show_3d_plot_grid,
            'cell': True
        }
        show_grid_button = Button(
            ax=button_show_grid_ax,
            label='Show/hide grid')
        show_cell_button = Button(
            ax=button_show_cell_ax,
            label='Show/hide cell')
        atom_selection_textbox = TextBox(ax=textbox_atom_selection_ax,
            label='Atom selection',
            initial=', '.join([str(i) for i in atom_selection['selection']]))

        # Retreive reference coords based on reference_traj_index or
        # reference_frac_coords
        if isinstance(reference_frac_coords, np.ndarray):
            if reference_frac_coords.shape == traj.base_positions.shape:
                ref_frac_coords = reference_frac_coords
        elif isinstance(reference_traj_index, int):
            ref_frac_coords = traj.frac_coords[reference_traj_index]
        ref_cart_coords = lattice.get_cartesian_coords(ref_frac_coords)

        lines_3d_traj = {}
        lines_pos_vs_time = {}

        def update_lines_3d_traj():
            if len(tracked_coords.keys()) == 0:
                self.print('tracked_coords is empty. Threshold value may be too low.')
            for atom_index in tracked_coords.keys():
                if atom_index not in lines_3d_traj.keys():
                    lines_3d_traj[atom_index] = {}
                # Skip if type of atom_index not in atom_types
                atom_type = traj.get_structure(0).species[atom_index].name
                if (
                    atom_selection['select_by'] == 'type' and
                    atom_type not in atom_selection[ 'selection']
                    ) or (
                    atom_selection['select_by'] == 'index' and
                    atom_index not in atom_selection['selection']
                    ):
                    continue
                # Plot initial position as black open o
                atom_color = colors[atom_types.index(atom_type)]
                [x_0, y_0, z_0] = list(ref_cart_coords[atom_index,:])
                x = [tracked_coords[atom_index]['cart_coords'][i][0] for i in
                     range(len(tracked_coords[atom_index]['cart_coords']))]
                y = [tracked_coords[atom_index]['cart_coords'][i][1] for i in
                     range(len(tracked_coords[atom_index]['cart_coords']))]
                z = [tracked_coords[atom_index]['cart_coords'][i][2] for i in
                     range(len(tracked_coords[atom_index]['cart_coords']))]
                self.print('{}-{}\n\tx = {}\n\ty = {}\n\tz = {}'.format(atom_type, atom_index, x, y, z),
                    verb_th=3)
                lines_3d_traj[atom_index]['initial_position'] = ax_3d_traj.plot(
                    x_0, y_0, z_0, marker='o', color=atom_color, markersize=markersize,
                    fillstyle='none')
                # TODO: annotate initial point
                lines_3d_traj[atom_index]['before_tracking'] = ax_3d_traj.plot([x_0, x[0]], [y_0, y[0]], [z_0, z[0]],
                    ':', color=atom_color)
                lines_3d_traj[atom_index]['tracking'] = ax_3d_traj.plot(
                    x, y, z, 'o', color=atom_color, fillstyle='full',
                    markersize=markersize, label='{}-{}'.format(atom_type, atom_index))
                ax_3d_traj.set_proj_type('ortho')
                # Plot full trajectory between initial simul time and
                # tracking time limit
                # TODO: add condition based on corresponding button
                tracking_time_limit = timelim_slider_3d_traj.val
                self.print('tracking_time_limit = {} fs.'.format(tracking_time_limit),
                           verb_th=3)
                start_traj_index = self.get_first_traj_index_from_initial_time()
                end_traj_index = self.get_first_traj_index_beyond_time_limit(
                    tracking_time_limit, return_if_none=None)
                cart_coords = lattice.get_cartesian_coords(traj.frac_coords[
                    start_traj_index:end_traj_index, atom_index, :])
                lines_3d_traj[atom_index]['full'] = ax_3d_traj.plot(
                    cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2], '-',
                    color=atom_color)

        def update_lines_pos_vs_time():
            if len(tracked_coords.keys()) == 0:
                self.print('tracked_coords is empty. Threshold value may be too low.')
            for atom_index in tracked_coords.keys():
                if atom_index not in lines_pos_vs_time.keys():
                    lines_pos_vs_time[atom_index] = {}
                # Skip if type of atom_index not in atom_types
                atom_type = traj.get_structure(0).species[atom_index].name
                if (
                    atom_selection['select_by'] == 'type' and
                    atom_type not in atom_selection[ 'selection']
                    ) or (
                    atom_selection['select_by'] == 'index' and
                    atom_index not in atom_selection['selection']
                    ):
                    continue
                # Plot initial position as black open o
                atom_color = colors[atom_types.index(atom_type)]
                # Define a local _ref_frac_coords for the condifered atom_index
                _ref_frac_coords = list(ref_frac_coords[atom_index,:])
                # WARNING: this will not work if an external reference structure
                # is used.
                ref_simul_time = self.simul_time[reference_traj_index]
                simul_times = [self.simul_time[tracked_coords[atom_index]['traj_indexes'][i]]
                               for i in range(len(tracked_coords[atom_index]['traj_indexes']))]
                for cell_index, cell_axis in enumerate(['a', 'b', 'c']):
                    if cell_axis not in lines_pos_vs_time[atom_index].keys():
                        lines_pos_vs_time[atom_index][cell_axis] = {}
                    frac_coords = [tracked_coords[atom_index]['frac_coords'][i][cell_index]
                                   for i in range(len(tracked_coords[atom_index][
                                   'frac_coords']))]
                    subplot_indexes = np.unravel_index(cell_index, (2, 2))
                    lines_pos_vs_time[atom_index][cell_axis]['initial_position'
                        ] = ax_pos_vs_time[subplot_indexes].plot(ref_simul_time,
                        _ref_frac_coords[cell_index], marker='o', color=atom_color,
                        markersize=markersize)
                    lines_pos_vs_time[atom_index][cell_axis]['before_tracking'] = \
                        ax_pos_vs_time[subplot_indexes].plot(
                        [ref_simul_time, simul_times[0]],
                        [_ref_frac_coords[cell_index], frac_coords[0]],
                        ':', color=atom_color)
                    lines_pos_vs_time[atom_index][cell_axis]['tracking'] = \
                        ax_pos_vs_time[subplot_indexes].plot(simul_times,
                        frac_coords, 'o', color=atom_color, fillstyle='full',
                        markersize=markersize, label='{}-{}'.format(atom_type, atom_index))

                # Plot full trajectory up to tracking_time_limit
                start_traj_index = self.get_first_traj_index_from_initial_time()
                tracking_time_limit = timelim_slider_pos_vs_time.val
                self.print('tracking_time_limit = {} fs.'.format(tracking_time_limit),
                           verb_th=3)
                end_traj_index = self.get_first_traj_index_beyond_time_limit(
                    tracking_time_limit, return_if_none=None)
                simul_times = self.simul_time[start_traj_index:end_traj_index]

                for cell_index, cell_axis in enumerate(['a', 'b', 'c']):
                    if cell_axis not in lines_pos_vs_time[atom_index].keys():
                        lines_pos_vs_time[atom_index][cell_axis] = {}
                    frac_coords = traj.frac_coords[start_traj_index:end_traj_index,
                        atom_index, cell_index]
                    subplot_indexes = np.unravel_index(cell_index, (2, 2))
                    lines_pos_vs_time[atom_index][cell_axis]['full'] = \
                        ax_pos_vs_time[subplot_indexes].plot(simul_times,
                        frac_coords, '-', color=atom_color)            

        # Initialize plots with the input threshold value
        update_tracked_coords(threshold)
        update_lines_3d_traj()
        if not buttons_states['grid']:
            ax_3d_traj.set_axis_off()
        update_lines_pos_vs_time()

        # Define axes titles and x, y (, z) labels, and legends
        ax_3d_traj.set(xlabel='x (\u212B)', ylabel='y (\u212B)',
                       zlabel='z (\u212B)',
                       title=('Coordinates of atoms with displacement \nabove '
                              'threshold'))
        ax_pos_vs_time[0, 0].set(title='a fractional coordinate vs simulation time',
                                 ylabel='Fractional coordinate (1/a)')
        ax_pos_vs_time[0, 1].set(title='b fractional coordinate vs simulation time',
                                 xlabel='Simulation time (fs)',
                                 ylabel='Fractional coordinate (1/b)')
        ax_pos_vs_time[1, 0].set(title='c fractional coordinate vs simulation time',
                                 xlabel='Simulation time (fs)',
                                 ylabel='Fractional coordinate (1/c)')

        legend_elements = []
        for atom_index, atom_type in enumerate(atom_types):
            legend_elements.append(Patch(facecolor=colors[atom_index],
                edgecolor=colors[atom_index], 
                label='{} (tracked)'.format(atom_type)))
        ax_pos_vs_time[0, 1].legend(handles=legend_elements, loc='upper left',
                                    bbox_to_anchor=(0.7, -0.2))

        # Add reference atoms to ax_3d_traj
        if reference_frac_coords is None:
            ref_struct = traj.get_structure(reference_traj_index)
            ref_atom_types = list(ref_struct.symbol_set)
            cmap = get_cmap('Greys')
            ref_atom_colors = cmap(np.linspace(start=0.5, stop=1.0,
                num=len(ref_atom_types)))
            lines_ref_atoms = {}
            for type_index, atom_type in enumerate(ref_atom_types):
                site_indexes = [i for i, site in enumerate(ref_struct.sites) if
                                site.species_string == atom_type]
                print('{} sites: {}'.format(atom_type, site_indexes))
                lines_ref_atoms[atom_type] = ax_3d_traj.plot(
                    ref_struct.cart_coords[site_indexes, 0],
                    ref_struct.cart_coords[site_indexes, 1],
                    ref_struct.cart_coords[site_indexes, 2], 'o', 
                    alpha=reference_atoms_alpha,
                    markersize=reference_atoms_markersize, fillstyle='full',
                    color=ref_atom_colors[ref_atom_types.index(atom_type)])
        for atom_index, atom_type in enumerate(atom_types):
            legend_elements.append(Line2D([0], [0], marker='o', 
                linestyle='none', alpha=reference_atoms_alpha,
                markersize=reference_atoms_markersize, fillstyle='full',
                color=ref_atom_colors[ref_atom_types.index(atom_type)], 
                label='{} (reference)'.format(atom_type)))
        ax_3d_traj.legend(handles=legend_elements, bbox_to_anchor=[1.0, 1.0],
                          loc='upper left')
        fig_3d_traj.tight_layout()
        
        # plot unit cell
        ax_3d_traj, cell_edges = self.add_unit_cell_to_3d_plot(ax_3d_traj, lattice=lattice)
        # ax_3d_traj.add_collection3d(Poly3DCollection(cell_vert_cart, facecolor='none', edgecolor='k'))
        # TODO: add all reference atom positions with a small alpha
        #       and colors ranging from 50% gray to black.
        # TODO: add buttons to add or remove cell and reference atom positions

        # The functions to be called anytime the slider's value changes
        def update_3d_traj_thr(val):
            self.print('Slider value has been changed to {}.'.format(thr_slider_3d_traj.val),
                       verb_th=3)
            # Remove all existing line3D objects in lines_3d_traj :
            # DO NOT RESET lines_3d_traj OR GLOBAL CHARACTER WILL BE LOST
            for k1 in list(lines_3d_traj.keys()):
                for k2 in list(lines_3d_traj[k1].keys()):
                    for i in lines_3d_traj[k1][k2]:
                        i.remove()
                if k1 in lines_3d_traj.keys():
                    lines_3d_traj.pop(k1)
            update_tracked_coords(thr_slider_3d_traj.val,
                                  time_limit=timelim_slider_3d_traj.val)
            update_lines_3d_traj()
            thr_slider_3d_traj.eventson = False
            fig_3d_traj.canvas.draw_idle()
            thr_slider_pos_vs_time.set_val(thr_slider_3d_traj.val)
            simul_time_slider.set_val(thr_slider_3d_traj.val)
            thr_slider_3d_traj.eventson = True
            # TODO: change simul_time slider value to max traj_index

        def update_pos_vs_time_thr(val):
            self.print('Slider value has been changed to {}.'.format(
                thr_slider_3d_traj.val), verb_th=3)
            # Remove all existing line3D objects in lines_pos_vs_time :
            # DO NOT RESET lines_pos_vs_time OR GLOBAL CHARACTER WILL BE LOST
            for k1 in list(lines_pos_vs_time.keys()):
                for k2 in list(lines_pos_vs_time[k1].keys()):
                    for k3 in list(lines_pos_vs_time[k1][k2].keys()):
                        for i in lines_pos_vs_time[k1][k2][k3]:
                            i.remove()
                if k1 in lines_pos_vs_time.keys():
                    lines_pos_vs_time.pop(k1)
            update_tracked_coords(thr_slider_pos_vs_time.val,
                                  time_limit=timelim_slider_pos_vs_time.val)
            update_lines_pos_vs_time()
            thr_slider_pos_vs_time.eventson = False
            fig_pos_vs_time.canvas.draw_idle()
            thr_slider_3d_traj.set_val(thr_slider_pos_vs_time.val)
            simul_time_slider.set_val(thr_slider_pos_vs_time.val)
            thr_slider_pos_vs_time.eventson = True

        def get_tracked_traj_indexes():
            traj_indexes = set()
            self.print('Finding traj_indexes associated with Tracked atoms {}'.format(
                list(tracked_coords.keys())), verb_th=3)
            for atom_index in tracked_coords.keys():
                traj_indexes = traj_indexes.union(set(tracked_coords[atom_index]['traj_indexes']))
            traj_indexes = list(traj_indexes)
            traj_indexes.sort()
            self.print('tracked trajectory indexes = {}'.format(
                traj_indexes), verb_th=3)
            return traj_indexes

        def get_max_tracked_step_index():
            return max(get_tracked_traj_indexes())

        simul_time_lines = []
        def update_simul_time(val):
            self.print('Simulation-time slider value has been changed to {:.1f} fs.'.format(
                simul_time_slider.val), verb_th=2)
            # Clean simul_time_lines without removing it to keep global character
            for i in simul_time_lines:
                i.remove()
            simul_time_lines.clear()
            # get x, y, z values from ax_3d_traj[]
            
            start_traj_index = self.get_first_traj_index_from_initial_time()
            tracking_time_limit = timelim_slider_pos_vs_time.val
            self.print('tracking_time_limit = {} fs.'.format(tracking_time_limit),
                       verb_th=3)
            end_traj_index = self.get_first_traj_index_beyond_time_limit(
                tracking_time_limit, return_if_none=len(self.simul_time))
            traj_index = int(np.argmin(np.abs(self.simul_time - simul_time_slider.val)))
            tracked_atoms = list(tracked_coords.keys())
            if traj_index <= end_traj_index and len(tracked_atoms):
                cart_coords = lattice.get_cartesian_coords(traj.frac_coords[
                    traj_index, :, :])
                x = cart_coords[tracked_atoms, 0]
                y = cart_coords[tracked_atoms, 1]
                z = cart_coords[tracked_atoms, 2]
                simul_time_lines.extend(ax_3d_traj.plot(x, y, z, 'o',
                    color='k', fillstyle='none', markersize=markersize + 1))
                # add vertical lines on ax_pos_vs_time plots
                for cell_index in range(3):
                    subplot_indexes = np.unravel_index(cell_index, (2, 2))
                    x = 2*[self.simul_time[traj_index]]
                    y = ax_pos_vs_time[subplot_indexes].get_ylim()
                    simul_time_lines.extend(ax_pos_vs_time[subplot_indexes].plot(
                        x, y, '-', color='k', zorder=0))
                    ax_pos_vs_time[subplot_indexes].set_ylim(y)
                
            fig_3d_traj.canvas.draw_idle()
            fig_pos_vs_time.canvas.draw_idle()

        def update_3d_traj_timelim(val):
            tracking_time_limit = val
            # update figure by updating its threshold slider
            thr_slider_3d_traj.set_val(thr_slider_3d_traj.val)
            # Change the twin slider, avoiding infinite loop with eventson
            timelim_slider_3d_traj.eventson = False
            timelim_slider_pos_vs_time.set_val(val)
            # Reset ax_pos_vs_time x and y lim
            [ax_pos_vs_time[np.unravel_index(cell_index, (2, 2))].relim() 
             for cell_index in range(3)]
            timelim_slider_3d_traj.eventson = True
            end_traj_index = self.get_first_traj_index_beyond_time_limit(
                tracking_time_limit, return_if_none=len(self.simul_time))
            simul_time_slider.valmax = self.simul_time[end_traj_index]
            simul_time_slider.set_val(self.simul_time[end_traj_index])
            simul_time_slider.ax.set_xlim(simul_time_slider.valmin, 
                                          simul_time_slider.valmax)
            fig_3d_traj.canvas.draw_idle()

        def update_pos_vs_time_timelim(val):
            tracking_time_limit = val
            # update figure by updating its threshold slider
            thr_slider_pos_vs_time.set_val(thr_slider_pos_vs_time.val)
            # Reset ax_pos_vs_time x and y lim
            [ax_pos_vs_time[np.unravel_index(cell_index, (2, 2))].relim() 
             for cell_index in range(3)]
            # Change the twin slider, avoiding infinite loop with eventson
            timelim_slider_pos_vs_time.eventson = False
            timelim_slider_3d_traj.set_val(val)
            timelim_slider_pos_vs_time.eventson = True
            end_traj_index = self.get_first_traj_index_beyond_time_limit(
                tracking_time_limit, return_if_none=len(self.simul_time))
            simul_time_slider.valmax = self.simul_time[end_traj_index]
            simul_time_slider.set_val(self.simul_time[end_traj_index])
            simul_time_slider.ax.set_xlim(simul_time_slider.valmin, 
                                          simul_time_slider.valmax)
            fig_pos_vs_time.canvas.draw_idle()

        def show_hide_grid(event):
            if buttons_states['grid']:
                ax_3d_traj.set_axis_off()
            else:
                ax_3d_traj.set_axis_on()
            buttons_states['grid'] = not buttons_states['grid']

        def show_hide_cell(event):
            for lines in cell_edges:
                for line in lines:
                    line.set_visible(not buttons_states['cell'])
            buttons_states['cell'] = not buttons_states['cell']

        def select_atoms(val):
            # Clean atom_selection
            for k in list(atom_selection.keys()):
                atom_selection.pop(k)
            _atom_selection = self.get_atom_selection_from_text(text=val)
            self.print('atom selection set to : {}'.format(_atom_selection))
            for k, v in _atom_selection.items():
                atom_selection[k] = v
            thr_slider_pos_vs_time.set_val(thr_slider_pos_vs_time.val)
            thr_slider_3d_traj.set_val(thr_slider_3d_traj.val)
            atom_selection_textbox.eventson = False
            if atom_selection['select_by'] == 'type':
                atom_selection_textbox.set_val(', '.join(atom_selection['selection']))
            else:
                atom_selection_textbox.set_val(', '.join([str(i ) for i in
                                                          atom_selection['selection']]))
            fig_3d_traj.canvas.draw_idle()
            atom_selection_textbox.eventson = True

        # register the update function with each slider
        thr_slider_3d_traj.on_changed(update_3d_traj_thr)
        thr_slider_pos_vs_time.on_changed(update_pos_vs_time_thr)
        simul_time_slider.on_changed(update_simul_time)
        timelim_slider_3d_traj.on_changed(update_3d_traj_timelim)
        timelim_slider_pos_vs_time.on_changed(update_pos_vs_time_timelim)
        show_grid_button.on_clicked(show_hide_grid)
        show_cell_button.on_clicked(show_hide_cell)
        atom_selection_textbox.on_submit(select_atoms)

        plt.show()

        return fig_3d_traj, ax_3d_traj, fig_pos_vs_time, ax_pos_vs_time

    def add_unit_cell_to_3d_plot(self, axis, lattice=None, color='k',
                                 linewidth=1, linestyle='-', **kwargs):
        """
        Add unit cell to a 3D matplotlib plot

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        """

        if not self.trajectory.constant_lattice:
            raise ValueError(('Trajectory does not have a constant_lattice. '
                              ' lattice must be set.'))
        elif isinstance(lattice, np.ndarray):
            lattice = Lattice(matrix=lattice)
        elif isinstance(lattice, Lattice):
            lattice = lattice
        else:
            lattice = Lattice(self.trajectory.lattice)
        cell_vert_frac = []
        for i in range(2):
            cell_vert_frac.append(np.asarray([[i, 0, 0], [i, 1, 0], [i, 1, 1], [i, 0, 1]]))
        for j in range(2):
            cell_vert_frac.append(np.asarray([[0, j, 0], [1, j, 0], [1, j, 1], [0, j, 1]]))
        for k in range(2):
            cell_vert_frac.append(np.asarray([[0, 0, k], [1, 0, k], [1, 1, k], [0, 1, k]]))
        cell_vert_cart = [lattice.get_cartesian_coords(cell_vert_frac[i])
                          for i in range(len(cell_vert_frac))]
        cell_edges = []
        for poly in cell_vert_cart:
            for i in range(4):
                j = i + 1 if i < 3 else 0
                cell_edges.append(axis.plot(poly[(i, j), 0], poly[(i, j), 1], poly[(i, j), 2],
                                            color=color, linewidth=linewidth, linestyle=linestyle,
                                            **kwargs))
        return axis, cell_edges

    def plot_all_fractional_coordinates(self, show_plot=False):
        """
        Plot all fractional coordinates of a trajectory
        
        Can be useful to catch pbc issues.
        """        
        if self.trajectory.coords_are_displacement:
            self.trajectory.to_positions()
            atom_types = self.trajectory.get_structure(0).symbol_set
            sites = self.trajectory.get_structure(0).sites
            self.trajectory.to_displacements()
        else:
            atom_types = self.trajectory.get_structure(0).symbol_set
            sites = self.trajectory.get_structure(0)
        
        # TODO: check whether coords are positions
        frac_coords = np.asarray(self.trajectory.frac_coords)
        
        # Define colors based on the default cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
             
        fig, ax = plt.subplots(2, 2)
        for cell_index in range(3):
            for atom_index in range(frac_coords.shape[1]):
                atom_type = sites[atom_index].species.elements[0].name
                type_index = atom_types.index(atom_type)
                ax[np.unravel_index(cell_index, (2,2))].plot(
                    np.arange(frac_coords.shape[0]), 
                    frac_coords[:, atom_index, cell_index],'-',
                    color=colors[type_index])
        
        legend_elements = []
        for atom_index, atom_type in enumerate(atom_types):
            legend_elements.append(Patch(facecolor=colors[atom_index],
                edgecolor=colors[atom_index], 
                label='{} (tracked)'.format(atom_type)))
        ax[1, 1].set_axis_off()
        ax[1, 1].legend(handles=legend_elements, loc='upper left',
                        bbox_to_anchor=(0.1, 0.9))
        
        if show_plot:
            plt.show()
            
        return fig, ax

    def correct_pbc_problems(self, tolerances=[0.1, 0.1, 0.1]):
        """
        Correct periodicity errors typically encountered in concatenated trajectories
        """
        # Convert coords to positions if not already the case
        coords_are_displacement_orig = self.trajectory.coords_are_displacement
        if coords_are_displacement_orig:
            self.trajectory.to_positions()
        
        nb_of_atoms = self.trajectory[0].num_sites
        nb_of_frames = len(self.trajectory)
        while 1:
            # Calculate difference between frames frames. Initiate with zeros.
            frac_coord_diff = np.diff(self.trajectory.frac_coords, n=1, axis=0, 
                                prepend=self.trajectory.frac_coords[0:1, :, :])
            # Broadcast tolerances and ones to frac_coord_diff shape
            tol = np.broadcast_to(np.expand_dims(tolerances, axis=(0, 1)),
                                  (nb_of_frames, nb_of_atoms, 3))
            ones = np.broadcast_to(1.0, (nb_of_frames, nb_of_atoms, 3))
            indexes = np.nonzero(np.abs(frac_coord_diff) > ones - tol)
            if len(indexes[0]):
                self.print(('indexes associated with pbc issues:\n'
                   '{:20}{}\n{:20}{}\n{:20s}{}').format('Frames', list(indexes[0]),
                   'atoms', list(indexes[1]), 'axis', list(indexes[2])), verb_th=2)
                atoms_with_pbc_issue = set(indexes[1])
                for atom_index in atoms_with_pbc_issue:
                    for cell_index in range(3):
                        for index in range(len(indexes[0])):
                            if indexes[1][index] == atom_index and indexes[2][index] == cell_index:
                                frame_index = indexes[0][index]
                                correction = -np.sign(frac_coord_diff[frame_index, atom_index, cell_index])
                                self.trajectory.frac_coords[frame_index:, atom_index, cell_index] += correction
                                self.print(('{} was added to fractional coordinates of atom {} in '
                                            'cell direction {} for frames {} to {}').format(
                                            correction, atom_index, cell_index, frame_index, 
                                            nb_of_frames-1), verb_th=2)
                                break
            else:
                break
        
        """
        for traj_index in range(1, len(self.trajectory)):
            frac_diff.append(self.trajectory.frac_coords[traj_index] - 
                self.trajectory.frac_coords[traj_index - 1])
            for cell_index in range(3):
                indexes = np.nonzero(np.abs(frac_diff[:, cell_index]))
                    self.trajectory.frac_coords[traj_index][:, cell_index] - 
                    self.trajectory.frac_coords[traj_index - 1][:, cell_index]) > 1 - tolerances[cell_index]) 
                if len(indexes[0]):
                    self.print(('PBC problem detected for axis {} between '
                        'trajectory indexes {} and {}').format(cell_index, 
                        traj_index - 1, traj_index), verb_th=2)
        """            
                
                # NOW APPLY CORRECTION.
        
        # Convert coords back to positions if it was orignally the case
        if coords_are_displacement_orig:
            self.trajectory.to_displacements()
     