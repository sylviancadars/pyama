# io module for molcular dynamics input/outputs
import os
import numpy as np
import warnings
from inspect import getmembers
from pymatgen.io.vasp import Vasprun, Oszicar, Xdatcar, Incar, Poscar
from pymatgen.core.trajectory import Trajectory
from pymatgen.core.structure import Structure

def get_trajectory_from_vasp_run(dir_name, use_displacements=False,
                                 get_thermodynamics_data=True, verbosity=1,
                                 **kwargs):
    """
    Get a pymatgen.core.trajectory.Trajectory object from a vasp MD  or geom opt run

    Takes the same keyword arguments as paymatgen.Vasprun,including:
        ionic_step_skip (default: None), ionic_step_offset (default: 0),
        parse_dos (default: False), parse_eigen (default: False),
        parse_potcar_file(default: False), occu_tol (default: 1e-08),
        exception_on_bad_xml (default: True)

    The returned Trajectory instance may be used as follows:
        traj[index_number] is a numpy.darray of dimention [NATOMS, 3]
        containing positions or displacements.
        traj.base_positions contains the initial positions
        traj.time_step contains the time step in fs

    Note that trajectory objects may be combined in particular with:
        traj.extend(TRAJECTORY_OBJECT_TO_APPEND)

    Args:
        dir_name: str
            Directory containing the VASP molecular dynamics (or data.
        use_displacements: bool (default is True)
            whether displacements (rather than positions) should be used for
            the coords property of the returned trajectory object. Note that
            this may easily be reverted with the to_displacements() and
            to_positions() methods of pymatgen's Trajectory class.

    Returns:
        traj: pymatgen.core.trajectory.Trajectory instance.
    """
    file_name = os.path.join(dir_name, 'vasprun.xml')
    try:
        vr = Vasprun(file_name, **kwargs)
    except Exception as e:
        raise Exception(('Exception of type {} while creating Vasprun instance '
            'from file {}: {}.').format(type(e), file_name, e))
    if verbosity >= 2:
        print('Vasprun instance created from file {}.'.format(file_name))
        print('List of properties available for each of ionic step:')
        print(list(vr.ionic_steps[0].keys()))

    # Look for time_step in Vasprun instance
    if 'POTIM' in vr.parameters.keys():
        time_step = vr.parameters['POTIM']
        if verbosity >= 2:
            print(('Trajectory time_step ({:.2f} fs) read from vasprun '
                   'parameters[\'POTIM\']').format(time_step))
    else:
        time_step = 1.0
        warnings.warn(('POTIM not found in vasprun parameters. Trajectory '
                       'time step ({:2f} fs) may be wrong.').format(
                      time_step), UserWarning)
    # Set constant_lattice from ISIF in Vasprun instance
    if 'ISIF' in vr.parameters.keys():
        isif = int(vr.parameters['ISIF'])
        if verbosity >= 2:
            print('ISIF parameter = {} read from Vasprun instance.'.format(isif))
        constant_lattice = True if isif <= 2 else False
    else:
        constant_lattice = True
        if verbosity >= 2:
            print('ISIF parameter not found in vasprun.xml file.'
                  'constant_lattice set to True.')

    # traj = vr.get_trajectory(constant_lattice=constant_lattice)  # Currently does not work
    # define trajectory from structures instead
    structs = []
    for step in vr.ionic_steps:
        struct = step["structure"].copy()
        struct.add_site_property("forces", step["forces"])
        structs.append(struct)
    traj = Trajectory.from_structures(structs, constant_lattice=constant_lattice)
    traj.time_step = time_step

    nb_of_ionic_steps = vr.nionic_steps

    # Calculate ionic_steps and nb_of_ionic_steps
    keep_one_every = 1
    if 'ionic_step_skip' in kwargs.keys():
        if isinstance(kwargs['ionic_step_skip'], int):
            keep_one_every = kwargs['ionic_step_skip']
    ionic_step_offset = 0
    if 'ionic_step_offset' in kwargs.keys():
        if isinstance(kwargs['ionic_step_offset'], int):
            ionic_step_offset = kwargs['ionic_step_offset']
    ionic_step_indexes = list(range(ionic_step_offset, nb_of_ionic_steps, keep_one_every))

    if use_displacements:
        traj.to_displacements()

    if verbosity >= 3:
        print('Trajectory instance created with properties:')
        for name, value in getmembers(traj):
            if name[0] != '_':
                string = '{} (type: {})'.format(name, type(value))
                if isinstance(value, (list, tuple)):
                    string += ' of length {}'.format(len(value))
                print(string)

    if get_thermodynamics_data:
        thermo_data, is_md = get_energetics_from_vasp(dir_name, vasprun_instance=vr,
                                                      verbosity=verbosity, **kwargs)
        return traj, ionic_step_indexes, nb_of_ionic_steps, thermo_data, is_md
    else:
        return traj, ionic_step_indexes, nb_of_ionic_steps


def get_energetics_from_vasp(dir_name, vasprun_instance=None,
                             is_md=None, verbosity=1, **kwargs):
    """
    Get thermodynamics data from VASP OSZICAR and vasprun.xml files

    Takes the same keyword arguments as paymatgen.Vasprun,including:
        ionic_step_skip (default: None), ionic_step_offset (default: 0),
        parse_dos (default: False), parse_eigen (default: False),
        parse_potcar_file(default: False), occu_tol (default: 1e-08),
        exception_on_bad_xml (default: True)

    Args:
        TO BE COMPLETED

        is_md: bool or None: (default is None)
            Whether VASP data in dir_name correspond to molecular dynamics,
            in which case the temperature will be read from the OSZICAR file.
            If None the temperature will be searched in the OSZICAR file
            and is_md will be set accordingly (True is temperature information
            is found).

    Returns:
        thermo_data: a list of energies
    """
    thermo_data = {}
    oc = Oszicar(os.path.join(dir_name, 'OSZICAR') )
    if is_md is None:
        if 'T' in oc.ionic_steps[0]:
            is_md = True
        else:
            is_md = False
    if is_md is True:
        ionic_step_offset = 0
        if 'ionic_step_skip' in kwargs.keys():
            ionic_step_skip = kwargs['ionic_step_skip']
        else:
            ionic_step_skip = 1
        if 'ionic_step_offset' in kwargs.keys():
            ionic_step_offset = kwargs['ionic_step_offset']
        else:
            ionic_step_offset = 0
        thermo_data['temperature'] = [oc.ionic_steps[k]['T'] for k in
                                      range(ionic_step_offset, len(oc.ionic_steps),
                                            ionic_step_skip)]
    # TODO: add other 'energy types' for other types of dynamics
    avail_energy_types = ['e_fr_energy', 'total', 'e_0_energy', 'kinetic',
                          'nosepot', 'nosekinetic']
    if vasprun_instance is None:
        file_name = os.path.join(dir_name, 'vasprun.xml')
        try:
            vasprun_instance = Vasprun(os.path.join(file_name, **kwargs))
        except Exception as e:
            raise Exception(('Exception of type {} while creating Vasprun instance '
                             'from file {}: {}.').format(type(e), file_name, e))
        if verbosity >= 2:
            print('Vasprun instance created from file {}.'.format(file_name))

    energy_types = [energy_type for energy_type in avail_energy_types if
                    energy_type in vasprun_instance.ionic_steps[0].keys()]
    # initialize thermo_data
    for energy_type in energy_types:
        thermo_data[energy_type] = []

    for ionic_step in vasprun_instance.ionic_steps:
        for energy_type in energy_types:
            thermo_data[energy_type].append(ionic_step[energy_type])

    return thermo_data, is_md


def create_xdatcar_condensed_copy(dir_name, new_xdatcar_name=None,
                                  ionic_step_offset=0, keep_one_every=10,
                                  return_0_based_indexing=False, verbosity=1):
    """
    Create a copy of the VASP XDATCAR output with only a selection of steps*
    to reduce the size of associated trajectory

    WARNING: To be tested with variable-cell trajectory files.

    Args:
        dir_name: str
            directory containing the VASP XDATCAR output file
        new_xdatcar_name: str or None (default is None)
            name of the new (reduced) XDATCAR file
        ionic_step_offset: int (default is 0)
            Number of ionic steps to skip at the beginning
        keep_one_every: int (default is 10)
            To keep only one step out of keep_one_every starting
            from step ionic_step_offset + 1 (one-based indexing)
        return_0_based_indexing: bool (default is False)
            Whether the returned ionic_step_indexes are 0-based.
            If False (the default) the 1-based ionic step indexing used
            in VASP outputs will be used.
        verbosity: int (default is 1)
            Verbosity level.

    Returns:
        new_xdatcar_abspath: str
            Absolute path of the XDATCAR copy. if new_xdatcar_name is None
            it will be set to the absulte pat of DIRNAME/new_xdatcar_name
        ionic_step_indexes: list
            List of ionic steps as given in the original XDATCAR file
            (one-based indexing).
        nb_of_ionic_steps: int
            Total number of ionic steps in the original XDATCAR
    """
    if new_xdatcar_name is None:
        new_xdatcar_name = os.path.abspath(os.path.join(dir_name, 'XDATCAR_reduced'))
    orig_xdatcar_filename = os.path.join(dir_name, 'XDATCAR')
    ionic_step_indexes = []
    with open(new_xdatcar_name, 'w') as outfile, open(orig_xdatcar_filename, 'r') as infile:
        is_header = True
        copy_is_on = True
        for line in infile:
            # copy header, cell and composition
            if is_header:
                if not line.startswith('Direct configuration='):
                    outfile.write(line)
                    continue
                else:
                    is_header = False
            if line.startswith('Direct configuration='):
                ionic_step = int(line.split('=')[1])
                if ionic_step > ionic_step_offset and (
                        (ionic_step - ionic_step_offset - 1) % keep_one_every == 0):
                    copy_is_on = True
                    outfile.write(line)
                    ionic_step_indexes.append(ionic_step)
                    continue
                else:
                    copy_is_on = False
                    continue
            elif copy_is_on:
                outfile.write(line)
    nb_of_ionic_steps_in_orig_xdatcar = ionic_step
    if return_0_based_indexing:
        ionic_step_indexes = [i - 1 for i in ionic_step_indexes]

    if verbosity >= 2:
        print(('A copy containing {} ionic steps of file {} ({} ionic steps) '
               'was created as {}.').format(len(ionic_step_indexes),
              orig_xdatcar_filename, nb_of_ionic_steps_in_orig_xdatcar,
              new_xdatcar_name))

    return os.path.abspath(new_xdatcar_name), ionic_step_indexes, nb_of_ionic_steps_in_orig_xdatcar


def get_nblock_from_incar(dir_name, incar_file_name='INCAR', verbosity=1):
    """
    get nblock parameter from VASP INCAR file.
    """
    incar = Incar().from_file(os.path.join(dir_name, incar_file_name))
    if 'nblock' in incar.keys():
        nblock = incar['nblock'].split()[0]
        if verbosity >= 2 or (verbosity >= 1 and incar['nblock'] > 1):
            print('nblock parameter read from INCAR file: {}'.format(incar['nblock']))
    else:
        nblock = 1
    return nblock


def get_trajectory_from_xdatcar(dir_name, use_displacements=False,
                                constant_lattice=True, get_thermodynamics_data=True,
                                keep_one_every=1, ionic_step_offset=0,
                                new_xdatcar_name=None, nblock=None, verbosity=1):
    """
    Get trajectory from VASP XDATCAR file

    Can be used in situations where the vasprun.xml file cannot be read,
    which is typically the case for unfinished calculations.

    It can also be used to accelerate data analyses in the case
    of large trajectories (number of steps and/or number of atoms).

    TODO: add the possibility to remove the reduced XDATCAR file

    Args:
        dir_name: str
            directory containing the VASP XDATCAR output file
        constant_lattice: bool (default is True)
            Use True for NVT simulations, False otherwise (not tested).
        get_thermodynamics_data: bool (default is True)
            Whether thermodynamics_data read from OSZICAR file should
            be returned.
        ionic_step_offset: int (default is 0)
            Number of ionic steps to skip at the beginning
        keep_one_every: int (default is 1)
            To keep only one step out of keep_one_every starting
            from step ionic_step_offset + 1 (one-based indexing)
            if nblock is not 1 and keep_one_every is 1 it
            will automatically be changed to nblock.
        new_xdatcar_name: str or None (default is None)
            name of the reduced XDATCAR file that will be created
            if ionic_step_offset > 0 or if keep_one_every is not 1
            or equal to nblock. If None it will automatically default
            to dir_name/XDATCAR_reduced.
        nblock: int or None (default is None)
            The parameter equivalent ti keep_one_every in VASP.
            if None this parameter will be read from the INCAR file
            and set to 1 if not found therein. This is the safest option.
        verbosity: int (default is 1)
            Verbosity level.

    Returns:
        traj: pymatgen.core.trajectory.Trajectory instance.
        ionic_step_indexes: list
            0-based ionic step indexes. Add 1 to get the 1-based ionic step indexes
            as used in VASP.
        nb_of_ionic_steps: int
            Nb of ionic steps in the original XDATCAR file. It will be different
            the total number of ionic steps computed if the nblock INCAR paramater
            is not 1.
        thermo_data: dict
            Dictionary containing a list of values the size of ionic_step_indexes
            for each thermodynamic parameter ossociated with ionic steps in the
            OSZICAR file.
        is_md: bool
            Will be True if the OSZICAR file contains temperature information.
    """
    incar = Incar().from_file(os.path.join(dir_name, 'INCAR'))
    if nblock is None:
        if 'nblock' in [k.lower() for k in incar.keys()]:
            nblock = int(incar['nblock'].split()[0])
            if verbosity >= 2 or (verbosity >= 1 and nblock > 1):
                print('nblock parameter read from INCAR file: {}'.format(nblock))
        else:
            nblock = 1

    if keep_one_every == 1 and nblock > 1:
        keep_one_every = nblock
        if verbosity >= 1:
            warnings.warn(('keep_one_every changed to {} to match nblock INCAR '
                           'parameter.').format(keep_one_every))
    elif keep_one_every % nblock != 0:
        raise ValueError(('Keep_one_every should be a multiple of nblock INCAR '
                          'parameter ({})').format(nblock))

    if ionic_step_offset > 0 or (keep_one_every > 1 and keep_one_every != nblock):
        # Make a copy of XDATCAR file
        # TODO: introduce a warning in case 'XDATCAR_reduced' exists in directory
        new_xdatcar_name, ionic_step_indexes, nb_of_ionic_steps = create_xdatcar_condensed_copy(
            dir_name, new_xdatcar_name=new_xdatcar_name, ionic_step_offset=ionic_step_offset,
            keep_one_every=keep_one_every/nblock, return_0_based_indexing=True)
        traj = Trajectory.from_file(new_xdatcar_name, constant_lattice=constant_lattice)
    else:
        traj = Trajectory.from_file(os.path.join(dir_name, 'XDATCAR'), constant_lattice=constant_lattice)
        # WARNING: THE FOLLOWING WILL NOT WORK IF NBLOCK IS NOT ONE
        if nblock == 1:  # ionic_step_offset = 0, keep_one_every = 1
            nb_of_ionic_steps = len(traj)
            ionic_step_indexes = list(range(nb_of_ionic_steps))
        else:
            # TODO: handle this situation if it occurs
            raise Exception(('Situation not currently implemented: ionic_step_offset = {}, '
                             'keep_one_every = {}, nblock = {}.').format(ionic_step_offset,
                            keep_one_every, nblock))

    if 'potim' in [k.lower() for k in incar.keys()]:
        traj.time_step = float(incar['potim'].split()[0])
        if verbosity >= 2:
            print(('Trajectory time_step ({:.2f} fs) read from INCAR '
                   'parameter \'POTIM\'').format(traj.time_step))
    else:
        traj.time_step = 1.0
        if verbosity >= 2:
            print('Trajectory time_step set to default: 1.0 fs.')

    if verbosity >= 3:
        print('Trajectory instance created with properties:')
        for name, value in getmembers(traj):
            if name[0] != '_':
                string = '{} (type: {})'.format(name, type(value))
                if isinstance(value, (list, tuple)):
                    string += ' of length {}'.format(len(value))
                print(string)

    if get_thermodynamics_data:
        thermo_data, is_md = get_energetics_from_vasp_oszicar(dir_name,
            ionic_step_offset=ionic_step_offset, keep_one_every=keep_one_every,
            verbosity=verbosity)
        return traj, ionic_step_indexes, nb_of_ionic_steps, thermo_data, is_md
    else:
        return traj, ionic_step_indexes, nb_of_ionic_steps


def get_energetics_from_vasp_oszicar(dir_name, is_md=None, ionic_step_offset=0,
                                     keep_one_every=1, verbosity=1):
    """
    Get thermodynamics data from VASP OSZICAR file only

    To be use in cases where the vasprun.xml file cannot be read.

    Takes the same keyword arguments as paymatgen.Vasprun,including:
        ionic_step_skip (default: None), ionic_step_offset (default: 0),
        parse_dos (default: False), parse_eigen (default: False),
        parse_potcar_file(default: False), occu_tol (default: 1e-08),
        exception_on_bad_xml (default: True)

    Args:
        dir_name: str
            directory containing the VASP OSZICAR output file
        is_md: bool or None: (default is None)
            Whether VASP data in dir_name correspond to molecular dynamics,
            in which case the temperature will be read from the OSZICAR file.
            If None the temperature will be searched in the OSZICAR file
            and is_md will be set accordingly (True is temperature information
            is found).
        ionic_step_offset: int (default is 0)
            Number of ionic steps to skip at the beginning
        keep_one_every: int (default is 1)
            To keep only one step out of keep_one_every starting
            from step ionic_step_offset + 1 (one-based indexing).
        verbosity: int (default is 1)
            Verbosity level.

    Returns:
        thermo_data: dict
            Dictionary containing a list of values the size of ionic_step_indexes
            for each thermodynamic parameter ossociated with ionic steps in the
            OSZICAR file.
        is_md: bool
            Will be True if the OSZICAR file contains temperature information.
    """
    thermo_data = {}
    oc = Oszicar(os.path.join(dir_name, 'OSZICAR') )
    if is_md is None:
        if 'T' in oc.ionic_steps[0]:
            is_md = True
        else:
            is_md = False
    if is_md is True:
        # Set correspondance between vasprun and ozsicar thermodynamics data
        # TODO: define other energy types and mapping for MD runs other than
        #       NVT with Nose_Hoover thermostat
        energy_mapping = {
            'T': 'temperature',
            'E': 'total',
            'F': 'e_fr_energy',
            'E0': 'e_0_energy',
            'EK': 'kinetic',
            'SP': 'nosepot',
            'SK': 'nosekinetic',
        }
        avail_energy_types = list(oc.ionic_steps[0].keys())
    else:
        energy_mapping = {}  # TO BE COMPLETED
        warnings.warn('Function may not be adapted for non-md runs. Use with caution.')

    for oszicar_energy in avail_energy_types:
        if oszicar_energy in energy_mapping.keys():
            energy_name = energy_mapping[oszicar_energy]
        else:
            energy_name = oszicar_energy
        thermo_data[energy_name] = []
        for ionic_step in range(ionic_step_offset, len(oc.ionic_steps),
                                keep_one_every):
            thermo_data[energy_name].append(oc.ionic_steps[ionic_step][oszicar_energy])

    return thermo_data, is_md


def get_frac_coords_and_velocities_from_xdatcar(dir_name, ionic_step,
        file_name='XDATCAR', use_zero_based_indexing=True,
        time_step=None, verbosity=1):
    """
    Extract coordinates from ionic step N and velocities from N and N-1

    Requires that NBLOCK parameter is set to 1.
    The function currently assumes that direct (i.e. internal
    coordinates are used. Velocities are returned in units of cell_vector/time_step as used in VASP POSCAR when direct
    coordinates are used.

    CURRENTLY IMPLEMENTED FOR CONSTANT-LATTICE TRAJECTORIES ONLY.

    Args:
        TO BE COMPLETED

    Returns:
        frac_coords: numpy.ndarray of floats of shape (nb_of_atoms, 3)
            coords are in units of cell_vector
        velocites: numpy.ndarray of floats of shape (nb_of_atoms, 3)
            in units of cell_vector/time_step
    """
    # Get nblock and time_step from INCAR file:
    incar = Incar().from_file(os.path.join(dir_name, 'INCAR'))
    if 'nblock' in [k.lower() for k in incar.keys()]:
        nblock = int(incar['nblock'].split()[0])
        if verbosity >= 2 or (verbosity >= 1 and nblock > 1):
            print('nblock parameter read from INCAR file: {}'.format(nblock))
    else:
        nblock = 1
    # Time step is not needed if coordinates in XDATCAR file are direct.
    if 'potim' in [k.lower() for k in incar.keys()]:
        time_step = float(incar['potim'].split()[0])
        if verbosity >= 2:
            print(('time step read from POTIM parameter in INCAR '
                   'file: {}').format(time_step))
    else:
        time_step = 1.0

    # Check that nblock is 1 (consecutive ionic steps required)
    if nblock > 1:
        # TODO: implement search for velocities from vasprun.xml
        # This will not work for interrupted calculations
        raise ValueError(('Velocities cannot be obtained from xdatcar '
                          'if nblock > 1 (here {}). Use vasprun.xml file.').format(nblock))
    # Set target step numbers in VASP one-based indexing
    step_N = ionic_step + 1 if use_zero_based_indexing else ionic_step
    if step_N == 1:
        warnings.warn('Velocities for the initial ionic step will be '
                      'set to zero.')
    step_N_frac_coords = []
    step_N_minus_1_frac_coords = []
    with open(os.path.join(dir_name, file_name), 'r') as infile:
        is_header = True
        parsing_N = False
        parsing_N_minus_1 = False
        for line in infile:
            if is_header:
                if line.startswith('Direct configuration='):
                    is_header = False
            if line.startswith('Direct configuration='):
                current_ionic_step = int(line.split('=')[1])
                if current_ionic_step == step_N - 1:
                    parsing_N_minus_1 = True
                    parsing_N = False
                    if verbosity >= 1:
                        print(('Parsing ionic step {} (one-based'
                               ' indexing)').format(step_N - 1))
                    continue
                elif current_ionic_step == step_N:
                    parsing_N_minus_1 = False
                    parsing_N = True
                    if verbosity >= 1:
                        print(('Parsing ionic step {} (one-based'
                               ' indexing)').format(step_N))
                    continue
                elif current_ionic_step > step_N:
                    break
            elif parsing_N:
                step_N_frac_coords.append([float(a) for a in line.split()])
            elif parsing_N_minus_1:
                step_N_minus_1_frac_coords.append([float(a) for a in line.split()])
    
    if len(step_N_frac_coords) and not len(step_N_minus_1_frac_coords): 
        step_N_minus_1_frac_coords = step_N_frac_coords
    
    step_N_frac_coords = np.asarray(step_N_frac_coords)
    step_N_minus_1_frac_coords = np.asarray(step_N_minus_1_frac_coords)
    # Calculating velocities in lattice_vector/time_step units
    # as used in the VASP POSCAR/CONTCAR files
    velocities = step_N_frac_coords - step_N_minus_1_frac_coords

    return step_N_frac_coords, velocities


def export_poscar_file_with_velocity(trajectory_file, ionic_step, 
                                     poscar_file_name, 
                                     use_zero_based_indexing=True,
                                     verbosity=1):
    """
    Export a POSCAR file with velocities from any ionic step in a trajectory file
    
    Currently implemented trajectiry files:
        - XDATCAR
        
    Args:
        TO BE COMPLETED
    Returns:
        Absolute path of the exported POSCAR file
    """
    dir_name = os.path.dirname(trajectory_file)
    if 'xdatcar' in os.path.basename(trajectory_file).lower():
        frac_coords, velocities = get_frac_coords_and_velocities_from_xdatcar(
            dir_name, ionic_step, file_name=os.path.basename(trajectory_file),
            use_zero_based_indexing=use_zero_based_indexing,
            verbosity=verbosity)
        # Reading ref structure from CONTCAR in case species are not read
        ref_structure = Structure.from_file(os.path.join(dir_name, 'CONTCAR'))
        structure = Structure(lattice=ref_structure.lattice , 
            species=[specie.name for specie in ref_structure.species], 
            coords=frac_coords, to_unit_cell=False, coords_are_cartesian=False)
        poscar = Poscar(structure=structure)
        poscar.velocities = velocities
        poscar.write_file(poscar_file_name)
    elif 'vasprun.xml' in s.path.basename(trajectory_file).lower():
        # TODO: calculate velocities from Vasprun
        raise ValueError('Implpementation for vasprun.xml trajectory files in progress.')
    else:
        raise ValueError('Function currently implemented yet for vasprun.xml trajectory files.')
    _poscar_file_name = os.path.abspath(poscar_file_name)
    if verbosity >= 1:
        indexing_str = 'zero-based' if use_zero_based_indexing else 'one-based'
        print(('POSCAR file with velocities exported from ionic step {} '
               '({} indexing) as {}').format(ionic_step, indexing_str, _poscar_file_name))
    return _poscar_file_name
    

        