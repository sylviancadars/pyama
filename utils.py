from ase.visualize import view
from ase.io import read
from ase import Atoms

from pymatgen.core.structure import Structure, IStructure, Molecule, IMolecule
from pymatgen.core.lattice import Lattice
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.units import Energy, Length, FloatWithUnit
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import os
from subprocess import call
import sys
import numpy as np


def is_float(a_string):
    """
    Helper function to check if a string is a float
    """
    try:
        float(a_string)
        return True
    except ValueError:
        return False


class BaseDataClass():
    def __init__(self, verbosity=1, print_to_console=True, print_to_file=False,
                 output_text_file='output.txt'):
        self.verbosity = verbosity
        self.print_to_console = print_to_console
        self.print_to_file = print_to_file
        self.output_text_file = output_text_file

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

    def write_to_output_file(self, reinitialize_output_text=True):
        with open(self.output_text_file, 'a') as f:
            f.write(self._output_text)
        if reinitialize_output_text:
            self._output_text = []


def get_pymatgen_structure(struct_or_file, struct_description=None):
    if isinstance(struct_or_file, str):
        structure = Structure.from_file(struct_or_file)
        if struct_description is None:
            base_name = os.path.basename(struct_or_file)
            if base_name.lower() in ('poscar', 'contcar'):
                file_name = os.path.join(os.path.dirname(struct_or_file), base_name)
            else:
                file_name = base_name
            struct_description = '{} structure loaded from file {}'.format(structure.formula,
                                                                           file_name)
    elif isinstance(struct_or_file, Atoms):
        structure = AseAtomsAdaptor.get_structure(struct_or_file)
        if struct_description is None:
            struct_description = '{} structure loaded from ASE Atoms'.format(
                structure.formula)
    elif isinstance(struct_or_file, Structure):
        structure = struct_or_file
        if struct_description is None:
            struct_description = '{} pymatgen structure'.format(structure.formula)
        # TODO: Convert IStructure to Structure
    else:
        raise TypeError('get_pymatgen_structure function does not work for type {}.'.format(
            type(struct_or_file)))
    return structure, struct_description


def get_ase_atoms(struct_or_file, struct_description=None,
                  return_description=False):
    """
    Get an ASE Atoms object from a file or pymatgen Structure or Molecule

    Atoms object will be left untouched.

    Args:
        struct_or_file: pymatgen (I)Structure or (I)Molecule, str
            structure or file name to be converted to ASE atoms
        struct_description: str or None (default is None)
            If None a description will be created automatically using formula
            and origin.
        return_description: bool (default is False).
            If True the function will return a tuple of ASE Atoms and
            description

    Returns:
        ASE Atoms
        description: str
            if return_description is True.
    """
    if struct_description is not None and not isinstance(struct_description, str):
        raise TypeError('struct_description shoulf be a str of None')
    if isinstance(struct_or_file, str):
        atoms = read(struct_or_file)
        if struct_description is None:
            base_name = os.path.basename(struct_or_file)
            if base_name.lower() in ('poscar', 'contcar'):
                file_name = os.path.join(os.path.dirname(struct_or_file), base_name)
            else:
                file_name = base_name
            description = '{} from file {}'.format(
                atoms.get_chemical_formula(), file_name)
        elif isinstance(struct_description, str):
            description = struct_description
        # TODO: add exceptions for file formats unknown to ASE
    elif isinstance(struct_or_file, (Structure, IStructure, Molecule, IMolecule)):
        aaa = AseAtomsAdaptor()
        atoms = aaa.get_atoms(struct_or_file)
        if struct_description is None:
            description = '{} from pymatgen Structure or Molecule.'.format(
                atoms.get_chemical_formula())
    elif isinstance(struct_or_file, Atoms):
        atoms = struct_or_file
        if struct_description is None:
            description = '{}'.format(
                atoms.get_chemical_formula())
    else:
        raise TypeError('struct_or_file should be a pymatgen structure or molecule, ASE Atoms or file name.')

    if return_description:
        return atoms, description
    else:
        return atoms


def visualize_structure(struct_or_file, viewer='ase'):

    tmp_file_name = None
    if isinstance(struct_or_file, str):
        if os.path.isfile(struct_or_file):
            if viewer.lower() == 'ase':
                view(read(struct_or_file))
            elif viewer.lower() == 'vesta':
                call(['vesta', struct_or_file])
        else:
            sys.exit('No file with name {}'.format(struct_or_file))

    elif isinstance(struct_or_file,
                    (Structure, IStructure, Molecule, IMolecule)):
        struct = struct_or_file
        if viewer.lower() == 'ase':
            aaa = AseAtomsAdaptor()
            view(aaa.get_atoms(struct))
        if viewer.lower() == 'vesta':
            tmp_file_name = os.path.join(os.getcwd(), '.tmp.cif')
            struct.to(filename=tmp_file_name, fmt='cif')
            call(['vesta', tmp_file_name])

    if tmp_file_name is not None:
        if os.path.isfile(tmp_file_name):
            os.remove(tmp_file_name)

def convert_dispersion_coeff(coeff_value, coeff_name='C_6',
                             input_energy_unit='J',
                             input_length_unit='nm',
                             input_natoms_unit='mol',
                             output_energy_unit='Ha',
                             output_length_unit='bohr',
                             output_natoms_unit='atom',
                             return_unit=False, verbosity=0):
    """
    Convert c6 dispersion coefficient

    [in/out]put_[energy/length/natoms]_unit parmaters should use the same
    units as in pymatgen.core.units, exept for nanometers
    """
    E = Energy(1, input_energy_unit)
    if input_length_unit == 'nm':
        l = Length(1e-9, 'm')
    else:
        l = Length(1, input_length_unit)
    N = FloatWithUnit(1, unit=input_natoms_unit)
    if output_length_unit == 'nm':
        l_out = 1e9 * l.to('m')
    else:
        l_out = l.to(output_length_unit)

    if coeff_name in ['C_6', 'C6']:
        output_coeff_value = coeff_value * ( E.to(output_energy_unit) *
            pow(l_out, 6) / N.to(output_natoms_unit) )
        output_unit = ''.join([output_energy_unit, ' ', output_length_unit,
                               '^6 ', output_natoms_unit, '^-1'])

    if verbosity >= 1:
        print('Converted dispersion coefficient {} {}'.format(
              output_coeff_value, output_unit))
    if return_unit:
        return float(output_coeff_value), output_unit
    else:
        return float(output_coeff_value)

def concatenate_trajectories(traj_1: Trajectory,
                             traj_2: Trajectory):
    """
    Concatenate two pymatgen.core.trajectory.Trajectory instances

    This function was created to overcome a bug in Trajectory.extend()
    method detected while trying to extend trajectories created
    from pymatgen.io.vasp.Vasprun instances with the to_trajectory()
    method.

    Args:
        traj_1:
        traj_2:

    Returns
        concatenated_trajectory
    """
    if traj_1.species != traj_2.species:
        raise ValueError('Trajectories to concatenate have different \'species\' properties.')
    if traj_1.time_step != traj_2.time_step:
        raise ValueError('Trajectories to concatenate have different \'time_step\' properties.')
    concatenated_traj = traj_1.copy()
    concatenated_traj.site_properties += traj_2.site_properties
    # Convert both trajectories to positions before concatenation
    concatenated_traj.to_positions()
    _traj_2 = traj_2.copy()
    _traj_2.to_positions()
    concatenated_traj.frac_coords += _traj_2.frac_coords
    if not all([traj_1.constant_lattice, traj_2.constant_lattice]):
        concatenated_traj.constant_lattice = False

    if isinstance(traj_1.lattice, list) and isinstance(traj_2.lattice, list):
        concatenated_traj.lattice += traj_2.lattice
    elif isinstance(traj_1.lattice, np.ndarray) and isinstance(traj_2.lattice, np.ndarray):
        concatenated_traj.lattice = np.append(concatenated_traj.lattice,
                                              traj_2.lattice, axis=0)
    if isinstance(traj_1.frame_properties, list) and isinstance(
            traj_2.frame_properties, list):
        concatenated_traj.frame_properties += traj_2

def sort_structure_by_frac_coord(structure_or_file, axis=2, reverse=False):
    """
    Return a copy of a structure with sites ordered along a chosen crystallographic axis

    Args:
        structure_or_file: str, pymatgen Structure or ase.atoms.Atoms
            files should be format known by pymatgen
        axis: int (default is 2)
            axis (from 0 to 2) along which sites should be sorted
        reverse: bool (default is False)

    Returns:
        sorted structure
    """
    (structure, description) = get_pymatgen_structure(structure_or_file)
    structure.sort(lambda site: site.frac_coords[axis], reverse=reverse)
    return structure

def get_pymatgen_struct_from_xyz(filename, lattice_or_cell_length=None,
                                 get_cell_length_from_xyz_comment=True,
                                 verbosity=0):
    """
    Get a pymatgen structure from an xyz file

    Args:
        filename: str
            path to xyz file.
        lattice_or_cell_length: float, list, tuple, Lattice, or None (default is None)
            cell length (cubic cell) , list or tuple of lattice vectors,
            or pymatgen.core.Lattice. If known function returns a Molecule object


    Returns:
        pymatgen Structure or Molecule (if lattice_or_cell_length is None)
    """
    pmg_mol = Molecule.from_file(filename)
    if lattice_or_cell_length is None:
        return pmg_mol
    if isinstance(lattice_or_cell_length, float):
        lattice = [[lattice_or_cell_length, 0., 0.],
                    [0., lattice_or_cell_length, 0.],
                    [0., 0., lattice_or_cell_length]]
    elif isinstance(lattice_or_cell_length, Lattice):
        lattice = lattice_or_cell_length
    else:
        lattice = Lattice(lattice_or_cell_length)
    if verbosity > 0:
        print('Trying to build structure with lattice:\n', lattice)
    pmg_struct = pmg_struct = Structure(lattice, pmg_mol.species,
                                        coords_are_cartesian=True,
                                        coords=pmg_mol.cart_coords)
    return pmg_struct


def convert_xyz_to(xyz_file_name, export_file_name, lattice_or_cell_length,
                   fmt=None, verbosity=0):
    """
    Convert xyz file to a structure file using pymatgen

    Args:
        xyz_file_name: str
            path to input xyz file.
        export_file_name: str
            path to output file. Extension will be used to identify format
            if fmt is None (as in pymatgen Structure.to() method).
        lattice_or_cell_length: float, list, tuple, Lattice, or None (default is None)
            cell length (cubic cell) , list or tuple of lattice vectors,
            or pymatgen.core.Lattice. If known function returns a Molecule object
        fmt: str or None (default is None):
            Export file format: e.g. 'cif', 'poscar', etc. Non case-sensitive.
            (see known formats in pymatgen Structure.to() method)

    Returns:
        file name or None
    """

    pmg_struct = get_pymatgen_struct_from_xyz(xyz_file_name, lattice_or_cell_length)
    pmg_struct.to(fmt=fmt, filename=export_file_name)
    if verbosity > 0:
        print('Structure file stored as {}.'.format(export_file_name))


def visualize_clusters(structure, site_indexes, r_cut=4.0, verbosity=1):
    """
    TO BE COMPLETED
    """
    def get_type(site):
        return site.species.elements[0].name

    """
    if folder is None:
        folder = os.path.join(os.getcwd(), 'clusters')
    try:
        os.mkdir(clusters_dir)
    except FileExistsError:
        pass
    """

    clusters = {'center_types': [], 'center_indexes': [], 'molecules': []}
    for site_index in site_indexes:
        site = structure.sites[site_index]
        site_type = get_type(site)
        if verbosity >= 3:
            print('Extracting cluster from site {}: {}'.format(site_index, site_type))
        nbrs = structure.get_neighbors(site, r_cut)
        if verbosity >= 3:
            print('Site {} of type {} has {} neighbors within {} \u212B'.format(
                site_index, site_type, len(nbrs), r_cut))

        cluster = Molecule(species=[site.species] + [nbr.species for nbr in nbrs],
                           coords=[site.coords] + [nbr.coords for nbr in nbrs])

        visualize_structure(cluster)


def get_indexes_by_type(structure_or_atoms, verbosity=0):
    """
    Get a dictionary of indexes by type in a pymatgen (I)Structure or ASE Atoms

    Args:
        structure_or_atoms: pymatgen Structure, IStructure, ASE Atoms
            The structure to be considered
        verbosity:
            verbosity level
    Returns:
        indexes_by_type: dict
            A dictionary representing with atom types as keys and lists of
            site indexes as values.
    """

    if isinstance(structure_or_atoms, Atoms):
        atoms = structure_or_atoms
    else:
        atoms = get_ase_atoms(structure_or_atoms)

    atom_types = list(set(atoms.get_chemical_symbols()))
    indexes_by_type = {atom_type: [] for atom_type in atom_types}
    for atom_type in atom_types:
        for site_index, atom in enumerate(atoms):
            if atom.symbol == atom_type:
                indexes_by_type[atom_type].append(site_index)

    if verbosity >= 1:
        print('indexes_by_type = ', indexes_by_type)

    return indexes_by_type


def is_unitary_matrix(m, **kwargs):
    """
    Test whether a matrix is unitary with numpy

    Args:
        m: numpy darray
            array to test. Should of square shape.
        kwargs:
            keyword arguments that should be passed to numpy.allclose

    Returns:
        bool
    """
    return np.allclose(m, m.dot(m.T.conj()), **kwargs)


def has_tetragonal_shape(cell_angles, cell_lengths,
                        return_unique_axis=False, **kwargs,):
    """
    Determines if a crystal cell is tetragonal based on its angles and lengths.

    IMPORTANT: Symmetry is completely ignored here.

    A cell is considered tetragonal if:
    - All three angles are close to 90°.
    - Two cell lengths are equal, and the third is different.

    Parameters:
    -----------
    cell_angles : list or tuple of 3 float
        The angles (in degrees) between the axes of the crystal cell. The angles should be [alpha, beta, gamma].

    cell_lengths : list or tuple of 3 float
        The lengths of the crystal axes (a, b, c).

    return_unique_axis : bool, optional, default=False
        If True, the function returns both a boolean indicating if the cell is tetragonal,
        and the unique axis ('a', 'b', or 'c') that is different from the others.
        If False, only the tetragonal status is returned.

    **kwargs : additional keyword arguments
        Additional arguments to be passed to `np.isclose`, such as `atol` and `rtol`.

    Returns:
    --------
    bool or tuple
        - If `return_unique_axis=False`, returns a boolean indicating if the cell is tetragonal.
        - If `return_unique_axis=True`, returns a tuple `(is_tetragonal, unique_axis)` where:
          - `is_tetragonal` is a boolean indicating if the cell is tetragonal.
          - `unique_axis` is either 'a', 'b', 'c', or `None` (if the cell is not tetragonal).

    Example:
    --------
    >>> is_tetragonal([90, 90, 90], [5.0, 5.0, 10.0])
    True

    >>> is_tetragonal([90, 90, 90], [5.0, 6.0, 10.0])
    False

    >>> is_tetragonal([90, 90, 90], [5.0, 5.0, 10.0], return_unique_axis=True)
    (True, 'c')
    """
    print('WARNING. FUNCTION NEEDS TO BE TESTED.')

    # Unpack angles and lengths (assuming cell_angles is in degrees)
    alpha, beta, gamma = cell_angles  # Angles in degrees
    a, b, c = cell_lengths  # Lengths of the axes

    # Check if the angles match the tetragonal criteria: all three 90°
    tetragonal = False
    for perm in cyclic_permutations(cell_angles):
        alpha, beta, gamma = perm

        # Check if two angles are 90° and one is 120°
        if (np.isclose(alpha, 90, **kwargs) and np.isclose(beta, 90, **kwargs)
            and np.isclose(gamma, 90, **kwargs)):
                tetragonal = True

    if not tetragonal:
        # If it's not tetragonal, return False and None for unique axis
        if return_unique_axis:
            return False, None
        else:
            return False

    # If return_unique_axis is True, find and return the unique axis
    if return_unique_axis:
        if np.isclose(a, b, **kwargs) and not np.isclose(a, c, **kwargs):
            return True, 'c'  # 'c' is unique
        elif np.isclose(a, c, **kwargs) and not np.isclose(a, b, **kwargs):
            return True, 'b'  # 'b' is unique
        elif np.isclose(b, c, **kwargs) and not np.isclose(a, b, **kwargs):
            return True, 'a'  # 'a' is unique
        return True, None  # If no unique axis is found, which shouldn't happen in tetragonal cells

    # If return_unique_axis is False, just return whether it's tetragonal
    return True


def has_hexagonal_shape(cell_angles, cell_lengths,
                        return_unique_axis=False, **kwargs,):
    """
    Determines if a crystal cell is hexagonal based on its angles and lengths.

    IMPORTANT: Symmetry is completely ignored here.

    A cell is considered hexagonal if:
    - Two angles are close to 90° and one is close to 120°.
    - Two cell lengths are equal, and the third is different.

    Parameters:
    -----------
    cell_angles : list or tuple of 3 float
        The angles (in degrees) between the axes of the crystal cell. The angles should be [alpha, beta, gamma].

    cell_lengths : list or tuple of 3 float
        The lengths of the crystal axes (a, b, c).

    return_unique_axis : bool, optional, default=False
        If True, the function returns both a boolean indicating if the cell is hexagonal,
        and the unique axis ('a', 'b', or 'c') that is different from the others.
        If False, only the hexagonal status is returned.

    **kwargs : additional keyword arguments
        Additional arguments to be passed to `np.isclose`, such as `atol` and `rtol`.

    Returns:
    --------
    bool or tuple
        - If `return_unique_axis=False`, returns a boolean indicating if the cell is hexagonal.
        - If `return_unique_axis=True`, returns a tuple `(is_hexagonal, unique_axis)` where:
          - `is_hexagonal` is a boolean indicating if the cell is hexagonal.
          - `unique_axis` is either 'a', 'b', 'c', or `None` (if the cell is not hexagonal).

    Example:
    --------
    >>> is_hexagonal([90, 90, 120], [5.0, 5.0, 10.0])
    True

    >>> is_hexagonal([90, 90, 120], [5.0, 6.0, 10.0])
    False

    >>> is_hexagonal([90, 90, 120], [5.0, 5.0, 10.0], return_unique_axis=True)
    (True, 'c')
    """
    print('WARNING. FUNCTION NEEDS TO BE TESTED.')

    # Unpack angles and lengths (assuming cell_angles is in degrees)
    alpha, beta, gamma = cell_angles  # Angles in degrees
    a, b, c = cell_lengths  # Lengths of the axes

    # Check if the angles match the hexagonal criteria: two 90° and one 120°
    hexagonal = False
    for perm in cyclic_permutations(cell_angles):
        alpha, beta, gamma = perm

        # Check if two angles are 90° and one is 120°
        if np.isclose(alpha, 90, **kwargs) and np.isclose(beta, 90, **kwargs):
            if np.isclose(gamma, 120, **kwargs):
                # Gamma is the 120° angle, so check if a ≈ b
                if np.isclose(a, b, **kwargs):
                    hexagonal = True
            elif np.isclose(alpha, 120, **kwargs):
                # Alpha is the 120° angle, so check if b ≈ c
                if np.isclose(b, c, **kwargs):
                    hexagonal = True
            elif np.isclose(beta, 120, **kwargs):
                # Beta is the 120° angle, so check if a ≈ c
                if np.isclose(a, c, **kwargs):
                    hexagonal = True

    if not hexagonal:
        # If it's not hexagonal, return False and None for unique axis
        if return_unique_axis:
            return False, None
        else:
            return False

    # If return_unique_axis is True, find and return the unique axis
    if return_unique_axis:
        if np.isclose(a, b, **kwargs) and not np.isclose(a, c, **kwargs):
            return True, 'c'  # 'c' is unique
        elif np.isclose(a, c, **kwargs) and not np.isclose(a, b, **kwargs):
            return True, 'b'  # 'b' is unique
        elif np.isclose(b, c, **kwargs) and not np.isclose(a, b, **kwargs):
            return True, 'a'  # 'a' is unique
        return True, None  # If no unique axis is found, which shouldn't happen in hexagonal cells

    # If return_unique_axis is False, just return whether it's hexagonal
    return True


def cyclic_permutations(lst, direction='right'):
    """
    Return a list of all cyclic permutations of lst, starting with lst itself.

    Args:
        lst: list or tuple
        direction: 'right' for right cyclic permutations (default), 'left' for left cyclic permutations

    Returns:
        A list of lists or a list of tuples.
    """
    permutations = [lst[:]]  # Add the original list as the first permutation

    for i in range(1, len(lst)):
        if direction == 'left':
            lst = lst[-1:] + lst[:-1]  # Perform a right cyclic shift
        elif direction == 'right':
            lst = lst[1:] + lst[:1]  # Perform a left cyclic shift
        else:
            raise ValueError("Direction must be either 'right' or 'left'.")

        permutations.append(lst[:])  # Add a copy of the shifted list

    return permutations


def get_cell_lengths_and_angles(structure_or_lattice):
    if isinstance(structure_or_lattice, (Structure, IStructure)):
        cell_lengths_and_angles = structure_or_lattice.lattice.lengths + \
            structure_or_lattice.lattice.angles
    elif isinstance(structure_or_lattice, Lattice):
        cell_lengths_and_angles = structure_or_lattice.lengths + \
            structure_or_lattice.angles
    elif isinstance(structure_or_lattice, (list, tuple, np.ndarray)):
        if len(structure_or_lattice) == 6:
            cell_lengths_and_angles = list(structure_or_lattice)
        if len(structure_or_lattice) == 3:
            lattice = Lattice(structure_or_lattice)
            cell_lengths_and_angles = lattice.lengths + lattice.angles
    else:
        raise TypeError('structure_or_lattice should be a pymatgen Structure, '
                        'Lattice, 3x3 matrix, or list of 6 cell parameters.')
    return cell_lengths_and_angles


def get_lattice_system_with_unique_param(structure_or_lattice,
                                       return_as_dict=True,
                                       verbosity=1, **kwargs,):
    """
    Determines if a crystal cell is monoclinic based on its angles and lengths.

    IMPORTANT: Symmetry is completely ignored here.

    A cell is considered monoclinic if:
    - Two angles are close to 90°, and the third different from 90°.
    - Two cell lengths are equal, and the third is different.

    Parameters:
    -----------
    cell_lengths : list or tuple of 3 float
        The lengths of the crystal axes (a, b, c).

    cell_angles : list or tuple of 3 float
        The angles (in degrees) between the axes of the crystal cell. The angles should be [alpha, beta, gamma].

    return_unique_param : bool, optional, default=False
        If True, the function returns both the lattice_system and
        the unique axis or angle that is different from the others.
        If False, only the lattice_system is returned.

    **kwargs : additional keyword arguments
        Additional arguments to be passed to `np.isclose`,
        such as `atol` and `rtol`.

    Returns:
    --------
    str or tuple
        - If `return_unique_param=False`, returns a boolean indicating if the cell is monoclinic.
        - If `return_unique_param=True`, returns a tuple `(is_monoclinic, unique_param)` where:
          - `is_monoclinic` is a boolean indicating if the cell is monoclinic.
          - `unique_axis` is either 'a', 'b', 'c', or `None` (if the cell does not have a unique parameter).

    Example:
    --------
    TO BE COMPLETED
    """
    cell_lengths_and_angles = get_cell_lengths_and_angles(structure_or_lattice)
    cell_lengths = cell_lengths_and_angles[:3]
    cell_angles = cell_lengths_and_angles[3:]

    # TOBETESTED... order between hexagonal and thombohedral is not obvious
    lattice_systems_hierarchy = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal',
                                 'rhombohedral', 'monoclinic', 'triclinic']

    cell_change_perm = np.array(cyclic_permutations(
        [[0, 0, 1], [0, 1, 0], [0, 1, 0]], direction='left'))

    possible_lattice_systems = []
    for i, (lengths_perm, angles_perm) in enumerate(
            zip(cyclic_permutations(cell_lengths),
                cyclic_permutations(cell_angles))):

        unique_param = None

        a, b, c = lengths_perm
        alpha, beta, gamma = angles_perm

        _swap_axes_matrix = cell_change_perm[i]

        if verbosity >= 2:
            print('Permutation index {}: cell lengths {} and angles {}'.format(
                i, lengths_perm, angles_perm))

        lattice_system = get_lattice_system_from_cell_parameters(
            a, b, c, alpha, beta, gamma, verbosity=0, **kwargs)

        # Define the unique parameter (axis or angle)
        if lattice_system in ['triclinic', 'orthorhombic'] :
            possible_lattice_systems.append({
                'name': lattice_system,
                'unique_param': None,
                'swap_axes_matrix': _swap_axes_matrix})

        elif lattice_system in ['hexagonal', 'tetragonal', 'monoclinic']:
            perm_index = i
            if lattice_system in ['hexagonal',  'tetragonal']:
                unique_param = ('c', 'a', 'b')[perm_index]
            else:
                unique_param = ('β', 'γ', 'α')[perm_index]

            if verbosity >= 1:
                print(('Axes permutation index {}: Cell parameters are '
                       'compatible with a {}-unique {} lattice.').format(
                    perm_index, unique_param, lattice_system))

            possible_lattice_systems.append({
                'name': lattice_system,
                'unique_param': unique_param,
                'swap_axes_matrix': _swap_axes_matrix})

        else:
            # Cubic or rhombohedral crystal systems cannot be mistaken
            # for others. Exit.
            print(('Axes permutation index {}: Cell parameters are '
                       'compatible with a {} lattice.').format(
                    i, lattice_system))

            if return_as_dict:
                return {'lattice_system': lattice_system,
                        'unique_parameter': unique_param,
                        'swap_axes_matrix': _swap_axes_matrix}
            else:
                return lattice_system

    if verbosity >= 2:
        print('possible_lattice_systems = ', possible_lattice_systems)

    # Choose lattice system with best rank in the hierarchy.
    lowest_lattice_index = len(lattice_systems_hierarchy)
    unique_param = None
    for _lattice_system in possible_lattice_systems:
        i = lattice_systems_hierarchy.index(_lattice_system['name'])
        if i < lowest_lattice_index:
            lattice_system = _lattice_system['name']
            unique_param = _lattice_system['unique_param']
            swap_axes_matrix = _lattice_system['swap_axes_matrix']
            lowest_lattice_index = i

    if verbosity >= 1:
        print(('The highest-symmetry lattice system compatible with '
               'the provided lattice is {}{}.').format(
            lattice_system,
            ' ({}-unique)'.format(unique_param) if unique_param else ''))

    if return_as_dict:
        return {'lattice_system': lattice_system,
                'unique_param': unique_param,
                'swap_axes_matrix': swap_axes_matrix}
    else:
        return lattice_system


def get_lattice_system_from_cell_parameters(a, b, c, alpha, beta, gamma,
                                            verbosity=1, **kwargs):
    """
    Determine the lattice type (e.g., cubic, tetragonal, etc.) based on cell parameters,
    with resilience to numerical precision errors.

    cell should be changed to standardized conventional before passing
    lengths and angles.

    Args:
        a, b, c: Lattice lengths
        param alpha, beta, gamma: Lattice angles
        **kwargs: Additional keyword arguments passed to np.isclose or np.allclose
    Returns:
        Lattice type as a string (e.g., 'cubic', 'tetragonal', etc.)
    """
    # Orthogonal cell (alpha = beta = gamma = 90)
    if np.allclose([alpha, beta, gamma], [90.0, 90.0, 90.0], **kwargs):
        if np.isclose(a, b, **kwargs) and np.isclose(b, c, **kwargs):
            lattice_system = 'cubic'  # a = b = c
        elif np.isclose(a, b, **kwargs) and not np.isclose(a, c, **kwargs):
            lattice_system = 'tetragonal'  # (a = b ≠ c)
        else:  # Orthorhombic (a ≠ b ≠ c)
            lattice_system = 'orthorhombic'
    # Hexagonal (alpha = beta = 90, gamma = 120)
    elif (np.allclose([alpha, beta, gamma], [90.0, 90.0, 120.0], **kwargs)
          and np.isclose(a, b, **kwargs)):
        lattice_system = 'hexagonal'
    elif (np.isclose(a, b, **kwargs) and np.isclose(b, c, **kwargs)
          and np.isclose(alpha, beta, **kwargs)
          and np.isclose(beta, gamma, **kwargs)):
        lattice_system = 'rhombohedral'
    # Monoclinic (c-unique)
    elif (np.allclose([alpha, gamma], [90.0, 90.0], **kwargs)
          and not np.isclose(beta, 90.0, **kwargs)):
        lattice_system = 'monoclinic'
    elif (not np.isclose(alpha, 90.0, **kwargs)
          and  not np.isclose(beta, 90.0, **kwargs)
          and not np.isclose(gamma, 90.0, **kwargs)):
        lattice_system = 'triclinic' # Triclinic (no 90-degree angles)
    else:
        if verbosity >= 1:
            print('Lattice type could not be identified from cell paramaters '
                  '{}. Assuming triclinic.'.format((a, b, c, alpha, beta, gamma)))
        lattice_system = 'triclinic'

    return lattice_system


def get_lattice_system_from_sym_or_cell(structure, return_as_dict=True,
                                        verbosity=1, **kwargs):
    """
    Get the lattice system of a pymatgen structure based on its internal symmetries or its cell paramaters

    Symmetry search uses pymatgen SpaceGroupAnalyzer, which usess spglib.

    Args:
        structure: pymatgen Structure
            initial structure. Note that symmetry search can use the
            conventional standard cell instead.

        return_as_dict

        **kwargs:
            keyward arguments that should be passed to numpy
            isclose and allclose functions (atol, rtol, etc.)

    Returns:
        lattice_system:
        is_conventional: bool
    """

    # Convert to conventional when relevant.
    sga = SpacegroupAnalyzer(structure)

    # Inititialize crystal_systrm_dict values
    unique_param = None
    swap_axes_matrix = None
            
    if sga.get_space_group_number() == 1:
        _structure = structure
        change_to_conventional = False

    """
    TODO: better yet:
        If space group is not P1:
            1. create primitive and conventional cell
            2. check whether (a) primitive or (b) conventional satisfy
               lattice system cell angles
            3. if (2a) or (2b) is True, recommand using the 
               corresponding structure (in this priority order).
            
    """

    else:
        conventional_structure = sga.get_conventional_standard_structure()
        # TODO: change_to_conventional should only be applied for cases
        #       where it is essential to keep the right angles fixed
        #       based on Bravais lattice.
        change_to_conventional = not np.allclose(conventional_structure.lattice.matrix,
                                                  structure.lattice.matrix, **kwargs)

        lattice_system = sga.get_lattice_type()
        
        """
        TODO: In the case of a Rhombohedral lattice system, the conventional cell
        is similar the hexagonal system and identified as such by the 
        get_lattice_system_from_cell_parameters function. This means that in the particular case of this system, we should not switch to conventional cell.
        
        if lattice_system == 'rhombohedral':
            change_to_conventional = True only if vol(conventional) vol()
            
        """
        if lattice_system in ['cubic', 'rhombohedral']:
            
            lattice_system_dict = {
                'lattice_system': lattice_system,
                'change_to_primitive': True,
                'change_to_conventional': False,
                'unique_param': None,
                'swap_axes_matrix': None,
            }
            
            if verbosity >= 1:
                print(('{}-atom {} structure has a {} Bravais lattice '
                       'system based on pymatgen SpacegroupAnalyzer. Primitive'
                       'cell should be use to impose corresponding cell constraints.'
                       ).format(structure.num_sites, structure.formula, 
                                ))
            
            return lattice_system_dict if return_as_dict else lattice_system

        # TODO: check whether cubic lattice should also use the primitive cell.
         
        elif lattice_system != 'triclinic':
            if verbosity >= 1:
                print('Lattice type {} identified with pymatgen SpacegroupAnalyzer'.format(
                    lattice_system))
            
            if lattice_system in ['hexagonal',  'tetragonal', 'monoclinic']: 
                swap_axes_matrix_perm = cyclic_permutations([[1, 0, 0], 
                                                             [0, 1, 0],
                                                             [0, 0, 1]])
                lengths_perm = cyclic_permutations(conventional_structure.lattice.lengths)
                angles_perm = cyclic_permutations(conventional_structure.lattice.angles)
                
                for perm_index in range(3):
                    a, b, c = lengths_perm[perm_index]
                    alpha, beta, gamma = angles_perm[perm_index]
                    if (lattice_system in ['hexagonal', 'tetragonal']
                        and np.isclose(a, b, **kwargs) 
                        and not np.isclose(a, c, **kwargs)):
                        unique_param = ['c', 'a', 'b'][perm_index]
                        swap_axes_matrix = [perm_index]
                        break
                    elif (lattice_system == 'monoclinic'
                          and np.isclose(alpha, gamma, **kwargs)
                          and not np.isclose(alpha, beta, **kwargs)):
                        unique_param = ['beta', 'gamma', 'alpha'][perm_index]
                        swap_axes_matrix = [perm_index]
                        break
                                    
            lattice_system_dict = {
                'lattice_system': lattice_system,
                'change_to_conventional': change_to_conventional,
                'unique_param': unique_param,
                'swap_axes_matrix': swap_axes_matrix,
            }

            if verbosity >= 2:
                print('lattice_system_dict = ', lattice_system_dict)

            return lattice_system_dict if return_as_dict else lattice_system

        if verbosity >= 2:
            print('Pymatgen SpacegroupAnalyzer identified the lattice type as triclinic. '
                  'Lattice parameters will be used to further explore lattice types.')

        _structure = structure if not change_to_conventional else conventional_structure

    if change_to_conventional and verbosity >= 1:
        print(('{}-atom {} conventional cell representation (vs {} in initial '
            'cell) will be used to identify the lattice type.').format(
                _structure.num_sites, structure.num_sites, _structure.formula))

    if verbosity >= 2:
        print('Looking for Bravais lattice system based on cell '
              'parameters : {}'.format(
            _structure.lattice.lengths + _structure.lattice.angles))

    lattice_system_dict = get_lattice_system_with_unique_param(
        _structure, return_as_dict=True, verbosity=verbosity, **kwargs)

    lattice_system_dict['change_to_conventional'] = change_to_conventional

    if verbosity >= 1:
        print(('Lattice type has been identified as {} based on '
               'cell parameters.').format(
            lattice_system_dict['lattice_system']))

    if verbosity >= 2:
        print('lattice_system_dict = ', lattice_system_dict)

    if return_as_dict:
        return lattice_system_dict
    else:
        return lattice_system_dict['lattice_system']


def permutate_structure(structure, perm_matrix, in_place=False):
    """
    Permutate the cell parameters and atomic coordinates of a pymatgen Structure
    based on a given permutation matrix. Optionally modifies the structure in place.

    Args:
        structure (Structure): pymatgen Structure object to permutate.
        perm_matrix (np.ndarray): 3x3 permutation matrix to apply.
        in_place (bool): If True, modify the structure in place. Otherwise, return a new structure.

    Returns:
        Structure: The permutated pymatgen Structure object (modified in place if in_place=True).
    """
    # Ensure the permutation matrix is a 3x3 numpy array
    perm_matrix = np.array(perm_matrix)
    if perm_matrix.shape != (3, 3):
        raise ValueError("Permutation matrix must be 3x3.")

    # Get the current lattice (cell) matrix of the structure
    lattice = structure.lattice.matrix

    # Apply the permutation matrix to the lattice
    new_lattice = np.dot(perm_matrix, lattice)

    # Now permutate the atomic positions by applying the same permutation
    new_positions = np.dot(perm_matrix, structure.frac_coords.T).T

    if in_place:
        # Modify the original structure in place
        structure.lattice = new_lattice
        structure.frac_coords = new_positions
        return None  # Nothing is returned when modifying in place
    else:
        # Return a new structure with the updated lattice and atomic positions
        new_structure = Structure(new_lattice, structure.species, new_positions,
                                  coords_are_cartesian=False)
        return new_structure


"""
def determine_ibrav(structure: StructureData):
    Determines the ibrav value based on the structure's cell parameters (lengths and angles).
    First, ensures that the structure uses the conventional representation.

    :param structure: AiiDA StructureData object
    :return: The corresponding ibrav integer value for pw.x

    # Convert AiiDA StructureData to pymatgen Structure object
    pymatgen_structure = Structure.from_sites(structure.sites)

    # Standardize the cell to its conventional representation
    pymatgen_structure = pymatgen_structure.get_conventional_standard_structure()

    # Get the lengths (a, b, c) and angles (alpha, beta, gamma)
    lengths = pymatgen_structure.lattice.abc
    angles = pymatgen_structure.lattice.angles

    a, b, c = lengths
    alpha, beta, gamma = angles

    # Mapping cell parameters to ibrav
    if alpha == 90.0 and beta == 90.0 and gamma == 90.0:  # Orthogonal cell
        if a == b == c:  # Simple cubic (a=b=c)
            return 1  # ibrav 1: Simple cubic
        elif a == b:  # Tetragonal (a=b≠c)
            return 4  # ibrav 4: Tetragonal
        else:  # Orthorhombic (a≠b≠c)
            return 8  # ibrav 8: Orthorhombic
    elif alpha == beta == 90.0 and gamma == 120.0:  # Hexagonal
        return 6  # ibrav 6: Hexagonal
    elif alpha == beta and gamma != 120.0:  # Rhombohedral (or trigonal)
        return 7  # ibrav 7: Rhombohedral
    elif alpha != 90.0 and beta != 90.0 and gamma != 90.0:  # Triclinic
        return 12  # ibrav 12: Triclinic
    elif alpha == beta == 90.0 and gamma != 90.0:  # Monoclinic (general)
        return 13  # ibrav 13: Monoclinic with non-90° angles, unique axis along b or c
    elif alpha == 90.0 and beta != 90.0 and gamma == 90.0:  # Monoclinic (with unique axis along a)
        return -13  # ibrav -13: Monoclinic with unique axis along a
    else:
        return 0  # No symmetry or unusual case: ibrav 0

# Example usage:
# structure = StructureData(cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
# print(determine_ibrav(structure))
"""
