from ase.visualize import view
from ase.io import read
from ase import Atoms

from pymatgen.core.structure import Structure, IStructure, Molecule, IMolecule
from pymatgen.core.lattice import Lattice
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.units import Energy, Length, FloatWithUnit
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Species, Element
from pymatgen.transformations.standard_transformations import OxidationStateDecorationTransformation

import os
from subprocess import call
import sys
import numpy as np
from itertools import product
from warnings import warn

from scipy.spatial.transform import Rotation


# TOBETESTED... order between hexagonal and thombohedral is not obvious
# From highest to lowest constraints.
LATTICE_SYSTEMS_HIERARCHY = ['cubic',
                             'tetragonal', 'rhombohedral', 'hexagonal',
                             'orthorhombic', 'monoclinic', 'triclinic']

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


def get_pymatgen_structure(struct_or_file, struct_description=None,
                           return_description=False):
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

    return (structure, struct_description) if return_description else structure


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


def get_compact_formula(struct_or_file):
    """
    Get a compact formula for printout purposes.

    Uses the same atom_type ordering as in pymatgen
    composition, based on the IUPAC formula.

    NOT TESTED FOR STRUCTURES CONTAINING PARTIAL OCCUPANCIES

    Args:
        struct_or_file: coordinate file, pymatgen (I)Structure or ASE Atoms

    Returns:
        compact_formula: str
    """
    structure = get_pymatgen_structure(struct_or_file)
    compact_formula = ''
    for atom_type, amount in structure.composition.as_dict().items():
        if amount == 1:
            amount_str = ''
        elif not amount % 1:
            amount_str = '{}'.format(int(amount))
        else:
            amount_str = '{}'.format(amount)
        compact_formula += f"{atom_type}{amount_str}"

    return compact_formula


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
    _m = np.array(m)
    return np.allclose(_m, _m.dot(_m.T.conj()), **kwargs)


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

    cell_change_perm = np.array(cyclic_permutations(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], direction='left'))

    possible_lattice_systems = []
    for perm_index, (lengths_perm, angles_perm) in enumerate(
            zip(cyclic_permutations(cell_lengths),
                cyclic_permutations(cell_angles))):

        unique_param = None

        a, b, c = lengths_perm
        alpha, beta, gamma = angles_perm

        _swap_axes_matrix = cell_change_perm[perm_index]

        if verbosity >= 2:
            print('Permutation index {}: cell lengths {} and angles {}'.format(
                perm_index, lengths_perm, angles_perm))

        lattice_system = get_lattice_system_from_cell_parameters(
            a, b, c, alpha, beta, gamma, verbosity=0, **kwargs)

        # Define the unique parameter (axis or angle)
        if lattice_system in ['triclinic', 'orthorhombic'] :
            possible_lattice_systems.append({
                'name': lattice_system,
                'unique_parameter': None,
                'swap_axes_matrix': _swap_axes_matrix,
                'cell_parameters': (a, b, c, alpha, beta, gamma)})

        elif lattice_system in ['hexagonal', 'tetragonal', 'monoclinic']:
            if lattice_system in ['hexagonal',  'tetragonal']:
                unique_param = ('c', 'a', 'b')[perm_index]
            else:
                unique_param = ('β', 'γ', 'α')[perm_index]

            if verbosity >= 2:
                print(('Axes permutation index {}: Cell parameters are '
                       'compatible with a {}-unique {} lattice.').format(
                    perm_index, unique_param, lattice_system))

            possible_lattice_systems.append({
                'name': lattice_system,
                'unique_parameter': unique_param,
                'swap_axes_matrix': _swap_axes_matrix,
                'cell_parameters': (a, b, c, alpha, beta, gamma)})

        else:
            # Cubic or rhombohedral crystal systems cannot be mistaken
            # for others. Exit.
            print(('Axes permutation index {}: Cell parameters are '
                       'compatible with a {} lattice.').format(
                    perm_index, lattice_system))

            if return_as_dict:
                return {'lattice_system': lattice_system,
                        'unique_parameter': unique_param,
                        'swap_axes_matrix': _swap_axes_matrix}
            else:
                return lattice_system

    if verbosity >= 2:
        print('possible_lattice_systems = ', possible_lattice_systems)

    # Choose lattice system with best rank in the hierarchy.
    lowest_lattice_index = len(LATTICE_SYSTEMS_HIERARCHY)
    unique_param = None
    for _lattice_system in possible_lattice_systems:
        i = LATTICE_SYSTEMS_HIERARCHY.index(_lattice_system['name'])
        if i < lowest_lattice_index:
            lattice_system = _lattice_system['name']
            unique_param = _lattice_system['unique_parameter']
            swap_axes_matrix = _lattice_system['swap_axes_matrix']
            lowest_lattice_index = i
            cell_parameters = _lattice_system['cell_parameters']

    if verbosity >= 1:
        print(('The most-contrained lattice system compatible with '
               'the provided lattice is {}{}.').format(
            lattice_system,
            ' ({}-unique)'.format(unique_param) if unique_param else ''))

        print(f'obtained with cell parameters {cell_parameters}.')

    if return_as_dict:
        return {'lattice_system': lattice_system,
                'unique_parameter': unique_param,
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

    Symmetry search uses pymatgen SpacegroupAnalyzer, which usess spglib.

    UPDATE COMMENTS

    Args:
        structure: pymatgen Structure
            initial structure. Note that symmetry search can use the
            conventional standard cell instead.

        return_as_dict

        **kwargs:
            keyward arguments that should be passed to numpy
            isclose and allclose functions (atol, rtol, etc.)

    Returns:
        lattice_system
        OR
        lattice_system_dict
    """

    # Get primitive and convention structures and space group based on symmetry
    sga = SpacegroupAnalyzer(structure)
    primitive_structure = sga.get_primitive_standard_structure()
    conventional_structure = sga.get_conventional_standard_structure()
    lattice_system = sga.get_lattice_type()

    if verbosity >= 1:
        print(('Searching Bravais lattice system compatible with the cell '
               'parameters of {}-atom {} structure with space group {} ({}).'
               ).format(structure.num_sites, get_compact_formula(structure),
                        sga.get_space_group_symbol(), sga.get_space_group_number()))

    # Inititialize crystal_system_dict values
    unique_param = None
    swap_axes_matrix = None
    change_to_primitive = False
    change_to_conventional = False

    if verbosity >= 1:
        print(('Searching Bravais lattice system compatible {}-atom '
               'original reprentation.').format(structure.num_sites))
        # print('with cell parameters: ',
              # structure.lattice.lengths + structure.lattice.anles)

    orig_lattice_syst_dict = get_lattice_system_with_unique_param(
                structure, return_as_dict=True,
                verbosity=verbosity, **kwargs)
    orig_lattice_syst_dict['change_to_primitive'] = False
    orig_lattice_syst_dict['change_to_conventional'] = False

    if verbosity >= 1:
        print(('Searching Bravais lattice system compatible with {}-atom '
               'primitive representation of composition {}.'
               ).format(primitive_structure.num_sites,
                        get_compact_formula(primitive_structure)))
        # print('with cell parameters: ',
            # primitive_structure.lattice.lengths + primitive_structure.lattice.angles)

    prim_lattice_syst_dict = get_lattice_system_with_unique_param(
        primitive_structure, return_as_dict=True,
        verbosity=verbosity, **kwargs)
    prim_lattice_syst_dict['change_to_primitive'] = True
    prim_lattice_syst_dict['change_to_conventional'] = False

    if verbosity >= 1:
        print(('Searching Bravais lattice system compatible with {}-atom '
               'conventional representation of composition {}.'
               ).format(conventional_structure.num_sites,
                        get_compact_formula(conventional_structure)))
        # print('with cell parameters: ',
              # conventional_structure.lattice.lengths + conventional_structure.lattice.angles)

    conv_lattice_syst_dict = get_lattice_system_with_unique_param(
            conventional_structure, return_as_dict=True,
            verbosity=verbosity, **kwargs)
    conv_lattice_syst_dict['change_to_primitive'] = False
    conv_lattice_syst_dict['change_to_conventional'] = True


    sym_based_lattice_rank = LATTICE_SYSTEMS_HIERARCHY.index(
        lattice_system)

    # Initialize lowest index
    lattice_system_dict = None
    lowest_index = sym_based_lattice_rank
    current_lattice_origin = 'symmetry'

    # TODO: insert on option to prevent using the primitive representation
    representations = ['original', 'conventional', 'primitive']
    structure_representation_mapping = {
        'original': structure,
        'conventional': conventional_structure,
        'primitive': primitive_structure
    }

    for _lattice_syst_dict, representation in zip(
        [orig_lattice_syst_dict, conv_lattice_syst_dict, prim_lattice_syst_dict],
        representations):

        index = LATTICE_SYSTEMS_HIERARCHY.index(
            _lattice_syst_dict['lattice_system'])
        if index <= lowest_index:
            lattice_system_dict = _lattice_syst_dict.copy()
            current_lattice_origin = representation + ' cell paramaters'
            lowest_index = index
            if ('unique_parameter' in lattice_system_dict.keys()
                    and lattice_system_dict['unique_parameter']):
                unique_param_str = ' ({}-unique)'.format(
                    lattice_system_dict['unique_parameter'])
            else:
                unique_param_str = ''
            if verbosity >= 2:
                print('Lattice system updated to {}{} from {}'.format(
                    lattice_system_dict['lattice_system'],
                    unique_param_str, current_lattice_origin))

    if verbosity >= 1:
        print(('{}-atom structure {} has cell paramaters compatible with '
               'a {}{} Bravais lattice system based on its {}-atom {} representation.'
               ).format(structure.num_sites, get_compact_formula(structure),
                        lattice_system_dict['lattice_system'], unique_param_str,
                        structure_representation_mapping[representation].num_sites,
                        representation))

    if verbosity >= 2:
        print('lattice_system_dict = ', lattice_system_dict)

    if return_as_dict:
        return lattice_system_dict
    else:
        return lattice_system_dict['lattice_system']


def permutate_structure_axes(structure, perm_matrix):
    """
    Permutate the cell parameters and atomic coordinates of a pymatgen Structure
    based on a given permutation matrix.

    Args:
        structure (Structure): pymatgen Structure object to permutate.
        perm_matrix (np.ndarray): 3x3 permutation matrix to apply.

    Returns:
        Structure: The permutated pymatgen Structure object.
    """
    # Ensure the permutation matrix is a 3x3 numpy array
    perm_matrix = np.array(perm_matrix)
    if perm_matrix.shape != (3, 3):
        raise ValueError("Permutation matrix must be 3x3.")

    # Get the current lattice (cell) matrix of the structure
    lattice = structure.lattice.matrix

    # Apply the permutation matrix to the lattice (column vectors)
    new_lattice = np.dot(lattice.T, perm_matrix).T

    # Now permutate the atomic positions by applying the same permutation
    new_positions = np.dot(structure.frac_coords, perm_matrix)

    # Return a new structure with the updated lattice and atomic positions
    new_structure = Structure(new_lattice, structure.species, new_positions,
                              coords_are_cartesian=False)
    return new_structure


def get_structure_with_compatible_lattice_system(structure_or_file,
                                                 lattice_system_dict=None,
                                                 verbosity=1, **kwargs):
    """
    Get a structure representation with cell axes compatible with the
    Bravais lattice system identified from cell paramaters.

    Args:
        structure_or_file: str, pymatgen Structure or ASE Atoms
            Structure or coordinate file to work with.

        lattice_system_dict: dict or None (default is None),
            Provide lattice_system_dict in cas it was already
            calculated before with get_lattice_system_from_sym_or_cell.

        verbosity: int (default is 1),
            Verbosity level
        **kwargs:
            Keword arguments to pass to numpy isclose and allclose
            functions, including atol and rtol absolute and
            relative tolerance values.

    Returns:
        new_structure: pymatgen Structure
    """
    structure = get_pymatgen_structure(structure_or_file)

    if not lattice_system_dict:
        lattice_system_dict = get_lattice_system_from_sym_or_cell(
        structure, return_as_dict=True, verbosity=verbosity, **kwargs)

    sga = SpacegroupAnalyzer(structure)

    if verbosity >= 1:
        print(('{}-atom {} structure representation will be modified '
               'if necessary to get cell parameters compatible with the '
               '{} Bravais lattice system identified based on cell parameters{}.'
               ).format(structure.num_sites, get_compact_formula(structure),
                        lattice_system_dict['lattice_system'],
                        '' if sga.get_lattice_type() == lattice_system_dict['lattice_system']
                        else ' (vs {} from symmetry)'.format(sga.get_lattice_type())))

    if lattice_system_dict['change_to_primitive']:
        new_struct = sga.get_primitive_standard_structure()
        print(('Converting initial {}-atom structure to its {}-atom '
               'primitive representation.').format(structure.num_sites,
                new_struct.num_sites))
    elif lattice_system_dict['change_to_conventional']:
        new_struct = sga.get_conventional_standard_structure()
        print(('Converting initial {}-atom structure to its {}-atom '
               'conventional representation.').format(structure.num_sites,
                new_struct.num_sites))
    else:
        new_struct = structure.copy()

    if not is_unitary_matrix(lattice_system_dict['swap_axes_matrix']):
        if verbosity >= 2:
            print('Structure axes will be permuted according to swap_axes_matrix:\n',
                  lattice_system_dict['swap_axes_matrix'])

        new_struct = permutate_structure_axes(new_struct,
            lattice_system_dict['swap_axes_matrix'])

        if verbosity >= 1:
            print('Cell axes were swapped to : ',
                new_struct.lattice.lengths + new_struct.lattice.angles)

    return new_struct


def get_rotation_axis_and_angle(v_1, v_2, angle_in_radians=False):
    """
    Get the rotation axis and angles between sets of angles expressed in different frames

    Args:
        v_1: array_like, shape (3,) or (N, 3)
            Vector components observed in initial frame 1.
            Each row of v_1 denotes a vector.
        v_2: array_like, shape (3,) or (N, 3)
            Vector components observed in initial frame 2.
            Each row of v_2 denotes a vector.
        angle_in_radians: bool (default is False)
            Whether rotation angle should be returned in radians rather than degrees.
    """
    rotation = Rotation.align_vectors(v_1, v_2)[0]
    mrp = rotation.as_mrp()
    axis_norm = np.linalg.norm(mrp)
    axis = mrp / axis_norm
    angle = 4 * np.arctan(axis_norm)
    if not angle_in_radians:
        angle = (180 / np.pi ) * angle

    return axis, angle


def has_transition_metal(struct_or_file):
    """
    Determine whether structure contains at least one transition metal

    Args:
        struct_or_file: pymatgen Structure, ase Atoms, file
    Returns:
        bool
    """
    structure = get_pymatgen_structure(struct_or_file)
    return any([elmt.is_transition_metal for elmt in structure.elements])


def get_composition(compo_formula_or_struct):
    """
    get pymatgen COmposition instance from a Composition, formula or Structure

    Args:
        compo_formula_or_struct: Composition, str or Structure
            composition, formula or Structure

    Returns:
        composition: pymatgen Composition instance
    """
    if isinstance(compo_formula_or_struct, str):
        composition = Composition(compo_formula_or_struct)
    elif isinstance(compo_formula_or_struct, Structure):
        composition = compo_formula_or_struct.composition
    elif isinstance(compo_formula_or_struct, Composition):
        composition = compo_formula_or_struct
    else:
        raise TypeError('compo_formula_or_struct should be a pymatgen '
                        'composition or structure or a chemical formula.')
    return composition


def find_neutral_oxidation_state_combinations(compo_formula_or_struct,
                                              fixed_oxidation_states=None,
                                              return_scores=True,
                                              verbosity=1):
    """
    Find combinations of oxidation states that result in a neutral global charge for a given composition.

    Args:
        compo_formula_or_struct: Composition, str or Structure
            composition, formula or Structure
        fixed_oxidation_states: dict or None (default is None)
            A dictionary of elements and oxidation states that should
            be fixed (e.g. {'O': -2})
        return_scores: bool (default is True)
            whether neutral combination scores should be returned
        verbosity: int (default is 1)
            verbosity level

    Returns:
        list: A list of dictionaries, where each dictionary represents a combination of oxidation states
              that results in a neutral global charge.
    """
    composition = get_composition(compo_formula_or_struct)
    if verbosity >= 2:
        print(f'Input composition is {composition}')

    # Get the Elements (rather than Species) in the composition
    elements = []
    for el_or_sp in composition:
        if isinstance(el_or_sp, Species):
            elements.append(el_or_sp.element)
        elif isinstance(el_or_sp, Element):
            elements.append(el_or_sp)
    if verbosity >= 3:
        print(f'Possible and common oxidation states in {composition.reduced_formula}: ')
        for el in elements:
            print(f'{el.name} has  possible ox. states '
                  f'{el.oxidation_states} and common ox. states  '
                  f'{el.common_oxidation_states}.')

    # Get possible oxidation states for each element
    possible_oxidation_states = []
    for element in elements:
        if (isinstance(fixed_oxidation_states, dict)
            and element.name in fixed_oxidation_states.keys()):
            # TODO check if int or list
            if isinstance(fixed_oxidation_states[element.name], int):
                oxidation_states = [fixed_oxidation_states[element.name]]
            else:
                oxidation_states = fixed_oxidation_states[element.name]
            if verbosity >= 2:
                print(f'{element} oxidation_states fixed to {oxidation_states}')

        else:
            oxidation_states = element.oxidation_states

        possible_oxidation_states.append(oxidation_states)

    # Generate all possible combinations of oxidation states
    oxidation_state_combinations = list(product(*possible_oxidation_states))

    # Check which combinations result in a neutral global charge
    neutral_combinations = []
    for combination in oxidation_state_combinations:
        total_charge = 0
        for element, oxidation_state in zip(elements, combination):
            # Calculate the total charge for the current combination
            total_charge += composition[element] * oxidation_state

        # Check if the total charge is zero (neutral)
        if total_charge == 0:
            neutral_combinations.append(dict(zip(elements, combination)))

    if not len(neutral_combinations):
        error_msg = ('No neutral charge compatible with possible oxidation ' +
                     f'states could be found for {composition.reduced_formula}.')
        raise ValueError(error_msg)


    # Rank combinations by plausibility:
    scores = []
    for neutral_combination in neutral_combinations:
        score = 0
        for element, oxidation_state in neutral_combination.items():
            if oxidation_state in element.common_oxidation_states:
                score += 1
        scores.append(score)

    sorted_indexes = np.argsort(scores)[::-1]

    neutral_combinations = [neutral_combinations[i] for i in sorted_indexes]
    scores = [scores[i] for i in sorted_indexes]

    if verbosity >= 1:
        for i, (nc, score) in enumerate(zip(neutral_combinations, scores)):
            print(f'combination {i + 1} out of {len(scores)} with score {score}')
            print({k.name: v for k, v in nc.items()})

    if return_scores:
        return neutral_combinations, scores
    else:
        return neutral_combinations


def get_plausible_oxidation_states(compo_formula_or_struct,
                                   fixed_oxidation_states=None,
                                   return_species=True,
                                   return_all_best_combinations=False,
                                   raise_error_for_uncommon_ox_states=True, 
                                   verbosity=1):
    """
    Find the most likely oxidation states of all elements for a given composition.

    Args:
        compo_formula_or_struct: Composition, str or Structure
            composition, formula or Structure
        fixed_oxidation_states: dict or None (default is None)
            A dictionary of elements and oxidation states that should
            be fixed (e.g. {'O': -2})
        return_species: bool (default is True)
            Return a list of Species instances (element with
            oxidation state) will be returned. If False, a dictionary
            of element names and oxidation states will be returned instead
            (or a list thereof if return_all_best_combinations is True).
        return_all_best_combinations: bool (default is False)
            In the case where several combinations share the highest
            score, a list of all will be returned instead of one.
        raise_error_for_uncommon_ox_states: bool (default is True)
            A ValueError will be raised if uncommon oxidation states
            are required to satisfy charge neutrality.
            If False, a simple warning will be thrown.
        verbosity: int (default is 1)
            verbosity level

    Returns:
        list: dictionary of elements and oxidation states or list of species
            or even a list of list or dictionnaries if return_all_best_combinations
            is True.
    """
    composition = get_composition(compo_formula_or_struct)
    formula = composition.reduced_formula
    neutral_combinations, scores = find_neutral_oxidation_state_combinations(
        composition, fixed_oxidation_states=fixed_oxidation_states,
        return_scores=True, verbosity=verbosity)

    # count the number of best-scores
    if not len(scores):
        error_msg = ('No neutral combination compatible with possible ' +
                     f'oxidation states could be found for {formula}.')
        raise ValueError(error_msg)

    # Count the number of uncommon ox_states
    best_score = scores[0]  # should be equal to the number of elements
                            # if all oxidation states in the best combination
                            # are common.
    nb_of_elements = len(neutral_combinations[0])
    nb_of_uncommon_ox_states = nb_of_elements - best_score
    if  nb_of_uncommon_ox_states:
        error_msg = ('At least {} elements must be in an uncommon oxidation state '
                    'to obtain a neutral charge for {}.').format(
                        nb_of_uncommon_ox_states, formula)
        if raise_error_for_uncommon_ox_states:
            raise ValueError(error_msg)
        else:
            warn(warn_msg)

    if len(scores) > 1 and scores[1] == scores[0]:
        warn(f'Several combinations of oxidation states were found for {formula}.')
        print('One of these combinations is : {}'.format(
            {k.name: v for k, v in neutral_combinations[0].items()}))
    else:
        print(('The most plausible combination of oxidation states for {} '
               ' is: {}').format(
               formula,
               {k.name: v for k, v in neutral_combinations[0].items()}))

    if return_all_best_combinations:
        best_combinations = []
        i = -1
        while 1:
            i += 1
            if i >= len(scores):
                if verbosity >= 3:
                    print('All combinations have been explored. Exiting while loop.')
                break
                
            if scores[i] < best_score:
                if verbosity >= 2:
                    print(f'All {i-1} best-score ({best_score}/{nb_of_elements}) '
                          f' combinations have been considered.')
                break
            
            if verbosity >= 3:
                print(('Considering neutral combination {} out of {} with '
                       'score {}/{}').format(i, len(neutral_combinations),
                                             scores[i], nb_of_elements))

            if return_species:
                combination = [Species(k, v) for k, v in neutral_combinations[i].items()]
            else:
                combination = {el.name: ox_state for el, ox_state
                               in neutral_combinations[i].items()}

            best_combinations.append(combination)
            if verbosity >= 2:
                print(('Neutral combination {} out of {} with '
                       'best score ({}/{}) wad added: {}').format(i,
                        len(neutral_combinations), scores[i], nb_of_elements,
                        combination))

        if verbosity >= 1:
            print(('{} combination with the best score ({}/{}) will be '
                  'returned.').format(len(best_combinations), best_score,
                                      nb_of_elements))
            for combination in best_combinations:
                print(combination)

        return best_combinations

    else:
        _ = ('Returning best neutral combination of oxidation states for ' +
             f'{formula} as a ')
        if return_species:
            list_of_species = [Species(k, v) for k, v in neutral_combinations[0].items()]
            if verbosity >= 2:
                print(f'list of Species: {list_of_species}.')
            return list_of_species
        else:
            dict_of_names_and_ox_states = {el.name: ox_state for el, ox_state
                                           in neutral_combinations[0].items()}
            if verbosity >= 2:
                print('dictionary of element names and oxidation states: '
                      f'{dict_of_names_and_ox_states}.')
            return dict_of_names_and_ox_states


def guess_structure_oxidation_states(structure, 
                                     fixed_oxidation_states=None,
                                     raise_error_for_uncommon_ox_states=True, 
                                     verbosity=1):
    """
    TO BE COMPLETED
    """
    # TODO: try to set oxidation states with pymatgen
    new_structure = get_pymatgen_structure(structure).copy()
    new_structure.add_oxidation_state_by_guess()
    formula = new_structure.composition.reduced_formula
    
    use_pymatgen_oxi_states = True
    
    if all([s.oxi_state == 0 for s in new_structure.composition.elements]):
        if verbosity >= 2:
            print('Pymatgen add_oxidation_state_by_guess method found 0 ox. states'
                  ' for all atom types in {formula} structure.')
        use_pymatgen_oxi_states = False
    
    elif fixed_oxidation_states:
        for atom_type, ox_states in fixed_oxidation_states.items():
            _ox_states = [ox_states] if isinstance(ox_states, int) else ox_states
                
            for site_index, species in enumerate(new_structure.species):
                site_type = species.element.name
                site_ox_state = species.oxi_state
                if site_type == atom_type:
                    if site_ox_state not in _ox_states:
                        if verbosity >= 2:
                            print(f'Oxidation state of {site_type} site with index {site_index} '
                                  f'({site_ox_state}) is not  within the list of '
                                  f'fixed_oxidation_states ({ox_states})')
                            
    if not use_pymatgen_oxi_states:
        if verbosity >= 2:
            print('Trying to guess oxidation states with Pyama get_plausible_oxidation_states.')
        
        best_combination = get_plausible_oxidation_states(new_structure, 
                                       fixed_oxidation_states=fixed_oxidation_states,
                                       return_species=True,
                                       return_all_best_combinations=False,
                                       raise_error_for_uncommon_ox_states=raise_error_for_uncommon_ox_states, 
                                       verbosity=verbosity)
        
        osdt = OxidationStateDecorationTransformation({s.element.name: s.oxi_state for s in best_combination})
        new_structure = osdt.apply_transformation()
    
    return new_structure
    
