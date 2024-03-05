from ase.visualize import view
from ase.io import read
from ase import Atoms

from pymatgen.core.structure import Structure, IStructure, Molecule, IMolecule
from pymatgen.core.lattice import Lattice
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.units import Energy, Length, FloatWithUnit

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


def get_ase_atoms(struct_or_file, struct_description=None):
    """
    Get an ASE Atoms object from a file or pymatgen Structure or Molecule

    Atoms object will be left untouched.
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
    return atoms, description


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



