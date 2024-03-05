"""
Build a polymer atom by atom in a periodic cell

Sylvian Cadars, Assil Bouzid
Institute of Research on Ceramics (IRCER), University of Limoges, CNRS, France
sylvian.cadars@unilim.fr

Version using scipy fsolve to find positions of atoms in the polyhedrons

The program reads input parameters defining the system, species_properties
and other parameters (such as relative and absolute bond length tolerances)
from the JSON input.json file or any other json file with the -i option.

All options are automatically saved in sample_input.json file which also
contains default parameters.

IMPORTANT: Although the program uses a lot of "random" generation (numpy)
routines, results may be reproduced by fixing the seed value with the -s
(--seed) CLI option. The seed value used is aways written at the beginning of
the output.

In the current version, to save output in a file one must use:
    python ceramicNetworkBuilder.py [OPTIONS] > OUTPUT.TXT
This will be modified in future versions with the addition of a -o CLI option
and an automatic output.txt file.
"""

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen import core
from pymatgen.core.sites import Site, PeriodicSite
from pymatgen.core.bonds import get_bond_length as pmg_get_bond_length

from ase.visualize import view

import numpy as np
from scipy.optimize import fsolve, minimize
import click
import sys
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import json

"""
import sympy as sym
sym.init_printing()
"""

__version__ = '2022.10.26_01'

# Maximum coordination number currently accepted by the program
# Increasing this number will require quite a bit of work...
MAX_COORD_NUMBER = 4

class ceramicNetworkBuilderData():
    """
    Class containing all methods and properties necessary to run ceramicNetworkBuilder

    TODO:
        - define ceramicNetworkBuilderData as inherited from baseDataClass
          (see exemple in pyama.mdAnalyses.utils.mdAnalysesData)
          change all "if self.verb >= ... print()" statements to
          self.print(TEXT, verb_th=INT)
        - Add get_xray_pdf(show_plot=True, **kwargs) function using
          pyama.diffractionPkg.nanopdf.nanopdfData
        - Add hydrogen atoms (step 3)
        -

    properties:
        seed
        rng
        species_properties
        system
        structure
        abs_bond_length_tol
        rel_bond_length_tol
        self.verbosity
    """



    def __init__(self, input_file='input.json', seed=None, search_radius=2.5,
                 abs_bond_length_tol=0.2, rel_bond_length_tol=0,
                 max_attempts=10, numeric_tolerance=1e-6,
                 max_iterations_step1=1000, max_iterations_step2=500,
                 visualizer='ase', export_format='poscar', verbosity=1,
                 print_to_console=True, print_to_file=False):
        """
        Initialization
        """
        self.verbosity = verbosity
        if isinstance(seed, int):
            self.seed = seed
        else:
            self.seed = np.random.randint(10000)
        if self.verbosity >= 1:
            print()
        # Intialize numpy random number generator (RNG) to ensure reproducibility
        if self.verbosity >= 1:
            print(('Initialization of random number generator with seed: {}'
                   ).format(self.seed))
        self.rng = np.random.default_rng(self.seed)

        # TODO: define atom_types, nb_atoms_by_type from file
        # TODO: define system
            # Define system
        # self._set_species_properties_manually()
        # self._set_system_manually()
        self.input_file = input_file
        self.set_properties_and_system_from_json()
        self.set_bond_length_matrix()
        self.max_attempts = max_attempts
        self.numeric_tolerance = numeric_tolerance
        self.max_iterations_step1 = max_iterations_step1
        self.max_iterations_step2 = max_iterations_step2
        self.search_radius = search_radius
        self.abs_bond_length_tol = abs_bond_length_tol    # in Angstroms
        self.rel_bond_length_tol = rel_bond_length_tol    # in fraction of the tabulated bond length
        self.visualizer = visualizer
        self.export_format = export_format
        self.verbosity = self.verbosity
        self.print_to_console = print_to_console
        self.print_to_file = print_to_file
        self._output_text = []   # list of strings to be ultimately written in file

        # self._outputfile_ = open('output.txt', 'w')   # This file will need to be closed automatically


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


    def print_versions(self):
        self.print('ceramicNetworkBuilder version: {}'.format(__version__))
        self.print('Pymatgen.core version: {}'.format(core.__path__))
        self.print('Pymatgen.core version: {}'.format(core.__version__))

    def _set_species_properties_manually(self):
        """
        Define species_properties manually.

        FOR TESTING PURPOSES. REPLACE BY AN INPUT FILE PARSING
        """
        self.species_properties = {
            'Si': {
                'coord_proba': (0, 0, 0, 0, 1),
                'clustering_proba': (0, 0.05, 0.95),
            },
            'C': {
                'coord_proba': (0, 0, 0, 0.9, 0.1),
                'clustering_proba': (0.025, 0.95, 0.025)
            },
            'N': {
                'coord_proba': (0, 0, 0, 0.7, 0.3),
                'clustering_proba': (0.95, 0.05, 0)
            },
        }

    def set_properties_and_system_from_json(self, file_name=None):
        if file_name is None:
            file_name = self.input_file
        with open(file_name) as f:
            input_dict = json.load(f)
        if 'system' in [k.lower() for k in input_dict.keys()]:
            self.system = input_dict['system']
            if len(self.system['atom_types']) != len(self.system['nb_of_atoms_by_type']):
                raise ValueError('Lengths of system atom_types and nb_of_atoms_by_type should match.')

            for cell_property in ['cell_lengths', 'cell_angles']:
                if isinstance(self.system[cell_property], (float, int)):
                    # Convert single vaue to a tuple of 3 identical values
                    self.system[cell_property] = tuple(3 * [float(self.system[cell_property])])
                elif not isinstance(self.system[cell_property], (tuple, list)):
                    raise TypeError(f'System {cell_property} should be a number or a list or tuple of length 3.')
                elif len(self.system[cell_property]) != 3:
                    raise ValueError(f'System {cell_property} should be a of length 3.')
                else:
                    pass # Nothing to do in this case

        if 'species_properties' in [k.lower() for k in input_dict.keys()]:
            # TODO: add checks for inputs: keys, types, shapes, etc.
            self.species_properties = input_dict['species_properties']
            if not set(self.system['atom_types']).issubset(set(self.species_properties.keys())):
                # TODO: return standardized error
                raise ValueError('Species_properties keys should match system[\'atom_types\']')
            for atom_type in self.species_properties.keys():
                if not isinstance(self.species_properties[atom_type]['coord_proba'],
                                  (list, tuple)):
                    raise TypeError(f'species_properties[{atom_type}]'
                                    '[\'coord_proba\'] should be a list or '
                                    'tuple.')
                elif len(self.species_properties[atom_type]['coord_proba']) != MAX_COORD_NUMBER + 1:
                    raise ValueError(('species_properties[{}]'
                                     '[\'coord_proba\'] should be of length {}'
                                     ).format(atom_type, MAX_COORD_NUMBER + 1))
                if not isinstance(self.species_properties[atom_type][
                        'clustering_proba'], (list, tuple)):
                    raise TypeError('species_properties[{atom_type}]'
                                    '[\'clustering_proba\'] should be a list '
                                    'or tuple.')
                elif len(self.species_properties[atom_type]['clustering_proba']
                         ) != len(self.system['atom_types']):
                    raise ValueError(('species_properties[{}][\'clustering_proba'
                                      '\'] should be of length {}').format(
                                          atom_type,
                                          len(self.system['atom_types'])))

        # TODO: offer the possibility to load other imput parameters from the
        # json file. commadline should take have higher priority.

    def save_sample_json_input_file(self, file_name='sample_input.json'):
        with open(file_name, 'w') as f:
            result = json.dump({
                'species_properties': self.species_properties,
                'system': self.system,
                'parameters': {
                    'seed': self.seed,
                    'max_iterations_step1': self.max_iterations_step1,
                    'max_iterations_step2': self.max_iterations_step2,
                    'max_attempts': self.max_attempts,
                    'search_radius': self.search_radius,
                    'abs_bond_length_tol': self.abs_bond_length_tol,
                    'rel_bond_length_tol': self.rel_bond_length_tol,
                    'export_format': self.export_format,
                    'print_to_console': self.print_to_console,
                    'print_to_file': self.print_to_file,
                    'numeric_tolerance': self.numeric_tolerance,
                    'verbosity': self.verbosity,  # TODO: continue this list
                }

            }, f, indent=4)
            self.print('Sample input JSON file saved as: {}'.format(file_name))

        return result

    def _set_system_manually(self):
        """
        Define the input target system 'manually'

        FOR TESTING PURPOSES. REPLACE BY AN INPUT FILE PARSING
        """
        # ******  100-atom test system *********
        """
        self.system = {
            'cell_lengths': (12, 12, 12),
            'cell_angles': (90, 90, 90),
            'atom_types': ('Si', 'C', 'N'),
            'nb_of_atoms_by_type': (39, 31, 30),
        }
        """
        # ****** Small 25-atom system *******
        self.system = {
            'cell_lengths': (8, 8, 8),
            'cell_angles': (90, 90, 90),
            'atom_types': ('Si', 'C', 'N'),
            'nb_of_atoms_by_type': (9, 8, 8),
        }



    def initialize_structure(self, first_atom_intern_coords=None):

        # Build system:
        lattice = Lattice.from_parameters(self.system['cell_lengths'][0],
            self.system['cell_lengths'][1], self.system['cell_lengths'][2],
            self.system['cell_angles'][0], self.system['cell_angles'][1],
            self.system['cell_angles'][2])

        # Construct structure with first atom
        # Pick atom type randomly with probabilities according to target system compo
        nb_of_atoms_by_type = np.asarray(self.system['nb_of_atoms_by_type'])
        [atom_type_index] = self.rng.choice(
            range(len(self.system['atom_types'])),
            size=1, p=nb_of_atoms_by_type / np.sum(nb_of_atoms_by_type))
        atom_type = self.system['atom_types'][atom_type_index]

        if first_atom_intern_coords is None:
            self.structure = Structure(lattice=lattice, species=[atom_type],
                               coords=[self.rng.random(3)])
        else:
            self.structure = Structure(lattice=lattice, species=[atom_type],
                               coords=[first_atom_intern_coords])

        # Initialize site properties
        self.initialize_site_properties(self.structure.sites[0])


    def visualize(self):

        if self.visualizer.lower() == 'ase':
            ase_struct = AseAtomsAdaptor.get_atoms(self.structure)
            view(ase_struct)
        if self.visualizer.lower() == 'vesta':
            self.structure.to(fmt='cif', filename='tmp.cif')
            # TODO: add a try/exceptions
            sp = subprocess.run(['vesta', 'tmp.cif'], capture_output=True)
            if self.verbosity >= 2:
                print('Opening structure save as tmp.cif with VESTA.')
                print(sp)


    def get_type_from_index(self, index):
        atom_type = self.structure.sites[index].species.elements[0].name
        return atom_type


    @staticmethod
    def get_atom_type(site):
        atom_type = site.species.elements[0].name
        return atom_type


    @staticmethod
    def get_bond_angle_from_CN(coord_number):
        if coord_number == 2:
            theta = 180.0
        elif coord_number == 3:
            theta = 120.0
        elif coord_number == 4:
            theta = 109.4712206
        elif coord_number == 6:
            theta = 90.0
        else:
            raise ValueError
        return theta


    def initialize_site_properties(self, site, **kwargs):
        """
        Initialize the custom properties of a site

        Args:
            site: pymatgen.core.sites.PeriodicSite or Site

        TODO:
            (if necessary) use a kwarg to change a specific property to non-default value.
        """
        if not isinstance(site, (PeriodicSite, Site)):
            raise(TypeError,
                  'Argument site should be of type pymatgen.core.site.(Periodic)Site.')
        site.properties['is_shell_complete'] = False
        site.properties['is_treated'] = False
        site.properties['treatment_attempts'] = 0
        site.properties['target_coord_number'] = \
            self.pick_coord_number_from_type(self.get_atom_type(site))
        site.properties['connected_neighbors'] = []

        return site


    def set_bond_length_matrix(self, user_matrix=None):
        """
        Set bond_length_matrix property using pymatgen table or manually
        """
        if user_matrix is None:
            self.bond_length_matrix = np.zeros(2*[len(self.system['atom_types'])])
            for i in range(len(self.system['atom_types'])):
                for j in range(i+1):
                    self.bond_length_matrix[i, j] = float(pmg_get_bond_length(
                        self.system['atom_types'][i], self.system['atom_types'][j]))
                    if j != i:
                        self.bond_length_matrix[j, i] = self.bond_length_matrix[i, j]
        else:
            # TODO: check user_matrix dimensions
            # TODO: allow user to define one or several bond(s) manually and let pymatgen
            # decide for the other
            self.bond_length_matrix = np.asarray(user_matrix, dtype=float)


    def get_bond_length(self, type_A, type_B):
        """
        get bond length for atom types type_A and type_B (case-insensitive) or type indexes

        Args:
            type_A: str or int
                First atom type or type index in the order of self.system['atom_types']
            type_B:
                Second atom type or type index in the order of self.system['atom_types']
        """

        if isinstance(type_A, str) or isinstance(type_B, str):
            atom_types = [atom_type.lower() for atom_type in self.system['atom_types']]
        if not isinstance(type_A, (str, int)) or not isinstance(type_B, (str, int)):
            sys.exit('Wrong argument type to method get_bond_length')

        if isinstance(type_A, str):
                type_index_A = atom_types.index(type_A.lower())
        else:
            type_index_A = type_A
        if isinstance(type_B, str):
            type_index_B = atom_types.index(type_B.lower())
        else:
            type_index_B = type_B
        return self.bond_length_matrix[type_index_A, type_index_B]


    def get_bond_length_with_tol(self, type_A, type_B, tol_sign='plus'):
        """
        Get bond length +/- relative and/or absolute tolerance
        """
        BL = self.get_bond_length(type_A, type_B)
        if tol_sign.lower() in ('+', 'plus', 'pos', 'positive'):
            BL_with_tol = BL * (1 + self.rel_bond_length_tol) + \
                          self.abs_bond_length_tol
        elif tol_sign.lower() in ('-', 'minus', 'neg', 'negative'):
            BL_with_tol = BL * (1 - self.rel_bond_length_tol) - \
                          self.abs_bond_length_tol
        return BL_with_tol


    def get_bond_length_boundaries(self, type_A, type_B):
        """
        Get bond length boundaries given relative and absolute tolerance
        """
        BL_min = self.get_bond_length_with_tol(type_A, type_B, tol_sign='-')
        BL_max = self.get_bond_length_with_tol(type_A, type_B, tol_sign='+')
        return BL_min, BL_max


    def get_max_bond_length_for_type(self, atom_type, include_tol=False,
                                     tol_sign='+'):
        """
        Get maximum bond length given the atom type, possibly with abs/rel tol

        Args:
            atom_type: str
                Atom type
            include_tol: bool (default is False)
                whether to include abs/rel bond length tolerance in result
            tol_sign: str or int (default is '+')
                if in ('-', 'minus', 'neg', 'negative', -1) absolute/relative
                bond length tolerancve will be substracted rather than added.

        Returns:
            max_bond_length : float
            maximum bond length between requested type and all other types in
            the structure
        """
        max_bond_length = np.max(self.bond_length_matrix[
            self.get_atom_type_index(atom_type)])
        if include_tol:
            if tol_sign.lower() in ('-', 'minus', 'neg', 'negative', -1):
                sign_factor = -1
            else:
                sign_factor = 1
            max_bond_length *= (1 + sign_factor*self.rel_bond_length_tol)
            max_bond_length += self.abs_bond_length_tol
        return max_bond_length


    def get_contact_dist(self, type_A, type_B, rel_contact=0.5):
        return self.get_bond_length(type_A, type_B) * rel_contact


    def get_remaining_atoms_by_type(self, as_dict=False):
        """
        get number of atoms of each type to be inserted in structure

        Args:
            as_dict: bool (default is False)
                if True the function will return a dict mapping of the form
                {
                    'atomic_type_1': nb_of_atoms_of_type_1,
                    ...
                }

        returns:
            - a list of number of remainingining atoms in the same order as
              atom_types
            - a mapping of atom_type: nb_of_remaining_atoms
        """
        remaining_atoms = list(self.system['nb_of_atoms_by_type'])
        if as_dict:
            mapping = {}
        for type_index, atom_type in enumerate(self.system['atom_types']):
            for site in self.structure.sites:
                if site.species.elements[0].name.lower() == atom_type.lower():
                    remaining_atoms[type_index] -= 1
            if as_dict:
                mapping[atom_type] = remaining_atoms[type_index]

        if as_dict:
            remaining_atoms = mapping
        return remaining_atoms


    def get_incomplete_sites(self, of_type=None):
        """
        Get list of incomplete sites, potentially of the given type(s)

        Args:
            of_type: str, list or tuple (default is None)
                Type or list of types to be considered. If None, all types in
                the structure are considered.

        Returns:
            incomplete_sites: list
                List of indexes
        """
        if of_type is None:
            incomplete_sites = [index for index, site in enumerate(
                self.structure.sites) if not site.properties[
                'is_shell_complete']]
        else:
            if isinstance(of_type, str):
                of_type = [of_type]
            incomplete_sites = [index for index, site in enumerate(
                self.structure.sites) if (not site.properties[
                'is_shell_complete']) and self.get_atom_type(site) in of_type]
        return incomplete_sites

    def get_complete_sites(self, of_type=None):
        """
        Get list of sites with complete shell.

        Make sure list of neighbors is up-to-date.

        Args:
            of_type: str, list or tuple (default is None)
                Type or list of types to be considered. If None, all types in
                the structure are considered.

        Returns:
            complete_sites: list
                List of site indexes with complete shell
        """
        if of_type is None:
            complete_sites = [index for index, site in enumerate(
                self.structure.sites) if site.properties['is_shell_complete']]

        else:
            # Convert single tye to list
            if isinstance(of_type, str):
                of_type = [of_type]
            complete_sites = [index for index, site in enumerate(
                self.structure.sites) if (site.properties['is_shell_complete']
                and self.get_atom_type(site) in of_type)]
        return complete_sites

    def get_nb_of_incomplete_sites(self, of_type=None):
        return len(self.get_incomplete_sites(of_type=of_type))

    def get_nb_of_complete_sites(self, of_type=None):
        return len(self.get_complete_sites(of_type=of_type))

    def pick_coord_number_from_type(self, atom_type):
        """
        pick coordination number randomly based on species_properties[atom_type]['coord_proba']

        atom_type is case sensitive.
        """
        if self.verbosity >= 4:
            print('Running pick_coord_number_from_type method for an atom of type {}.'.format(
                  atom_type))
            print('Selecting among type indexes: ', list(range(len(self.species_properties[atom_type]['coord_proba']))))
            print('with probabilities: ', self.species_properties[atom_type]['coord_proba'])

        [coord_number] = self.rng.choice(
            list(range(len(self.species_properties[atom_type]['coord_proba']))),
            size=1, p=self.species_properties[atom_type]['coord_proba'])
        # TODO ? take remaining atoms into account ?

        if self.verbosity >= 1:
            print('Picked coord number for {} atom: {}.'.format(atom_type, coord_number))
        return coord_number


    def pick_type_from_neighbor_type(self, neighbor_type):
        """
        Pick type of atom based on neighbor_type and species_properties[neighbor_type]['clustering_proba']
        Probability is set to zero if no atom of the corresponding type left.

        Args:
            neighbor_type: str
                Type of the neighbor based on which type will be picked using
                species_properties[neighbor_type]['clustering_proba'].
                Case sensitive.

        Returns:
            atom_type: str ot None
                str if atoms remain among those that have a non-zero clustering_proba
                to neighbor_type, None otherwise
        """
        remaining_atoms = self.get_remaining_atoms_by_type(as_dict=True)
        p = list(self.species_properties[neighbor_type]['clustering_proba'])
        for type_index, atom_type in enumerate(self.system['atom_types']):
            if not remaining_atoms[atom_type]:
                p[type_index] = 0.0
        if np.sum(p) < self.numeric_tolerance:
            if self.verbosity >= 2:
                print(('There are no atoms left among those that can be '
                      'bonded to {}.').format(neighbor_type))
            return None
        else:
            p_norm = p/np.sum(p)
        try:
            [atom_type_index] = self.rng.choice(
                list(range(len(self.species_properties[neighbor_type]['clustering_proba']))),
                size=1, p=p_norm)
        except ValueError as e:
            print('ValueError in function pick_type_from_neighbor_type: ', e)
            print(('Custering probabilities for a neighbor of type {}, '
                   'given the remaining composition of {} were set to p={}'
                   ).format(neighbor_type, remaining_atoms, p))
            return None

        atom_type = self.system['atom_types'][atom_type_index]

        if self.verbosity >= 2:
            print('Picked type for neighbor of {} atom: {}.'.format(neighbor_type, atom_type))
        return atom_type


    def get_atom_type_index(self, atom_type):
        """
        get atom_type index as defined in self.system['atom_types']
        """
        return self.system['atom_types'].index(atom_type)


    def get_clustering_proba_from_types(self, atom_type_1, atom_type_2):
        """
        Get clustering_proba between atom type A and B (order matters)
        """
        """
        # DEBUGGING
        print(self.species_properties)
        print(self.species_properties[atom_type_1]['clustering_proba'])
        print('{} atom type index: {}'.format(atom_type_2,
              self.get_atom_type_index(atom_type_2)))
        """
        clustering_proba = self.species_properties[atom_type_1][
            'clustering_proba'][self.get_atom_type_index(atom_type_2)]
        return clustering_proba


    def get_connected_neighbors(self, site_index):
        """
        Find neighbors whose distance to center match the expected bond_length

        Tolerance on bond length may be set based on absolute (in Angstroms)
        or relative value (in fraction of the expected bond length)
        """
        neighbors = self.structure.get_neighbors(self.structure.sites[site_index],
                                                 self.search_radius)
        connected_neighbors = []
        for nbr in neighbors:
            dist = nbr.nn_distance
            site_type = self.get_type_from_index(site_index)
            if dist <= self.get_bond_length_with_tol(site_type,
                                                     self.get_atom_type(nbr)):
                if self.verbosity >= 2:
                    print(('Found {} ({}) atom connected to {} ({}) (at {} '
                           '\u212B).').format(self.get_atom_type(nbr),
                          nbr.index, site_type, site_index, dist))
                connected_neighbors.append(nbr)

        return connected_neighbors


    def build_current_site_shell(self, site_index):
        """
        Construct a polyhedron around selected site based on coordination number
        """
        if self.verbosity >= 3:
            print('Running method build_current_site_shell on site {}.'.format(site_index))
        # Pick target coordination number if none has been selected
        if self.structure.sites[site_index].properties['target_coord_number'] is None:
            self.structure.sites[site_index].properties['target_coord_number'] = \
                self.pick_coord_number_from_type(
                    self.get_type_from_index(site_index))
        site_coord_number = self.structure.sites[site_index].properties[
            'target_coord_number']

        # Find neighbors connected to site_index
        nbrs = self.update_connected_neighbors(site_index)

        while len(nbrs) < site_coord_number and (
                self.structure.sites[site_index].properties['treatment_attempts'] <
                self.max_attempts):
            nbrs = self.add_neighbor(site_index, site_coord_number, nbrs)

        if self.structure.sites[site_index].properties['treatment_attempts'] \
                >= self.max_attempts:
            self.structure.sites[site_index].properties['is_treated'] = True
            print(('Max number of attemps ({}) to complete the {}-coordinated '
                   'shell of site {} ({}) has been reached. Switching to '
                   'another site.').format(
                  self.max_attempts, site_coord_number, site_index,
                  self.get_type_from_index(site_index)))

        if len(nbrs) == site_coord_number:
            if self.verbosity >= 1:
                print(('Number of neighbors connected to atom {}{} ' +
                       'match coordination number : {}').format(
                       site_index, self.get_type_from_index(site_index),
                       site_coord_number))
            self.structure.sites[site_index].properties['is_treated'] = True


    def add_neighbor(self, site_index, site_coord_number, nbrs):
        """
        Generic method to add a new neighbor (1st, 2nd, ...4th) to site_index

        Arguments:
            site_index: int
                Considered site index
            site_coord_number: int
                Coordination number of the considered site
            nbrs: list of pymatgen.core.structure.Neighbor
                aleardy-identified connected neighbors of the considered site

        Returns:
            nbrs: list of pymatgen.core.structure.Neighbor
                Updated list of identified connected neighbors
        """
        index_O = site_index
        X_O = self.structure.sites[index_O].coords
        type_O = self.get_type_from_index(site_index)

        if self.verbosity >= 3:
            print(('Running add_neighbor method on site {} ({}), which currently '
                   'has {} detected neighbors for a target coord. number of {}.'
                   ).format(index_O, type_O, len(nbrs), site_coord_number))

        # Set name of new neighbour for prints
        if len(nbrs) == 0:
            new_nbr_name = '1st'
        elif len(nbrs) == 1:
            new_nbr_name = '2nd'
        elif len(nbrs) == 2:
            new_nbr_name = '3rd'
        elif len(nbrs) >= 3:
            new_nbr_name = str(len(nbrs) + 1) + 'th'

        # New neighbour is from now-on designated as N
        # Picking type_N based on type_O, exiting function if none remaining
        type_N = self.pick_type_from_neighbor_type(type_O)
        if type_N is None:
            self.structure.sites[index_O].properties['treatment_attempts'] += 1
            return nbrs

        X_N = None
        if len(nbrs) == 0:
            # Specific procedure for the first neighbor
            max_local_attempts = 20
            local_attempts = 0
            while 1:  # local attempts : trying different A positions withouht changing type_A
                if local_attempts > max_local_attempts:
                    if self.verbosity >= 1:
                        print(('No position avoiding contact has been found after {} local '
                               'attempts for the {} neighbor of site {} ({}).').format(
                              local_attempts, new_nbr_name, index_O, type_O))
                    self.structure.sites[index_O].properties['treatment_attempts'] += 1
                    return nbrs
                elif local_attempts < max_local_attempts:
                    X_N_try = self.find_first_neighbor_position(site_index,
                        site_coord_number, nbrs, nbr_type=type_N, pos_choice_method='random')
                elif local_attempts == max_local_attempts:
                    if self.verbosity >= 2:
                        print(('Maximum number of local attempts ({}) for the random addition '
                               'of the {} neighbor of site {} ({}) has been reached. '
                               'Now trying a position that maximizes distances to nearby '
                               'atoms.').format(max_local_attempts, new_nbr_name, index_O,
                              type_O))
                    # try a position with distance to nearest sites maximized
                    X_N_try = self.find_first_neighbor_position(site_index,
                        site_coord_number, nbrs, nbr_type=type_N, pos_choice_method='max_dist')
                if self.is_space_clear(type_N, X_N_try, [index_O]):
                    if self.verbosity >= 2:
                        print(('A position avoiding contact has been found (after {} local '
                               'attempts) for the {} neighbor of site {} ({}): {}').format(
                              local_attempts, new_nbr_name, index_O, type_O, X_N_try))
                    break
                else:
                    local_attempts += 1

        elif len(nbrs) == 1:
            X_N_try = self.find_second_neighbor_position(site_index,
                site_coord_number, nbrs, nbr_type=type_N)
        elif len(nbrs) == 2:
            X_N_try = self.find_third_neighbor_position(site_index,
                site_coord_number, nbrs, nbr_type=type_N)
        elif len(nbrs) == 3:
            X_N_try = self.find_fourth_neighbor_position(site_index,
                site_coord_number, nbrs, nbr_type=type_N)
        # Add other functions here if some CN > 4 should be considered
        if self.verbosity >= 2:
            print('Trying to insert {} neighbor of type {} at position {}.'.format(
                  new_nbr_name, type_N, X_N_try))

        if self.is_space_clear(type_N, X_N_try,
                               [index_O] + [nbr.index for nbr in nbrs]):
            try:
                self.structure.insert(len(self.structure.sites), type_N, X_N_try,
                                      coords_are_cartesian=True, validate_proximity=True)
                X_N = X_N_try
                index_N = len(self.structure.sites)-1
                ON_vect = X_N - X_O
                if self.verbosity >= 1:
                    print('Site {}:{} added at position {}.'.format(index_N,
                          type_N, X_N))
                self.structure.sites[index_N] = self.initialize_site_properties(
                     self.structure.sites[index_N])
                # Update list of connected neighbors
                nbrs = self.update_connected_neighbors(index_O)
                if len(nbrs) == site_coord_number:
                        self.structure.sites[index_O].properties['is_shell_complete'] = True
                        self.structure.sites[index_O].properties['is_treated'] = True

            except ValueError as e:
                print(('{} neighbor could not be inserted at position {}.\n'
                       'ValueError: {}').format(new_nbr_name, X_N_try, e))
        else:
            if self.verbosity >= 2:
                print('Solution leads to contact between atoms.')

        if X_N is None:
            print('Exiting function add_xth_neighbor with no {} neighbor added.'.format(
                  new_nbr_name))
            self.structure.sites[index_O].properties['treatment_attempts'] += 1
            # struct is unchanged. No need to update nbrs.

        if self.verbosity >= 3:
            print(('Method add_xth_neighbor applied to {} neighbor of site '
                   '{} ({}) will return nbrs of length {}.').format(new_nbr_name,
                  index_O, type_O, len(nbrs)))

        if self.verbosity >= 3:
            print(('Exiting add_neighbor method. Site {} now has '
                   '{} detected neighbors (target coord. number: {}).').format(
                  site_index, len(nbrs), site_coord_number))

        return nbrs


    def is_space_clear(self, tried_type, tried_position, known_nbr_indexes=None):
        """
        Check that no atom exist within distances defined by types

        Args:
            tried_type: str
                type of the atom to be inserted. Will be used to determine
                the contact criteria along with type of detected neighbors
            tried_type: str
                type of the atom to be inserted. Will be used to determine
                the contact criteria along with type of detected neighbors
            known_nbr_indexes: int, list or tuple
                site indexes that should be ignored in the search for contacts

        returns:
            Bool: False if no contact other than with known_nbr_indexes
                  are found
        """

        is_clear = True
        nearby_atoms = self.structure.get_sites_in_sphere(tried_position,
            self.search_radius, include_index=True)
        if self.verbosity >= 3:
            print('{} atoms detected within {} \u212B of tried position {}'.format(
                  len(nearby_atoms), self.search_radius, tried_position))
        # Convert known_nbr_indexes to list if single element
        if not isinstance(known_nbr_indexes, (list, tuple)):
            known_nbr_indexes = [known_nbr_indexes]

        for nbr in nearby_atoms:
            if nbr.index not in known_nbr_indexes:
                contact_dist = (1 + self.rel_bond_length_tol) * \
                    float(self.get_bond_length(self.get_atom_type(nbr),
                    tried_type)) + self.abs_bond_length_tol
                if nbr.nn_distance < contact_dist:
                    if self.verbosity >= 2:
                        print('Atom {}{} within {} \u212B of tried position {}.'.format(
                              nbr.index, self.get_atom_type(nbr), contact_dist,
                              tried_position))
                    is_clear = False
                    break
        return is_clear


    def update_connected_neighbors(self, site_index, include_neighbors=True):
        """
        Update connected_neighbors property of requested site

        Returns:
            nbrs:
                if include_neighbors is True
            Nothing otherwise.
        """
        target_CN = self.structure.sites[site_index].properties[
            'target_coord_number']
        site_type = self.get_type_from_index(site_index)
        nbrs = self.get_connected_neighbors(site_index)
        self.structure.sites[site_index].properties['connected_neighbors'] = [
            nbr.index for nbr in nbrs]
        if len(nbrs) == target_CN:
            self.structure.sites[site_index].properties['is_shell_complete'
                                                        ] = True
        elif len(nbrs) > target_CN:
            self.print(('WARNING: Coordination number of site {} ({}) is {} : '
                        'exceeds target : {}').format(site_index, site_type,
                                                      len(nbrs), target_CN))
        else:  # len(nbrs) > target_CN
            self.structure.sites[site_index].properties['is_shell_complete'
                                                        ] = False

        self.print('Site {} ({}) connected_neighbors property updated : {}'.format(
                   site_index, site_type, self.structure.sites[
                   site_index].properties['connected_neighbors']), verb_th=2)

        if include_neighbors:
            self.print(('Method update_connected_neighbors applied to site {} '
                       'will return nbrs of length {}.').format(site_index,
                       len(nbrs)), verb_th=3)
            return nbrs


    def find_first_neighbor_position(self, site_index, site_coord_number, nbrs,
                                     nbr_type=None, pos_choice_method='random'):
        """
        Find target position for the first neighbor to a selected site in the structure


        """
        # Designate site of focus by O for practicity
        index_O = site_index
        X_O = self.structure.sites[index_O].coords
        type_O = self.get_type_from_index(index_O)
        if nbr_type is None:
            type_A = self.pick_type_from_neighbor_type(type_O)

        else:
            type_A = nbr_type
        # place A atom along a random direction V
        OA = float(self.get_bond_length(type_O, type_A))
        V = self.rng.standard_normal(3)
        if pos_choice_method.lower() == 'random':
            V = V / np.linalg.norm(V)
            OA_vect = V*OA
            X_A = X_O + OA_vect
        elif pos_choice_method.lower() == 'max_dist':
            # Warning: This procedure might be a bit long...
            nearby_sites = self.structure.get_neighbors(self.structure.sites[index_O],
                2*(np.amax(self.bond_length_matrix)*(1+self.rel_bond_length_tol) + \
                self.abs_bond_length_tol) )
            # function to minimize: sum of inverse distance to all nearby_sites around O
            # scaled to expected bond length
            def fun(v):
                f = 0.0
                for nearby_site in nearby_sites:
                    bond_length = self.get_bond_length(type_A,
                                                       self.get_atom_type(nearby_site))
                    f += bond_length/np.linalg.norm(nearby_site.coords - X_O)
                return f
            # impose that v = OA
            result = minimize(fun, V,
                              constrains=({'type': 'eq',
                                           'fun': lambda v: np.linalg.norm(v) - OA}))
            if result.success:
                X_A = result.x + X_O
                if self.verbosity >= 2:
                    print(('A position for a first neighbor to site {} ({}) that '
                           'maximizes distances to nearby sites has been found '
                           'at {}').format(index_O, type_O, X_A))
            else:
                print('Minimization failed. Decide what to do from here.')

        return X_A


    def find_second_neighbor_position(self, site_index, site_coord_number,
                                      nbrs, nbr_type=None):
        """
        Find target position for the second neighbor to a selected site in the structure
        """
        # Designate site of focus by O for practicity
        index_O = site_index
        X_O = self.structure.sites[index_O].coords
        type_O = self.get_type_from_index(index_O)
        index_A = nbrs[0].index
        X_A = nbrs[0].coords
        type_A = self.get_atom_type(nbrs[0])

        if nbr_type is None:
            # Picking type_B based on type_O, exiting function if none remaining
            type_B = self.pick_type_from_neighbor_type(type_O)
        else:
            type_B = nbr_type

        # create a random temporary point T to define the AOT plane
        X_T = X_O + self.rng.standard_normal(3)
        OA_vect = X_A - X_O
        OA = np.linalg.norm(OA_vect)
        n = np.cross(X_T - X_O, OA_vect)  # Normal vector of AOT plane
        # Calculate a, b, c, d parameters in plane equation: ax + by + cz + d = 0
        (a, b, c) = n / np.linalg.norm(n)
        d = -(a*X_T[0] + b*X_T[1] + c*X_T[2])

        theta = (np.pi/180)*self.get_bond_angle_from_CN(site_coord_number)
        if self.verbosity >= 3:
            print(('Using bond angle theta of {:.2f} ° for coordination '
                   'number {} at site {} ({})').format(180*theta/np.pi,
                  site_coord_number, index_O, type_O))
        # B is along a vector v such that v.OA = |v|*|OA|*cos(theta)
        # (x_B, y_B, z_B) = sym.symbols('x_B y_B z_B')
        OB = float(self.get_bond_length(type_B, type_O))
        if self.verbosity >= 2:
            print('OB distance set to {}-{} bond length: {} \u212B.'.format(type_B,
                  type_O, OB))

        def equations(X):
            x, y, z = X
            f1 = a*x + b*y + c*z + d  # B is in OAT plane
            f2 = (x-X_O[0])*OA_vect[0] + (y-X_O[1])*OA_vect[1] + \
                 (z-X_O[2])*OA_vect[2] - OB*OA*np.cos(theta)
            f3 = (x-X_O[0])**2 + (y-X_O[1])**2 + (z-X_O[2])**2 - OB**2
            return (f1, f2, f3)

        if self.verbosity >= 2:
            print(('Looking for solutions for atom B of type {} ' +
                   'around site {} ({})').format(type_B,
                  index_O, type_O))
        solution = fsolve(equations, (0, 0, 0))
        if self.verbosity >= 2:
            print('Solution found : {}.'.format(solution))
        X_B = np.asarray(solution, dtype=float)

        return X_B


    def find_third_neighbor_position(self, site_index, site_coord_number, nbrs,
                                      nbr_type=None):
        """
        Find target position for the third neighbor to a selected site in the structure
        """
        # Designate site of focus by O for practicity
        index_O = site_index
        X_O = self.structure.sites[index_O].coords
        type_O = self.get_type_from_index(index_O)

        # shuffle A-B neighbors order
        pick_order = self.rng.choice(range(2), replace=False, size=2, shuffle=False)
        index_A = nbrs[pick_order[0]].index
        X_A = nbrs[pick_order[0]].coords
        type_A = self.get_atom_type(nbrs[pick_order[0]])
        index_B = nbrs[pick_order[1]].index
        X_B = nbrs[pick_order[1]].coords
        type_B = self.get_atom_type(nbrs[pick_order[1]])

        if nbr_type is None:
            type_C = self.pick_type_from_neighbor_type(type_O)
        else:
            type_C = nbr_type

        # Define constraints to determine position based on existing neighbors
        theta = (np.pi/180)*self.get_bond_angle_from_CN(site_coord_number)
        OC = float(self.get_bond_length(type_O, type_C))
        OA_vect = X_A - X_O
        OA = np.linalg.norm(OA_vect)
        OB_vect = X_B - X_O
        OB = np.linalg.norm(OB_vect)
        def equations(v):
            (x, y, z) = v
            f1 = x*OA_vect[0] + y*OA_vect[1] + z*OA_vect[2] - OC*OA*np.cos(theta)
            f2 = x*OB_vect[0] + y*OB_vect[1] + z*OB_vect[2] - OC*OB*np.cos(theta)
            f3 = x**2 + y**2 + z**2 - OC**2
            return f1, f2, f3

        if self.verbosity >= 2:
            print(('Looking for solution to place atom C of type {} '
                  'around site {} ({})...').format(type_C, index_O, type_O))
        solution = fsolve(equations, (0, 0, 0))
        if self.verbosity >= 2: # debugging only
            print('Solution found: {}'.format(solution))
        X_C = np.asarray(solution, dtype=float) + X_O

        return X_C


    def find_fourth_neighbor_position(self, site_index, site_coord_number,
                                      nbrs, nbr_type=None):
        """
        Find target position for the third neighbor to a selected site in the structure
        """
        # Designate site of focus, and 1st-3rd neighbors by O, A, B, and C
        # for practicity
        index_O = site_index
        X_O = self.structure.sites[index_O].coords
        type_O = self.get_type_from_index(index_O)

        if self.verbosity >= 3:
            print(('Trying to add 4th neighbor to site {} ({}) with coord. '
                   'number {}.').format(index_O, type_O, site_coord_number))

        # shuffle A-B-C neighbors order
        pick_order = self.rng.choice(range(len(nbrs)), replace=False, size=len(nbrs),
                                     shuffle=False)
        index_A = nbrs[pick_order[0]].index
        X_A = nbrs[pick_order[0]].coords
        type_A = self.get_atom_type(nbrs[pick_order[0]])
        index_B = nbrs[pick_order[1]].index
        X_B = nbrs[pick_order[1]].coords
        type_B = self.get_atom_type(nbrs[pick_order[1]])
        index_C = nbrs[pick_order[2]].index
        X_C = nbrs[pick_order[2]].coords
        type_C = self.get_atom_type(nbrs[pick_order[2]])

        if nbr_type is None:
            type_D = self.pick_type_from_neighbor_type(type_O)
        else:
            type_D = nbr_type

        theta = (np.pi/180)*self.get_bond_angle_from_CN(site_coord_number)

        # Set constraints to place 4th atom position based on existing three neighbors
        OD = float(self.get_bond_length(type_O, type_D))
        # (x, y, z) = sym.symbols('x y z')
        OA_vect = X_A - X_O
        OA = np.linalg.norm(OA_vect)
        OB_vect = X_B - X_O
        OB = np.linalg.norm(OB_vect)
        OC_vect = X_C - X_O
        OC = np.linalg.norm(OC_vect)

        # Find vector of norm OD forming a theta angle with OA, OB and OC
        def equations(v):
            (x, y, z) = v
            f1 = x*OA_vect[0] + y*OA_vect[1] + z*OA_vect[2] - OD*OA*np.cos(theta)
            f2 = x*OB_vect[0] + y*OB_vect[1] + z*OB_vect[2] - OD*OB*np.cos(theta)
            f3 = x*OC_vect[0] + y*OC_vect[1] + z*OC_vect[2] - OD*OC*np.cos(theta)
            return f1, f2, f3

        if self.verbosity >= 2:
            print(('Looking for solution to place atom D of type {} '
                  'around site {} ({})...').format(type_D, index_O, type_O))
        solution = fsolve(equations, (0, 0, 0))
        if self.verbosity >= 2:
            print('Solution found: {}.'.format(solution))
        X_D  = np.asarray(solution, dtype=float) + X_O

        return X_D


    def export_vasp_poscar(self, dir_name=''):
        # order structure
        # create directory
        # save poscar
        # write system description in first line
        # TODO: add version
        poscar = Poscar(self.structure,
                        comment=('{} generated with ceramicNetworkBuilder, '
                                 'seed = {}.').format(self.structure.formula,
                                                      self.seed),
                        sort_structure = True)
        if dir_name=='' or dir_name is None:
            dir_name = os.getcwd()
        file_name = os.path.join(dir_name, 'POSCAR')
        poscar.write_file(os.path.join(dir_name, 'POSCAR'))
        self.export_file_name = file_name
        if self.verbosity >= 1:
            print('POSCAR file saved as {}.'.format(self.export_file_name))


    def update_all_connected_neighbors(self):
        """
        Check the coordination shell of all atoms in structure
        """
        for site_index, site in enumerate(self.structure.sites):
            self.update_connected_neighbors(site_index)


    def pick_missing_target_coord_numbers(self):
        for site_index, site in enumerate(self.structure.sites):
            if site.properties['target_coord_number'] is None:
                site.properties['target_coord_number'] = \
                    self.pick_coord_number_from_type(self.get_atom_type(site))

    def pick_atom_to_relocate(self):
        """
        TOBECOMPLETED
        """
        self.print('Function pick_atom_to_relocate().', verb_th=3)
        # Initialize propabilities
        p = np.ones(self.structure.num_sites)
        for site_index, site in enumerate (self.structure.sites):
            nb_of_connected_nbrs = len(site.properties['connected_neighbors'])
            target_CN = site.properties['target_coord_number']

            # Set p to 0 if sphere is complete, highest for most-incomplete
            # spheres
            p[site_index] *= (target_CN-nb_of_connected_nbrs) / target_CN

            # Account for completion of nearest-neighbors coordination spheres
            # excluding the current site_index from the count
            cumul_target_CN = 0
            cumul_nb_of_nbrs = 0
            for connected_site_index in site.properties['connected_neighbors']:
                connected_site = self.structure.sites[connected_site_index]
                # DEBUGGING
                self.print(('site_index {}, connected_site index {}, '
                            'connected_neighbors: {} ({}/{})'
                            ).format(site_index, connected_site_index,
                            connected_site.properties['connected_neighbors'],
                            len(connected_site.properties['connected_neighbors']),
                            connected_site.properties['target_coord_number']),
                           verb_th=3)
                if site_index in connected_site.properties[
                        'connected_neighbors']:  # should always be a check (just a double-check)
                    cumul_target_CN += connected_site.properties[
                        'target_coord_number'] - 1
                    # count connected neighbors excluding site_index
                    cumul_nb_of_nbrs += len(connected_site.properties[
                        'connected_neighbors']) - 1
                else:
                    self.print(('WARNING: in pick_atom_to_relocate: site_index'
                                ' {} not in connected_site.connected_neighbors'
                                ': {}').format(site_index,
                               connected_site.properties['connected_neighbors']
                               ))
            if cumul_target_CN > 0:
                p[site_index] *= (cumul_target_CN-cumul_nb_of_nbrs)/cumul_target_CN
            else:
                self.print('WARNING: cumulated target CN of neighbors is 0.',
                           verb_th=1)

            # other priority checks ? prioritize bigger atoms ?

        # Normalize probabilities:
        p_sum = np.sum(p)
        if p_sum > 0:
             p /= p_sum
        else:
             print('WARNING: picking probabilities in pick_atom_to_relocate '
                   'are all 0. Find out why.')
             return None
        [picked_atom_index] = self.rng.choice(np.arange(len(p)), size=1, p=p)
        self.print('Picked site_to_relocate: {} ({})'.format(
                   picked_atom_index, self.get_type_from_index(
                   picked_atom_index)), verb_th=2)
        self.print('from the following probabilities:', verb_th=3)
        if self.verbosity >= 3:
            for index, proba in enumerate(p):
                self.print('  {}: {}, p = {:.3f}'.format(index,
                           self.get_type_from_index(index), proba), verb_th=3)

        return picked_atom_index


    def pick_atom_to_relocate_old(self):
        """
        Use pandas to sort sites in a sensible order and pick one

        WARNING: the current ordering tends to select sites with the highest
        target_coord_number first (because the difference between actual and
        target coord numbers is highest). A possibility would be to favor atoms
        that are most difficult to insert (bigger atoms).

        The ordering method may be to rigid. Another solution could be to
        calculate probabiliteies based on a score that would reflect
        selection priorities. Already-complete sites would be zero. The
        difficult part will be that relative priority weights will be highly
        arbitrary... SEE HOW IT IS DONE IN pick_site_to_complete
        """
        # Create a pandas dataframe to sort atoms
        data = {'index_in_struct': [], 'connected_nbrs': [], 'nb_of_1st_nbrs': [],
                'missing_1st_nbrs': [], '2nd_nbrs_with_complete_shell': [],
                'missing_2nd_nbrs': []}
        for i, s in enumerate(self.structure.sites):
            data['index_in_struct'].append(str(i))
            data['connected_nbrs'].append(s.properties['connected_neighbors'])
            data['nb_of_1st_nbrs'].append(len(s.properties['connected_neighbors']))
            data['missing_1st_nbrs'].append(s.properties['target_coord_number'] -
                                            len(s.properties['connected_neighbors']))
            # Count 2nd neighbors with complete shells (if > 0 1st neighbor should not be depleted)
            nbrs_with_complete_shell = 0
            missing_2nd_nbrs = 0
            for nbr_index in s.properties['connected_neighbors']:
                nbr_prop = self.structure.sites[nbr_index].properties
                if len(nbr_prop['connected_neighbors']) >= nbr_prop['target_coord_number']:
                    nbrs_with_complete_shell += 1
                missing_2nd_nbrs += nbr_prop['target_coord_number'] - \
                                    len(nbr_prop['connected_neighbors'])
            data['2nd_nbrs_with_complete_shell'].append(nbrs_with_complete_shell)
            data['missing_2nd_nbrs'].append(missing_2nd_nbrs)

        df = pd.DataFrame(data)
        # Shuffle rows and then sort them
        nb_of_sites = len(self.structure.sites)
        randm_indexes = self.rng.choice(np.arange(nb_of_sites),
                                        size=nb_of_sites, replace=False)
        df = df.iloc[randm_indexes]
        df.sort_values(by=['missing_1st_nbrs', 'nb_of_1st_nbrs',
                           '2nd_nbrs_with_complete_shell', 'missing_2nd_nbrs'],
                       ascending=[False, True, True, False], inplace=True,
                       ignore_index=True)
        """
        if self.verbosity >= 3:
            with pd.option_context('display.max_rows', None):
                print(df)
        """

        """
        TODO: Here, instead of zero one could pick atom randomly with
        probablities decreasing as a function of order (down to zero for
        already-complete sites.
        """
        picked_atom_index = int(df.loc[0, 'index_in_struct'])
        self.print(('Method pick_atom_to_relocate will return atom index {}:\n'
                   '{}').format(picked_atom_index, self.structure.sites[
                   picked_atom_index]), verb_th=2)
        return picked_atom_index


    def pick_site_to_complete(self, site_to_relocate_index):
        # TO BE COMPLETED

        # remove site_to_relocate
        # eliminate sites with complete shells
        # calculate probabilities according to priorities:
        #
        #   1. p = 0 if shell is aleady complete
        #   1. favor nearly-complete shell
        #       p = CN/target_CN
        #   2. favor type according to target proba
        #       p = p * clustering_proba (average A-B and B-A ?)
        #   3. take current global clustering proba into account :
        #       p = 0 if current_global_clustering_proba(A-B) >
        #       clustering_proba(A-B)
        #       p = p *(target-current)/target clustering_proba(AB) otherwise
        #   4. favor atom size :
        #       p = p * sum(bond_lengths)/max(sum(bond_lengths))

        self.print(('Function pick_site_to_complete(site_to_relocate_index={}'
                    .format(site_to_relocate_index)), verb_th=2)
        site_to_relocate_type = self.get_type_from_index(site_to_relocate_index)
        # picking probability as a funciton of index
        p = np.ones(self.structure.num_sites)
        for site_index, site in enumerate (self.structure.sites):
            if (site_index == site_to_relocate_index) or \
                    site.properties['is_shell_complete']:
                p[site_index] = 0.0
            else:
                # favor sites whose shell is closest to completion
                p[site_index] *= (len(site.properties['connected_neighbors']) /
                                 site.properties['target_coord_number'])
                site_type = self.get_atom_type(site)
                clustering_proba = self.get_clustering_proba_from_types(
                    site_type, site_to_relocate_type)
                p[site_index] *= clustering_proba
                # TODO: favor types whose clustering to site_to_relocate is far
                # from target
                """
                current_clustering = self.get_current_clustering(site_type, )
                if current_clustering > clustering_proba:
                    p[site_index] = 0.0
                else:
                    p[site_index] *= 1 - (clustering_proba/current_clustering)
                """
                # favor larger atom types (more difficult to insert)
                p[site_index] *= np.sum(self.bond_length_matrix[
                    self.get_atom_type_index(site_type)]) / np.max(
                        np.sum(self.bond_length_matrix, axis=0))

        # Normalize probabilities:
        p_sum = np.sum(p)
        if p_sum > 0:
            p /= p_sum
        else:
            self.print(('WARNING: site-to_complete picking probabilities to '
                        'relocate site {} ({}) are all 0. returning None.'
                        ).format(site_to_relocate_index, site_to_relocate_type))
            # Check whether all sites have been completed.
            return None
        [picked_atom_index] = self.rng.choice(np.arange(len(p)), size=1, p=p)

        if self.verbosity >= 1:
            print('Picking site_to_complete index {} ({})'.format(
                  picked_atom_index, self.get_type_from_index(
                      picked_atom_index)))
        if self.verbosity >= 3:
            print('using the following probabilities:')
            for index, proba in enumerate(p):
                print('  {}: {}, p = {:.3f}'.format(index,
                      self.get_type_from_index(index), proba))

        return picked_atom_index


    def relocate_atom(self, site_index):
        """
        TOBECOMPLETED
        """
        # Pick site-to-complete to relocate site_index onto
        # Designate site_to_complete as A and site_to_relocate as X
        # for convenience
        A_index = self.pick_site_to_complete(site_index)
        if A_index is None:
            self.print(('Site {} ({}) could not be relocated because picking '
                        'picking probabilities of all sites-to-complete are 0.'
                        ).format(site_index, self.get_type_from_index(
                            A_index)))
            return 1, 'Picking probabilities of all sites-to-complete are 0.'
        A_type = self.get_type_from_index(A_index)
        A = self.structure.sites[A_index].coords
        A_target_CN = self.structure.sites[A_index].properties[
            'target_coord_number']
        X_index = site_index
        X_type = self.get_type_from_index(X_index)
        X_target_CN = self.structure.sites[X_index].properties[
            'target_coord_number']
        search_radius = self.get_bond_length_with_tol(A_type,X_type) + np.max(
                self.bond_length_matrix[self.get_atom_type_index(X_type)]) * \
            (1 + self.rel_bond_length_tol) + self.abs_bond_length_tol
        self.print(('Searching neighbors within {} \u212B of site-to-complete '
                   '{} ({}) to relocate site {} ({}).').format(search_radius,
                   A_index, A_type, X_index, X_type), verb_th=2)

        _nbrs = self.structure.get_neighbors(
            self.structure.sites[A_index], search_radius)
        self.print('{} neighbors detected: {}'.format(len(_nbrs),
                   [nbr.index for nbr in _nbrs]), verb_th=3)

        def _is_bondable(nbr):
            if nbr.properties['is_shell_complete'] or (len(nbr.properties[
                    'connected_neighbors']) >= nbr.properties[
                    'target_coord_number']):
                return False
            if np.isclose(self.get_clustering_proba_from_types(
                    self.get_atom_type(nbr), X_type), 0):
                return False
            if np.isclose(self.get_clustering_proba_from_types(
                    X_type, self.get_atom_type(nbr)), 0):
                return False
            return True

        bondable_nbrs = [nbr for nbr in _nbrs if _is_bondable(nbr)]
        nonbondable_nbrs = [nbr for nbr in _nbrs if not _is_bondable(nbr)]
        self.print(('Among {} neighbors of site {} ({}),  {} are potentially'
                    'bondable and {} non-bondable to {}.').format(len(_nbrs),
                   A_index, A_type, len(bondable_nbrs), len(nonbondable_nbrs),
                   X_type), verb_th=2)

        # Given a relocation_position X around site_to_complete A:
        #   1. identify neighbors within X-N + bond_length_tol
        #   2. distinguish :
        #     - bondable neighbors: non-zero clustering proba, non-complete
        #       shell (include possibility to increase CN ?)
        #     - non-bondable neighbors
        #   3. calculate RMSD of X-N vs corresponding bond length
        #   4. verify than X-N to non-bondable neighbors >
        #      bond_length + exclusion_tol
        #
        #   5. Avoid contact.

        def fun(X):
            """
            Function of the atom-to-relocate coordinates X to minimize

            Calculate :
                - MSD of AX vs BL(A,X)
                - MSD of NX vs BL(N,X) for bonded N sites
                - MSD of XAM angles to known nighbors M of site_to_complete A
                - MSD of NXA angles to site_to_complete A
            TODO- MSD of XNL angles to known neighbors L of N (WARNING: get
                  them with get_neighbors function to ensure that correct
                  image is considered).

            Bring distance to N sites identifed as bonded sites as close as
            possible to target X-N bond length and A-X-N bond angles as close
            as possible to target bond angle (given X target coord number).

            The function returns the sum of the RMSD
            """
            # debugging:
            self.print('Minimization function: X = {}'.format(X), verb_th=3)

            # Initialize distance square deviations
            AX = np.linalg.norm(X-A)  # A is the site-to-complete
            AX_0 = self.get_bond_length(X_type, A_type)
            dist_SD = np.square((AX - AX_0) / AX_0)
            dist_count = 1

            # Intitialize angle square deviation
            angle_SD = 0.0
            angle_count = 0
            # Calculate square deviation to bond angles of known nbrs M of A
            # Use a neighbor search rather than indexes o account for
            # periodic boundary conditions
            for M_nbr in self.structure.get_neighbors(
                    self.structure.sites[A_index],
                    self.get_max_bond_length_for_type(A_type,
                                                      include_tol=True)):
                # exclude X in case it was bonded to A in its former position
                # consider only connected neighbors of A
                if M_nbr.index != X_index and (M_nbr.index in
                        self.structure.sites[A_index].properties[
                        'connected_neighbors']):
                    M = M_nbr.coords
                    AM = np.linalg.norm(M-A)
                    XAM = 180/np.pi*np.arccos(np.dot(A-X, M-A) / (AX*AM))
                    XAM_0 = self.get_bond_angle_from_CN(A_target_CN)
                    angle_SD += np.square((XAM - XAM_0) / XAM_0)
                    angle_count += 1
                    M_index = M_nbr.index
                    M_type = self.get_atom_type(M_nbr)
                    self.print(('XAM bond angle to connected M neighbor {} '
                                '({}) of site-to-complete A {} ({}): {:.2f}° '
                                '(vs {:.2f}°)').format(M_index, M_type,
                               A_index, A_type, XAM, XAM_0), verb_th=3)

            for N_nbr in bondable_nbrs:
                # TODO: only take the X_target_CN - 1 first neighbors in the
                # calculation ?? In principle a constraint is here to avoid
                # this situation. Could be better to add a repulsion
                # contribution to the function to minimize.
                # Designate potentially-bonded neighbor as N for convenience
                N = N_nbr.coords
                NX = np.linalg.norm(X-N)
                N_type = self.get_atom_type(N_nbr)
                N_index = nbr.index
                NX_0 = self.get_bond_length(X_type, N_type)
                NX_BL_max = self.get_bond_length_with_tol(X_type, N_type)
                if NX <= NX_BL_max:
                    AX = np.linalg.norm(X-A)
                    AXN = 180/np.pi*np.arccos(np.dot(A-X, N-X) /
                                                (NX * AX) )
                    # TODO: add a relative weigth between angle and
                    # distance deviations
                    dist_SD += np.square((NX - NX_0) / NX_0)
                    dist_count += 1
                    AXN_0 = self.get_bond_angle_from_CN(X_target_CN)
                    angle_SD += np.square((AXN - AXN_0) / AXN_0)
                    angle_count += 1
                    self.print(('Bondable N neighbor {} ({}) at {} \u212B'
                                '(vs {} \u212B) with AXN angle of {}° '
                                '(vs {}°).').format(nbr.index,
                                N_type, NX, NX_0, AXN, AXN_0), verb_th=3)

                    # Calculate square deviations of XNL angles to connected
                    # neighbors L of N. TAKE IMAGE OF N INTO ACCOUNT
                    N_target_CN = self.structure.sites[N_index].properties[
                        'target_coord_number']
                    for L_nbr in self.structure.get_neighbors(
                            self.structure.sites[N_index],
                            self.get_max_bond_length_for_type(N_type,
                            include_tol=True)):
                        # Consider only connected neighbors of N, excluding X
                        # in case it was bonded to N in its former position.
                        if L_nbr.index != X_index and (L_nbr.index in
                                self.structure.sites[N_index].properties[
                                'connected_neighbors']):
                            # Set L coordinates relative to correct image of N
                            L = self.structure.lattice.get_cartesian_coords(
                                L_nbr.frac_coords + N_nbr.image)
                            L_index = L_nbr.index
                            L_type = self.get_atom_type(L_nbr)
                            NL = np.linalg.norm(L-N)
                            XNL = 180/np.pi*np.arccos(np.dot(X-N, L-N) /
                                                      (NL*NX))
                            XNL_0 = self.get_bond_angle_from_CN(N_target_CN)
                            angle_SD += np.square((XNL - XNL_0) / XNL_0)
                            angle_count += 1
                            self.print(('XNL bond angle to connected L {} ({})'
                                        ' neighbor of N {} ({}): {:.2f}° (vs '
                                        '{}°)').format(L_index, L_type,
                                       N_index, N_type, XNL, XNL_0),
                                       verb_th=3)

            # TODO: add a relative weight between angle and distance deviations
            # e.g, a relative variation of 20% for distances is perfectly
            # acceptable but possibly large for angles (larger than difference
            # between 90 and 109 degrees...
            f = 0
            if dist_count > 0:
                f += np.sqrt(dist_SD/dist_count)
            if angle_count > 0:
                f += np.sqrt(angle_SD/angle_count)
            return f


        # Define constraints
        # A-X should be within tol of BL(A,X)
        (AX_min, AX_max) = self.get_bond_length_boundaries(X_type, A_type)
        constraints = [
            {'type': 'ineq',
             'fun': lambda X, A, AX_max: AX_max - np.linalg.norm(X-A),
             'args': (A, AX_max)},
            {'type': 'ineq',
             'fun': lambda X, A, AX_min: np.linalg.norm(X-A) - AX_min,
             'args': (A, AX_min)}
        ]
        # Impose minimum distance of BL(N,X)+tol for neighbors that cannot be
        # bonded to X (clustering_proba = 0 or
        for nbr in _nbrs:
            N_type = self.get_atom_type(nbr)
            N = nbr.coords
            (NX_min, NX_max) = self.get_bond_length_boundaries(X_type, N_type)
            if not _is_bondable(nbr):
                constraints.append(
                    {'type': 'ineq',
                     'fun': lambda X, N, NX_max: np.linalg.norm(X-N) - NX_max - 0.01,
                     'args': (N, NX_max)})
                self.print(('Adding a constraint: non-bondable site {} ({}) '
                            ' distance to {} ({}) should be > {} \u212B.'
                            ).format(nbr.index, N_type, X_index, X_type,
                                     NX_max + 0.01), verb_th=2)
            if _is_bondable(nbr):
                constraints.append(
                    {'type': 'ineq',
                     'fun': lambda X, N, NX_min: np.linalg.norm(X-N) - NX_min,
                     'args': (N, NX_min)})
                self.print(('Adding a constraint: bondable site {} ({}) '
                            'distance to {} ({}) should be > {} \u212B.'
                            ).format(nbr.index, N_type, X_index, X_type,
                                     NX_min), verb_th=2)


        # Number of bondable_nbrs N within BL(N,X)+tol of X should be <=
        # X_target_CN-1
        def fun_max_nb_of_bondable_nbrs(X):
            """
            Function >= 0 if nb_of_bondable_nbrs <= X_target_CN-1
            """
            self.print('fun_max_nb_of_bondable_nbrs constraint function.',
                       verb_th=3)
            nb_of_bondable_nbrs = 0
            for nbr in bondable_nbrs:
                N = nbr.coords
                NX = np.linalg.norm(X-N)
                N_type = self.get_atom_type(nbr)
                NX_BL_max = self.get_bond_length_with_tol(X_type, N_type)
                self.print(('Bondable neighbor {} ({}) at {} \u212B from X '
                            '(BL({},{})_max = {} \u212B)').format(nbr.index,
                            N_type, NX, X_type, N_type, NX_BL_max), verb_th=3)
                if NX <= NX_BL_max:
                    nb_of_bondable_nbrs += 1
                    self.print('nb_of_bondable_nbrs increased to {}'.format(
                               nb_of_bondable_nbrs), verb_th=3)

            self.print(('Returning X_target_CN ({}) - 1 - nb_of_bondable_nb '
                        '({}) = {}').format(X_target_CN, nb_of_bondable_nbrs,
                       X_target_CN - 1 - nb_of_bondable_nbrs), verb_th=3)
            return X_target_CN - 1 - nb_of_bondable_nbrs  # >= 0 if nb_of_bondable_nbrs <= X_target_CN-1

        constraints.append({'type': 'ineq',
                            'fun': fun_max_nb_of_bondable_nbrs})

        # Define bounds : within +/- BL(A,X)+tol of A_x, A_y, A_z
        BL_max = self.get_bond_length_with_tol(A_type, X_type)
        bounds = ( (A[0] - BL_max, A[0] + BL_max) ,
                   (A[1] - BL_max, A[1] + BL_max) ,
                   (A[2] - BL_max, A[2] + BL_max) )

        # Initialize X
        v = 2*self.rng.random(3)-1   # random vector with x, y, z within [-1,1[
        X_0 = A + self.get_bond_length(A_type, X_type)*v/np.linalg.norm(v)

        self.print(('\nStarting SLSQP minimization with:\n'
                    '  - initial coordinates: {}\n'
                    '  - constraints: {}\n'
                    '  - bounds: {}\n\n').format(X_0, constraints, bounds),
                   verb_th=2)

        # MINIMIZE
        result = minimize(fun, X_0, constraints=constraints, bounds=bounds,
                          method='SLSQP', options={'disp': True})
        if result.success:
            X = result.x
            self.print(('Optimum position found to relocate site {} ({}) as ' +
                        'a neighbor of site {} ({}): {}').format(X_index,
                        X_type, A_index, A_type, X), verb_th=2)
            # Update properties of all sites within an appropriate distance
            # of the relocated site (max bond length involving this type )
            self.print('Minimization output:\n{}'.format(result), verb_th=3)

            X_old = self.structure.sites[X_index].coords

            self.print('Site X {} ({}) before relocation: {}, {}'.format(
                X_index, X_type, self.structure.sites[X_index],
                self.structure.sites[X_index].properties), verb_th=2)
            self.structure.sites[X_index].coords = X
            self.update_connected_neighbors(X_index, include_neighbors=False)
            self.print('Site X {} ({}) after relocation: {}, {}'.format(
                X_index, X_type, self.structure.sites[X_index],
                self.structure.sites[X_index].properties), verb_th=2)

            # Update properties of sites within an appropriate distance of the
            # relocated site
            _nbrs = self.structure.get_neighbors(self.structure.sites[X_index],
                self.get_max_bond_length_for_type(X_type, include_tol=True))
            for nbr in _nbrs:
                self.update_connected_neighbors(nbr.index,
                                                include_neighbors=False)

            # TODO: update properties os sites within appropriate distance
            # of former relocated-site position
            _nbrs = self.structure.get_sites_in_sphere(X_old,
                self.get_max_bond_length_for_type(X_type, include_tol=True),
                include_index=True)
            for nbr in _nbrs:
                self.update_connected_neighbors(nbr.index,
                                                include_neighbors=False)

            return 0, 'Relocate_atom sucess.'  # meaning sucess

        else:
            self.print(('No optimum found the relocation of site {} ({}) as ' +
                        'a neighbor of site {} ({}).').format(X_index,
                       X_type, A_index, A_type), verb_th=2)
            return 2, 'Minimization failure: {}'.format(result)  # Meaning minimization error

    def get_sites_by_type(self, types=None, to_single=False):
        """
        Get a list of site indexes for each atom_types in system['atom_types']
        or in given types

        Args:
            types: list, str or None (default is None)
                if None system['atom_types'] will be used. Otherwise the
                function will return a list of list of indexes ordered as in
                types.
            to_single:
                if true a flat list of indexes will be returned if types
                contains a single type. Nhat in this case
                self.structure.indices_from_symbol(ATOM_TYPE) works just as
                fine.

        Returns:
            A list of list of indexes (or a flat list if to_single is True and
            len(types) == 1
        """

        if types is None:
            types = self.system['atom_types']
        else:
            if isinstance(types, str):
                types=[types]
        sites_by_type = [list(self.structure.indices_from_symbol(t)) for t in
                         types]
        if to_single and len(sites_by_type) == 1:
            [sites_by_type] = sites_by_type
        return sites_by_type

    def get_coord_proba(self, types=None, to_single=True):
        """
        TO BE COMPLETED
        """
        coord_proba_by_type = []
        if types is None:
            types = self.system['atom_types']
        else:
            if isinstance(types, str):
                types=[types]
        sites_by_type = self.get_sites_by_type(types=types)
        for type_index, atom_type in enumerate(types):
            coord_proba = []
            for CN,proba0 in enumerate(self.species_properties[atom_type][
                'coord_proba']):
                # count number of atom with CN
                coord_proba.append(len(
                    [i for i in sites_by_type[type_index] if
                     len(self.structure.sites[i].properties[
                         'connected_neighbors']) == CN])
                                   / len(sites_by_type[type_index]))
            coord_proba_by_type.append(coord_proba)

        # Convert list of list to simple list if a single type is requested
        if len(types) == 1 and to_single:
            [coord_proba_by_type] = coord_proba_by_type

        return coord_proba_by_type


    def get_statistics(self):
        """
        Get a dict of statistics for the considered system
        """

        nb_of_complete_sites = self.get_nb_of_complete_sites()
        sites_by_type = self.get_sites_by_type()
        nb_of_sites_by_type = [len(sites_by_type[i]) for i,s in
                               enumerate(sites_by_type)]
        nb_of_sites_with_complete_shell_by_type = [
            self.get_nb_of_complete_sites(of_type=t) for t in
            self.system['atom_types']]
        fractions_of_sites_with_complete_shell_by_type = [n/tot for n,tot in
            zip(nb_of_sites_with_complete_shell_by_type, nb_of_sites_by_type)]

        stat_dict = {
            'seed': self.seed,
            'target_system': self.system,
            'density': self.structure.density,
            'nb_of_sites_with_complete_shell': nb_of_complete_sites,
            'fraction_of_sites_with_complete_shell':
                nb_of_complete_sites/self.structure.num_sites,
            'nb_of_sites_by_type': nb_of_sites_by_type,
            'nb_of_sites_to_target_by_type': [n/n0 for n,n0 in zip(
                nb_of_sites_by_type, self.system["nb_of_atoms_by_type"])],
            'nb_of_sites_with_complete_shell_by_type':
                nb_of_sites_with_complete_shell_by_type,
            'fractions_of_sites_with_complete_shell_by_type':
                fractions_of_sites_with_complete_shell_by_type,
            'coord_proba_by_type': self.get_coord_proba(),
        }
        # TODO: Calculate coord fractions by type
        # TODO: get clustering probability matrix.

        if self.verbosity >= 1:
            for k,v in stat_dict.items():
                print('{}: {}'.format(k, v))

        return stat_dict


    def plot_statistics(self, stat_dict=None):
        """
        Plot statistics on the considered system

        stat_dict should be generated with get_statistics method

        Args:
            stat_dict: dict or None (default is None)
                Statistic dictionary as generated with get_statistics method.
                If None, the method will be used to generate stat_dict on-the-
                fly.

        Returns:
            fig: matplotlib figure handle
        """
        if stat_dict is None:
            stat_dict = self.get_statistics()

        nrows = 2
        ncols = 2
        fig, axes = plt.subplots(nrows, ncols)

        # Compo vs target (hitogram plot)
        (col, row) = (0, 0)
        width = 0.25
        x_shift = width  # /(2-1)
        x = np.arange(len(self.system['atom_types']))
        axes[col, row].bar(x-0.5*x_shift, stat_dict['nb_of_sites_by_type'],
                           width=width, color='r', edgecolor='k')
        axes[col, row].bar(x+0.5*x_shift, self.system['nb_of_atoms_by_type'],
                           width=width, color='k', edgecolor='k',
                           tick_label=self.system['atom_types'])
        # axes[col, row].set_xticks(x)
        # axes[col, row].set_xticklabels(self.system['atom_types'])
        axes[col, row].legend(['Actual composition', 'target composition'])
        axes[col, row].set_ylabel('Number of sites')
        axes[col, row].set_title('Actual vs target composition, seed {}'.format(
            self.seed))

        # Sites with complete shell by type
        (col, row) = (0, 1)
        width = 0.25
        x_shift = width  # /(2-1)
        axes[col, row].bar(x-0.5*x_shift, stat_dict[
            'fractions_of_sites_with_complete_shell_by_type'], color='r',
            edgecolor='k',  tick_label=self.system['atom_types'])
        axes[col, row].set_ylabel('Fraction of sites with complete shell')
        axes[col, row].set_title('Actual vs target composition, seed {}'.format(
            self.seed))
        # TODO: add fraction of complete shells by target CN ?
        axes[col, row].set_title('Fraction of sites with complete shell by type')

        # Coord proba vs target
        (col, row) = (1, 0)
        x_shift_range = 0.2
        x_shift = x_shift_range/(len(self.system['atom_types'])-1)
        tot_shift = -0.5*x_shift_range
        legend = []
        for type_index, atom_type in enumerate(self.system['atom_types']):
            x = tot_shift + np.arange(len(
                stat_dict['coord_proba_by_type'][type_index]))
            y = stat_dict['coord_proba_by_type'][type_index]
            lines = axes[col, row].plot(x, y,'o')
            legend.append('{}'.format(atom_type))
            y = self.species_properties[atom_type]['coord_proba']
            axes[col, row].plot(x, y,'x', color=lines[0].get_color())
            legend.append('{} - target'.format(atom_type))
            tot_shift += x_shift

        axes[col, row].set_xlabel('Coordination number')
        axes[col, row].set_ylabel('Fraction')
        axes[col, row].set_title('{}, {:.3f} g.cm-3, seed={}'.format(
            self.structure.composition.reduced_formula, self.structure.density,
            self.seed))
        axes[col, row].legend(legend)

        # TODO: clustering_proba vs target
        (col, row) = (1, 1)
        axes[col, row].set_title('Clustering probability matrix (IN PROGRESS)')

        return fig


def main(seed=None, input_file='input.json', verbosity=1, abs_bond_length_tol=0.0,
         rel_bond_length_tol=0.1, max_attempts=20,
         max_iterations_step1=1000, max_iterations_step2=500,
         numeric_tolerance=1e-5, search_radius=3.0,
         first_atom_intern_coords=(0.5, 0.5, 0.5), visualizer='vesta',
         export_format='poscar'):

    cnbd = ceramicNetworkBuilderData(seed=seed, input_file=input_file,
                                     verbosity=verbosity,
                                     abs_bond_length_tol=abs_bond_length_tol,
                                     rel_bond_length_tol=rel_bond_length_tol,
                                     numeric_tolerance=numeric_tolerance,
                                     search_radius=search_radius,
                                     visualizer=visualizer,
                                     export_format=export_format,
                                     max_iterations_step1=max_iterations_step1,
                                     max_iterations_step2=max_iterations_step2,
                                     max_attempts=max_attempts)

    cnbd.print_versions()
    cnbd.save_sample_json_input_file()

    nb_of_atoms = sum(cnbd.system['nb_of_atoms_by_type'])
    remaining_atoms_by_type = cnbd.system['nb_of_atoms_by_type']
    nb_of_atoms_by_type = np.asarray(cnbd.system['nb_of_atoms_by_type'])

    cnbd.initialize_structure(first_atom_intern_coords)

    # ************************************************************************
    # Step 1: place atoms with strict constraints on
    #     - bond lengths : equal to expected value based on atom types
    #     - bond angles : equal to expected values based on coordination number
    #     - min. dist between non-bonded atoms: > bond_length + (abs/rel tol)
    # ************************************************************************
    iteration_index = -1
    while cnbd.structure.num_sites < np.sum(nb_of_atoms_by_type):
        iteration_index += 1
        if iteration_index >= cnbd.max_iterations_step1:
            print(f'Maximum number of iterations ({cnbd.max_iterations_step1}) reached.')
            break
        if cnbd.verbosity >= 1:
            print(f'Starting iteration number {iteration_index}.')

        # pick first atom not-yet-treated
        sites_not_yet_treated = [index for (index, site) in enumerate(cnbd.structure.sites)
                                 if (not site.properties['is_treated']) and
                                 (site.properties['treatment_attempts'] < cnbd.max_attempts)]
        if cnbd.verbosity >= 2:
            print('sites_not_yet_treated = ', sites_not_yet_treated)
        if len(sites_not_yet_treated):
            # Treat sites in order of creation until one type of atoms is exhausted
            remaining_atoms = cnbd.get_remaining_atoms_by_type()
            if min(remaining_atoms) > 0:
                current_site_index = sites_not_yet_treated[0]
            else:  # end then randomly
                [current_site_index] = cnbd.rng.choice(sites_not_yet_treated, size=1)
        else:
            print('All sites have been treated. A new random site must be added.')
            print('IMPLEMENTATION IN PROGRESS.')
            break
            # Add a new atom from scratch at a random position ?
            # It could be interesting at this point to start forcing the
            # insertion of atoms


        current_site_type = cnbd.get_type_from_index(current_site_index)
        # Set coord based on probabilities (easiest: ignore other sites)i
        current_site_CN = cnbd.pick_coord_number_from_type(current_site_type)
        cnbd.structure.sites[current_site_index].properties['target_coord_number'] = \
            current_site_CN
        if cnbd.verbosity >= 1:
            print('Current site[{}] : {} with coordination number {}.'.format(
                  current_site_index, current_site_type, current_site_CN))

        cnbd.build_current_site_shell(current_site_index)

        if cnbd.verbosity >= 2:
            print('\nStructure at the end of iteration {}:\n{}\n'.format(
                  iteration_index, cnbd.structure))

    cnbd.update_all_connected_neighbors()
    cnbd.pick_missing_target_coord_numbers()

    # FOR TESTING PURPOSES: RESET VERBOSITY TO USER-VALUE
    cnbd.verbosity = verbosity

    cnbd.print('\nStructure at the end of Step 1:\n{}\n'.format(
               cnbd.structure), verb_th=2)

    # ************************************************************************
    # Step 2: force atoms at positions matching
    #     - bond lengths : equal to expected value based on atom types
    #     - bond angles : equal to expected values based on coordination number
    #     - min. dist between non-bonded atoms: > bond_length + (abs/rel tol)
    # ************************************************************************


    iteration_index = -1
    while 1:
        iteration_index += 1
        if iteration_index >= cnbd.max_iterations_step2:
            cnbd.print(('Maximum number of iterations for step 2: site '
                        'relocations ({}) has been reached.').format(
                       cnbd.max_iterations_step2))
            break
        # Count sites with incomplete shells.
        if cnbd.get_nb_of_incomplete_sites() == 0:
            cnbd.print(('All sites have a complete shell after {} step 2: site'
                        ' relocations.').format(iteration_index))
            break

        # TODO : improve pick_atom_to_relocate() with a choice and probabilities
        # defining priorities (as in pick_site_to_complete())
        current_site_index = cnbd.pick_atom_to_relocate()
        if current_site_index is None:
            cnbd.print(('All sites in structure have a 0 probability of being '
                        'relocated. Hopefully this is because all sites that '
                        'could possibly be relocated have been relocated.\n'
                        'Exiting step 2.'),
                       verb_th=1)
            break
        exit_code, exit_msg = cnbd.relocate_atom(current_site_index)
        if not exit_code :
            cnbd.print(('\n' + 20*'*' + '\nStructure after relocation-iteration '
                        '\# {} (site {}):\n' + 20*'*' + '\n{}').format(
                       iteration_index, current_site_index, cnbd.structure),
                       verb_th=3)
        else:
            cnbd.print(('relocate_atom function failed with exit_code {}: {}'
                        ).format(exit_code, exit_msg), verb_th=2)
            cnbd.print('Trying to relocate another atom.', verb_th=1)



    # Rank candidate atoms-to-move by :
    #   (1) nb_of_missing_nbrs (descending order)
    #           = site.properties['target_coord_number'] - len(site.properties['connected_neighbors'])
    #       At first, selected sites will have 1 neighbor, and among those we will chose
    #   (2) nb_of_missing_nbrs within site.properties['connected_neighbors'] (exluding site-of-focus)
    # The idea is to deplete already-depleted sites.


    # Step 3: Try to repeat step 1 and 2 for remaining atoms until none can be inserted.

    if cnbd.verbosity >= 1:
        print('\n' + 20*'*' + '\n' + 'Final structure :' + '\n' + 20*'*' + '\n',
              cnbd.structure)
        print('Density = {:.3f} g/cm-3'.format(cnbd.structure.density))

    fig = cnbd.plot_statistics()

    # print('cnbd = ', cnbd)
    if cnbd.export_format.lower() == 'poscar':
        cnbd.export_vasp_poscar(dir_name=os.getcwd())
    cnbd.visualize()

    plt.show()

# Set label of the group containing the structures on which analyses should be
# performed.
# TODO: insert a density option to set density in g/cm-3 to a specifed value
# by adjusting the cell length parameters.

@click.command('cli')
@click.option('-s', '--seed', default=None, type=int,
              help='Seed for random number generator.')
@click.option('-i', '--input_file', default='input.json', type=str,
              help='Input file name.')
@click.option('-r', '--rel_bond_length_tol', default=0.1, type=float,
              help=('Relative contact tolerance in fraction of expected bond '
                    'length.'))
@click.option('-m', '--max_iterations_step1', default=1000, type=int,
              help='Maximum number of iterations for step 1.')
@click.option('-M', '--max_iterations_step2', default=500, type=int,
              help='Maximum number of iterations for step 2.')
@click.option('-a', '--max_attempts', default=10, type=int,
              help='Maximum number of completion attempts per site.')
@click.option('-V', '--visualizer', default='ase', type=str,
              help='Visualizer (vesta, ase)')
@click.option('-e', '--export_format', default='poscar', type=str,
              help='Export format: poscar')
@click.option('-v', '--verbosity', default=1, type=int,
              help='Verbosity level. Default is 1')
def cli(seed, input_file, rel_bond_length_tol, max_iterations_step1,
        max_iterations_step2, max_attempts, visualizer, export_format, verbosity):
    """
    Program for the generation of a ceramic network.
    """
    if (seed is not None) and (not isinstance(seed, int)):
        sys.exit('Seed for random number generator should be an integer.')

    input_file = os.path.abspath(input_file)
    if not os.path.isfile(input_file):
        sys.exit(f'Input file {input_file} not found.')

    if not isinstance(rel_bond_length_tol, float) or rel_bond_length_tol < 0:
        sys.exit('rel_bond_length_tol should be a positive float')

    if not isinstance(max_iterations_step1, int) or rel_bond_length_tol < 0:
        sys.exit('max_iterations_step1 should be a positive integer')

    if not isinstance(max_iterations_step2, int) or rel_bond_length_tol < 0:
        sys.exit('max_iterations_step2 should be a positive integer')

    verbosity_error_msg = 'Verbosity should be a positive integer'
    if not isinstance(verbosity, int):
        sys.exit(verbosity_error_msg)
    elif verbosity < 0:
        sys.exit(verbosity_error_msg)

    if visualizer.lower() not in ['ase', 'vesta']:
        sys.exit('Currently supported visualizers include : \n- ase, \n- vesta ')

    #TODO : add other user inputs here.

    if verbosity >= 1:
        print('Running script {}'.format(os.path.basename(__file__)))

    main(seed=seed, input_file=input_file,
         rel_bond_length_tol=rel_bond_length_tol,
         max_iterations_step1=max_iterations_step1,
         max_iterations_step2=max_iterations_step2,
         max_attempts=max_attempts,
         visualizer=visualizer.lower(),
         export_format=export_format, verbosity=verbosity)


if __name__ == '__main__':
    cli()


