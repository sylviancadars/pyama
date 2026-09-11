"""
Build a ceramic matrix atom by atom in a periodic cell

Sylvian Cadars, Fabien Mortier, Assil Bouzid
Institute of Research on Ceramics (IRCER), University of Limoges, CNRS, France
sylvian.cadars@unilim.fr

Please cite:
Fabien Mortier, Sylvian Cadars, Marwan Ben Miled, Olivier Masson, Guido Ori, Mauro Boero, Yun Wang, Samuel Bernard, Assil Bouzid, First-principles modeling of polysilazane-derived SiCNH ceramics: insights into the organization of the free-carbon phase, Phys. Chem. Chem. Phys. (2026) 28 (28): 17314–17332.
https://doi.org/10.1039/d5cp04954g

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
from pymatgen.core.composition import Composition
from pymatgen.core.units import FloatWithUnit

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

__version__ = '2026.09.04_01'

# Maximum coordination number currently accepted by the program
# Increasing this number will require quite a bit of work...
MAX_COORD_NUMBER = 4
ANOMALOUS_DIST_THLD = 5.0

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
        - IN PROGRESS: add terminal atoms a posteriori (step 3)
        - Add a function to export in extxyz with site_properties stored.
          This requires creating a structure copy where struct_copy.sites[site_index].properties["connected_neighbors"]
          is removed before the ASE Atoms object is created with pymatgen.io.ase import AseAtomsAdaptor
          and inserted a posteriori (in atoms.info ?)

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
                 max_iterations_step3=500,
                 visualizer='ase', export_format='poscar', verbosity=1,
                 print_to_console=True, print_to_file=False,
                 add_terminal_atoms_last=True):
        """
        Initialization
        """
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
        # TODO: max_iterations_step1,2,3 are currently not used directly in
        #       ceramicNetworkBuilderData class but in main. Theey could be removed
        #       from the class properties.
        self.max_iterations_step1 = max_iterations_step1
        self.max_iterations_step2 = max_iterations_step2
        self.max_iterations_step3 = max_iterations_step3
        self.search_radius = search_radius
        self.abs_bond_length_tol = abs_bond_length_tol    # in Angstroms
        self.rel_bond_length_tol = rel_bond_length_tol    # in fraction of the tabulated bond length
        self.visualizer = visualizer
        self.export_format = export_format
        self.verbosity = self.verbosity
        self.print_to_console = print_to_console
        self.print_to_file = print_to_file
        self._output_text = []   # list of strings to be ultimately written in file
        self.add_terminal_atoms_last = add_terminal_atoms_last

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
        # TODO: add authors and citation: Mortier et al. PCCP 2026
        self.print('ceramicNetworkBuilder version: {}'.format(__version__))
        self.print('Pymatgen.core path: {}'.format(core.__path__))
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

        self.print("Starting initialize_structure function.", verb_th=3)

        # Build system:
        lattice = Lattice.from_parameters(self.system['cell_lengths'][0],
            self.system['cell_lengths'][1], self.system['cell_lengths'][2],
            self.system['cell_angles'][0], self.system['cell_angles'][1],
            self.system['cell_angles'][2])

        # Construct structure with first atom
        # Pick atom type randomly with probabilities according to target system compo
        nb_of_atoms_by_type = np.asarray(self.system['nb_of_atoms_by_type'])

        while 1:
            atom_type = self.rng.choice(self.system['atom_types'],
                                        p=nb_of_atoms_by_type / np.sum(nb_of_atoms_by_type))

            self.print(f"Trying to initialize structure with an atom of type {atom_type}",
                       verb_th=2)

            if first_atom_intern_coords is None:
                self.structure = Structure(lattice=lattice, species=[atom_type],
                                           coords=[self.rng.random(3)])
            else:
                self.structure = Structure(lattice=lattice, species=[atom_type],
                                           coords=[first_atom_intern_coords])

            # Initialize site properties
            try:
                self.initialize_site_properties(self.structure.sites[0],
                                                skip_terminal=self.add_terminal_atoms_last)
            except ValueError as e:
                self.print(f"Structure could not be initialized with an atom "
                           f"of type {atom_type}: {e}. Retrying.", verb_th=2)
                continue

            site_target_CN = self.structure.sites[0].properties["target_coord_number"]

            if not self.add_terminal_atoms_last:
                self.print(f"The structure has been initialized with an {atom_type} atom "
                           f"with target coordination number {site_target_CN} "
                           f"at position:{self.structure.sites[0].coords}.", verb_th=2)
                break
            if (self.add_terminal_atoms_last and site_target_CN > 1):
                self.print(f"The structure has been initialized with non-terminal "
                           f"{atom_type} atom with target coordination number {site_target_CN} "
                           f"at position: {self.structure.sites[0].coords}.", verb_th=2)
                break
            else:  # terminal atom detected
                self.print(f"Terminal atoms (i.e. with target_CN = 1) should be added last. "
                           f"Structure cannot be initialized with an {atom_type} atom with "
                           f"target coordination number {site_target_CN}. A new initial atom "
                           f"will be picked.",
                           verb_th=2)
                self.structure.remove_sites([0])

        self.print("End of initialize_structure function.", verb_th=3)

    def pick_type_from_remaining(self, skip_terminal=False):
        """ 
        Pick type with probabilities based on the number of remaining atoms 
        
        If skip_terminal, the probability to select a potentially-terminal type 
        will take into account the probability that the target CN is >= 2.
        """
        remaining_atoms_by_type = self.get_remaining_atoms_by_type(as_dict=True)
        
        if skip_terminal:
            remaining_atoms_by_type = {t: n for t, n in remaining_atoms_by_type.items()
                                      if not self.is_strictly_terminal_type(t)}

        # Calculate probabilities from remaining atoms
        p = np.ones(len(remaining_atoms_by_type.keys()))
        n_remaining_atoms = np.sum([n for n in remaining_atoms_by_type.values()])

        if n_remaining_atoms == 0:
            self.print(f"\nStructure composition is now {self.structure.composition}. There "
                       f"are no more atoms to select.", verb_th=1)
            return None
        
        for type_index, atom_type in enumerate(remaining_atoms_by_type.keys()):
            if skip_terminal and self.is_potentially_terminal_type(atom_type):
                # Take into account probabilities that CN >= 2:
                p[type_index] *= (sum(self.species_properties[atom_type]["coord_proba"][2:]) 
                                  * remaining_atoms_by_type[atom_type]) / n_remaining_atoms
                self.print(f"The probability to pick a potentiall-terminal {atom_type} atom "
                           f"of target CN >= 2 within {n_remaining_atoms} remaining atoms is {p:.3f}", 
                           verb_th=2)
            else:
                p[type_index] *= remaining_atoms_by_type[atom_type] / n_remaining_atoms

        p = p / np.sum(p)
        picked_atom_type = self.rng.choice(list(remaining_atoms_by_type.keys()), 
                                           p=p)

        self.print(f"Type {picked_atom_type} has been picked from remaining atoms "
                   f"{'skipping terminal' if skip_terminal else ''}", 
                   verb_th=2)

        return picked_atom_type

    def add_random_site(self, atom_type=None, max_n_attempts=10000, skip_terminal=False):

        self.print(f"Trying to add a random site in the structure", verb_th=1)

        attempt_index = -1
        while 1:
            attempt_index += 1

            if attempt_index > max_n_attempts:
                self.print(f"The maximum number of attempts for add_random_site ({max_n_attempts}) "
                           f"has been reached. Returning None.", verb_th=1)
                return None

            if atom_type is None:
                atom_type = self.pick_type_from_remaining(skip_terminal=skip_terminal)
                if atom_type is None:
                    self.print("Cannot add a new random atom.", verb_th=1)
                    return None

            if attempt_index % 100 == 0:
                self.print(f"(attempt {attempt_index + 1}/{max_n_attempts}).",
                           verb_th=2)

            atom_intern_coord = self.rng.random(3)
            atom_cart_coord = atom_intern_coord @ self.structure.lattice.matrix

            if self.is_space_clear(atom_type, atom_cart_coord):
                # Insert atom
                try:
                    self.structure.insert(len(self.structure.sites), atom_type, atom_intern_coord,
                                      coords_are_cartesian=False, validate_proximity=True)

                    site_index = len(self.structure.sites) - 1
                    site = self.structure.sites[site_index]
                    self.initialize_site_properties(site, skip_terminal=skip_terminal)
                    self.update_connected_neighbors(site_index)
                    site_target_CN = self.get_target_CN_from_index(site_index)
                    
                    self.print(f"A new isolated site {atom_type}{site_index} with target "
                               f"coordination number {site_target_CN} has been added at "
                               f"randomly-picked cartesian position {atom_cart_coord}.", 
                               verb_th=1)

                    return site
                
                except ValueError as e:
                    self.print(f"{atom_type} atom could not be inserted at cartesian position "
                               f"{atom_cart_coord}: {e}.", 
                               verb_th=3)
            else:
                self.print(f"{atom_type} atom could not be inserted at (cartesian) "
                           f"position {atom_cart_coord}: space is not clear.", 
                           verb_th=3)
        

    def visualize(self):

        if self.visualizer.lower() == 'ase':
            # To avoid error upon conversion to ASE, one should first remove site_properties
            # or to retain site_propertis, manually convert site_properties["connected_neihghbors"]
            # to an atoms.info rather than an atoms.arrays
            struct_copy = self.structure.copy()
            for site in struct_copy.sites:
                site.properties.pop("connected_neighbors")
            ase_struct = AseAtomsAdaptor.get_atoms(struct_copy)
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


    def get_atom_type(self, site_or_index):
        if isinstance(site_or_index, int):
            site = self.structure.sites[site_or_index]
        else:
            site = site_or_index
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


    def initialize_site_properties(self, site, skip_terminal=False, **kwargs):
        """
        Initialize the custom properties of a site

        Args:
            site: pymatgen.core.sites.PeriodicSite or Site

        TODO:
            (if necessary) use a kwarg to change a specific property to non-default value.
        """

        self.print("Starting function initialize_site_properties.", verb_th=3)

        if not isinstance(site, (PeriodicSite, Site)):
            raise(TypeError,
                  'Argument site should be of type pymatgen.core.site.(Periodic)Site.')
        site_type = self.get_atom_type(site)
        site.properties['is_shell_complete'] = False
        site.properties['is_treated'] = False
        site.properties['treatment_attempts'] = 0

        if skip_terminal and self.is_strictly_terminal_type(site_type):
            raise ValueError(f"Atom type {site_type} is necessarily terminal.")

        site.properties['target_coord_number'] = self.pick_coord_number_from_type(
            site_type, skip_terminal=skip_terminal)

        site.properties['connected_neighbors'] = []

        self.print(f"Properties of site {site} have been initialized:\n{site.properties}",
                   verb_th=3)

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

    def pick_coord_number_from_type(self, atom_type, skip_terminal=False):
        """
        pick coordination number randomly based on species_properties[atom_type]['coord_proba']

        atom_type is case sensitive.
        """
        if self.verbosity >= 4:
            print('Running pick_coord_number_from_type method for an atom of type {}.'.format(
                  atom_type))
            print('Selecting among type indexes: ', list(range(len(self.species_properties[atom_type]['coord_proba']))))
            print('with probabilities: ', self.species_properties[atom_type]['coord_proba'])

        target_CN_proba = self.get_target_CN_proba(atom_type)

        self.print(f"Coordination number will be picked from species_properties"
                   f"['{atom_type}']['coord_proba'] = {target_CN_proba}", verb_th=3)

        if skip_terminal:
            if self.is_strictly_terminal_type(atom_type):
                raise ValueError(f"Atom type {atom_type} is necessarily terminal.")

            coord_number = self.rng.choice([i for i in range(len(target_CN_proba)) if i != 1],
                                           p=[tcnp for i, tcnp in enumerate(target_CN_proba) if i != 1])

        else:
            coord_number = self.rng.choice(list(range(len(target_CN_proba))),
                                             p=target_CN_proba)

        # TODO ? take remaining atoms into account ?

        self.print(f"Picked coord number for {atom_type} atom: {coord_number}.", verb_th=2)

        return coord_number


    def pick_type_from_neighbor_type(self, neighbor_type, skip_terminal=False):
        """
        Pick type of atom based on neighbor_type and species_properties[neighbor_type]['clustering_proba']
        Probability is set to zero if no atom of the corresponding type left.

        Args:
            neighbor_type: str
                Type of the neighbor based on which type will be picked using
                species_properties[neighbor_type]['clustering_proba'].
                Case sensitive.
            skip_terminal: bool (default is False)
                Whether stricly-terminal atom types should be skipped.

        Returns:
            atom_type: str ot None
                str if atoms remain among those that have a non-zero clustering_proba
                to neighbor_type, None otherwise
        """
        remaining_atoms = self.get_remaining_atoms_by_type(as_dict=True)

        if skip_terminal:
            # TODO: skip potentially-terminal types for which the target amount of non-terminal
            #       sites has been reached.  -> function should_skip_potentially_terminal()
            skip_types = [t for t in remaining_atoms.keys() if not self.is_strictly_terminal_type(t)]
            remaining_atoms = {t: n for t, n in remaining_atoms if t not in skip_types}
            p = [self.species_properties[neighbor_type]['clustering_proba'][self.get_atom_type_index(t)]
                 for t in remaining_atoms.keys() if t not in skip_types]
            self.print(f"Skipping terminal atoms. Type of {neighbor_type}-atom neighbor will be picked"
                       f"from remaining_atoms {remaining_atoms} with probabilities {p}", verb_th=3)
        else:
            p = list(self.species_properties[neighbor_type]['clustering_proba'])

        for type_index, atom_type in enumerate(self.system['atom_types']):
            if not remaining_atoms[atom_type]:
                p[type_index] = 0.0
        if np.sum(p) < self.numeric_tolerance:
            self.print(f"There are no atoms left among those that can be "
                       f"bonded to {neighbor_type}.", verb_th=2)
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

        self.print(f'Picked type for neighbor of {neighbor_type} atom: {atom_type}.',
                   verb_th=2)
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
                self.print(f"Found {self.get_atom_type(nbr)}{nbr.index} atom connected "
                           f"to {site_type}{site_index} (at {dist} \u212B)", verb_th=3)
                connected_neighbors.append(nbr)

        return connected_neighbors


    def build_current_site_shell(self, site_index, skip_terminal=False):
        """
        Construct a polyhedron around selected site based on coordination number
        """
        self.print(f'Running method build_current_site_shell on site {site_index}.', 
                   verb_th=3)

        site_type = self.get_type_from_index(site_index)

        # Pick target coordination number if none has been selected
        if self.structure.sites[site_index].properties['target_coord_number'] is None:
            self.structure.sites[site_index].properties['target_coord_number'] = \
                self.pick_coord_number_from_type(site_type,
                                                 skip_terminal=self.add_terminal_atoms_last)

        site_type = self.get_type_from_index(site_index)
        site_target_CN = self.get_target_CN_from_index(site_index)

        # Find neighbors connected to site_index
        nbrs = self.update_connected_neighbors(site_index)
        site_current_CN = len(nbrs)

        while site_current_CN < site_target_CN and (
                self.structure.sites[site_index].properties['treatment_attempts'] <
                self.max_attempts):
            nbrs = self.add_neighbor(site_index, site_target_CN, nbrs,
                                     skip_terminal=skip_terminal)
            site_current_CN = len(nbrs)

        if self.structure.sites[site_index].properties['treatment_attempts'] \
                >= self.max_attempts:
            self.structure.sites[site_index].properties['is_treated'] = True
            
            self.print(f"The maximum number of attemps ({self.max_attempts}) to complete "
                       f"the shell of site {site_type}{site_index}, currently "
                       f"{site_current_CN}/{site_target_CN} has been reached. "
                       f"Switching to another site.", verb_th=1)

        if site_current_CN == site_target_CN:
            self.print(f"The coordination number of site {site_type}{site_index} "
                       f"is complete : {site_current_CN}/{site_target_CN}.", verb_th=1)
            self.structure.sites[site_index].properties['is_treated'] = True


    def add_neighbor(self, site_index, site_coord_number, nbrs,
                     neighbor_type=None, skip_terminal=False):
        """
        Generic method to add a new neighbor (1st, 2nd, ...4th) to site_index

        The current atom (to which neighbor should be added) is designated as O,
        the new neighbor (to be added as N), and

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
        target_CN_O = site_coord_number

        self.print(f"Running add_neighbor method on site {type_O}{index_O} with "
                   f"current/target coordination number {len(nbrs)}/{target_CN_O}", 
                   verb_th=3)

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
        if neighbor_type is None:
            # Picking type_N based on type_O, exiting function if none remaining
            type_N = self.pick_type_from_neighbor_type(type_O)
        else:
            type_N = neighbor_type

        if type_N is None:
            self.structure.sites[index_O].properties['treatment_attempts'] += 1
            return nbrs

        if skip_terminal and self.add_terminal_atoms_last:
            # Exit if picked type_N corresponds to an atom with target_CN always equal to 1
            if self.is_strictly_terminal_type(type_N):
                self.structure.sites[index_O].properties['treatment_attempts'] += 1
                self.print(f"Terminal {type_N} atom will be added last. treatment attempt "
                           f"{self.structure.sites[index_O].properties['treatment_attempts']} "
                           f"out of {self.max_attempts} failed.",
                           verb_th=3)
                return nbrs

        X_N = None
        if len(nbrs) == 0:
            # Specific procedure for the first neighbor
            max_local_attempts = 20
            local_attempts = 0
            while 1:  # local attempts : trying different A positions withouht changing type_A
                if local_attempts > max_local_attempts:
                    self.print(f"No position avoiding contact has been found after {local_attempts} "
                               f"local attempts for the {new_nbr_name} neighbor of site {type_O}{index_O}.", 
                               verb_th=2)
                    self.structure.sites[index_O].properties['treatment_attempts'] += 1
                    return nbrs
                elif local_attempts < max_local_attempts:
                    X_N_try = self.find_first_neighbor_position(site_index,
                        target_CN_O, nbrs, nbr_type=type_N, pos_choice_method='random')
                elif local_attempts == max_local_attempts:
                    self.print(f"The maximum number of local attempts ({max_local_attempts}) for "
                               f"the random addition of the {new_nbr_name} neighbor of site "
                               f"{type_O}{index_O}) has been reached. Now trying a position "
                               f"that maximizes distances to nearby atoms.", verb_th=2)
                    # try a position with distance to nearest sites maximized
                    X_N_try = self.find_first_neighbor_position(site_index,
                        target_CN_O, nbrs, nbr_type=type_N, pos_choice_method='max_dist')
                if self.is_space_clear(type_N, X_N_try, [index_O]):
                    self.print(f"A position avoiding contact has been found (after {local_attempts} "
                               f"local attempts) for the {new_nbr_name} neighbor of site "
                               f"{type_O}{index_O}: {X_N_try}", verb_th=2)
                    break
                else:
                    local_attempts += 1

        elif len(nbrs) == 1:
            X_N_try = self.find_second_neighbor_position(site_index,
                target_CN_O, nbrs, nbr_type=type_N)
        elif len(nbrs) == 2:
            X_N_try = self.find_third_neighbor_position(site_index,
                target_CN_O, nbrs, nbr_type=type_N)
        elif len(nbrs) == 3:
            X_N_try = self.find_fourth_neighbor_position(site_index,
                target_CN_O, nbrs, nbr_type=type_N)
        # TODO: Add other functions here if some CN > 4 should be considered

        if X_N_try is None:
            self.print(f"Exiting function add_neighbor with no {new_nbr_name} neighbor "
                       f"added to site {type_O}{index_O} with current/target CN "
                       f"{len(nbrs)}/{target_CN_O}", 
                       verb_th=1)  # TODO: increase ver_th to 2
            self.structure.sites[index_O].properties['treatment_attempts'] += 1
            # struct is unchanged. No need to update nbrs.
            return nbrs  # TODO: consider returning None

        self.print(f"Trying to insert {new_nbr_name} neighbor of type {type_N} "
                   f"at position {X_N_try}.", verb_th=2)

        if self.is_space_clear(type_N, X_N_try,
                               [index_O] + [nbr.index for nbr in nbrs]):
            try:
                # Check for anomalous distances.
                ON_dist = np.linalg.norm(X_N_try - X_O)
                if ON_dist > ANOMALOUS_DIST_THLD:
                    raise ValueError(f"Anomalous distance has been detected between site {type_O}{index_O} "
                                     f"and optimized {type_N} neighbor position {X_N_try}: {ON_dist:.2f} \u212B")

                self.structure.insert(len(self.structure.sites), type_N, X_N_try,
                                      coords_are_cartesian=True, validate_proximity=True)
                X_N = X_N_try
                index_N = len(self.structure.sites)-1
                ON_vect = X_N - X_O

                self.structure.sites[index_N] = self.initialize_site_properties(
                    self.structure.sites[index_N])

                # Update list of connected neighbors
                nbrs = self.update_connected_neighbors(index_O)
                current_CN_O = len(nbrs)
                if current_CN_O == target_CN_O:
                    self.structure.sites[index_O].properties['is_shell_complete'] = True
                    self.structure.sites[index_O].properties['is_treated'] = True
                    self.print(f"The shell of {type_N}{index_N} neighbor {type_O}{index_O} "
                               f"is now complete: {current_CN_O}/{target_CN_O}", verb_th=1)

                nbrs = self.update_connected_neighbors(index_N)
                current_CN_N = self.get_current_CN_from_index(index_N)
                target_CN_N = self.get_target_CN_from_index(index_N)

                self.print(f"Site {type_N}{index_N} has been added at cartesian position {X_N} "
                           f" with current/target corrdination number {current_CN_N}/{target_CN_N}", 
                           verb_th=1)
                
            except ValueError as e:
                print(('{} neighbor could not be inserted at position {}: '
                       'ValueError: {}').format(new_nbr_name, X_N_try, e))
        else:
            if self.verbosity >= 2:
                print('Solution leads to contact between atoms.')

        if X_N is None:
            self.print(f"Exiting function add_neighbor with no {new_nbr_name} neighbor "
                       f"added to site {type_O}{index_O} with current/target CN "
                       f"{len(nbrs)}/{target_CN_O}", 
                       verb_th=2)
            self.structure.sites[index_O].properties['treatment_attempts'] += 1
            # struct is unchanged. No need to update nbrs.
            return nbrs  # TODO: consider returning None

        else:

            self.print(f"Exiting add_neighbor method. Site {type_O}{index_O} now has "
                       f"current/target coordination number {current_CN_O}/{target_CN_O}.", 
                       verb_th=2)

        self.print(f"Reaching the end of add_neighbor function. nbrs had length {len(nbrs)}", verb_th=4)
        nbrs = self.update_connected_neighbors(index_O)
        self.print(f"After update, nbrs now has length {len(nbrs)}", verb_th=4)

        self.print(f"Method add_neighbor applied to {new_nbr_name} neighbor of site "
                           f"{type_O}{index_O} will return nbrs of length {len(nbrs)}.", 
                           verb_th=4)
        
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
                    self.print(f"Atom {self.get_atom_type(nbr)}{nbr.index} within "
                               f"{contact_dist} \u212B of tried position {tried_position}.", 
                               verb_th=3)
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
            self.print(f"WARNING: Coordination number of site {site_type}{site_index} "
                       f"exceeds target : {len(nbrs)}/{target_CN}", verb_th=1)
        else:  # len(nbrs) > target_CN
            self.structure.sites[site_index].properties['is_shell_complete'
                                                        ] = False

        self.print('Site {} ({}) connected_neighbors property updated : {}'.format(
                   site_index, site_type, self.structure.sites[
                   site_index].properties['connected_neighbors']), verb_th=3)

        if include_neighbors:
            self.print(f"Method update_connected_neighbors applied to site {site_type}{site_index} "
                       f"will return nbrs of length {len(nbrs)}.", verb_th=4)
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
                              constraints=({'type': 'eq',
                                            'fun': lambda v: np.linalg.norm(v) - OA}), 
                              options={'disp': True if self.verbosity >= 3 else False})
            if result.success:
                X_A = result.x + X_O
                self.print(f"A cartesian position for a first neighbor to site {type_O}{index_O} "
                           f"that maximizes distances to nearby sites has been found at {X_A}", 
                           verb_th=2)
            else:
                self.print('WARNING: Minimization failed. Decide what to do from here.', 
                           verb_th=2)

        return X_A


    def find_second_neighbor_position(self, site_index, site_coord_number,
                                      nbrs, nbr_type=None):
        """
        Find target position for the second neighbor to a selected site in the structure
        
        Here O designates the central atom, A is the first neighbor, and B is the the 2nd neighbor to add.
        """
        
        # Designate site of focus by O for practicity
        index_O = site_index
        X_O = self.structure.sites[index_O].coords
        type_O = self.get_type_from_index(index_O)

        self.print(f"Running function find_second_neighbor_position for {type_O}{index_O} "
                   f"with nbrs: ", verb_th=4)
        [self.print(f"\t{nbr.species_string}{nbr.index} (image {nbr.image}) at {nbr.coords}", verb_th=4)
         for nbr in nbrs]
        
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

        self.print(f"X_A = {X_A}", verb_th=4)
        self.print(f"X_T = {X_T}", verb_th=4)
        self.print(f"OA_vect = {OA_vect}", verb_th=4) 
        self.print(f"n = np.cross(X_T - X_O, OA_vect) = {n}", verb_th=4)

        # Calculate a, b, c, d parameters in plane equation: ax + by + cz + d = 0
        (a, b, c) = n / np.linalg.norm(n)
        d = -(a*X_T[0] + b*X_T[1] + c*X_T[2])

        theta = (np.pi/180)*self.get_bond_angle_from_CN(site_coord_number)
        self.print(f"Using bond angle theta of {180*theta/np.pi:.2f} ° for coordination "
                   f"number {site_coord_number} at site {type_O}{index_O}.", verb_th=4)
        
        # B is along a vector v such that v.OA = |v|*|OA|*cos(theta)
        # (x_B, y_B, z_B) = sym.symbols('x_B y_B z_B')
        OB = float(self.get_bond_length(type_B, type_O))
        self.print(f"OB distance set to {type_B}-{type_O} bond length: {OB} \u212B.",
                   verb_th=4)

        def equations(X):
            x, y, z = X
            f1 = a*x + b*y + c*z + d  # B is in OAT plane
            f2 = (x-X_O[0])*OA_vect[0] + (y-X_O[1])*OA_vect[1] + \
                 (z-X_O[2])*OA_vect[2] - OB*OA*np.cos(theta)
            f3 = (x-X_O[0])**2 + (y-X_O[1])**2 + (z-X_O[2])**2 - OB**2
            return (f1, f2, f3)

        self.print(f"Looking for solutions for atom B of type {type_B} around "
                   f"site {type_O}{index_O}.", verb_th=3)
        initial_pos = X_O + OB * self.rng.random(3)
        solution, infodict, ier, msg = fsolve(equations, initial_pos, full_output=True)
        if ier != 1:
            self.print(f"find_second_neighbor_position fsolve for site {type_O}{index_O} did "
                       f"not find a solution to add a {type_B} neighbor: {msg}", 
                       verb_th=1)
            return None
        elif np.all(np.isclose(solution, initial_pos)):
            raise ValueError(f"find_second_neighbor_position fsolve for site {type_O}{index_O} did "
                             f"to add a {type_B} neighbor returned a solution identical to "
                             f"the initial_solution: {msg}")

        self.print(f"Solution found : {solution}.", verb_th=3)
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

        self.print(f"Looking for solution to place atom C of type {type_C} "
                   f"around site {type_O}{index_O}...", verb_th=3)
        solution = fsolve(equations, (0, 0, 0))
        self.print(f"Solution found: {solution}", verb_th=3)
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

        self.print(f"Trying to add 4th neighbor to site {type_O}{index_O} with target "
                   f"coordination number {site_coord_number}", verb_th=3)

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

        self.print(f"Looking for solution to place atom D of type {type_D} "
                   f"around site {type_O}{index_O}", verb_th=3)
        solution = fsolve(equations, (0, 0, 0))
        self.print(f"Solution found: {solution}.", verb_th=3)
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
        self.print(f"POSCAR file saved as {self.export_file_name}.")


    def update_all_connected_neighbors(self):
        """
        Check the coordination shell of all atoms in structure
        """
        for site_index, site in enumerate(self.structure.sites):
            self.update_connected_neighbors(site_index)


    def pick_missing_target_coord_numbers(self, skip_terminal=False):
        for site_index, site in enumerate(self.structure.sites):
            if site.properties['target_coord_number'] is None:
                site.properties['target_coord_number'] = \
                    self.pick_coord_number_from_type(self.get_atom_type(site),
                                                     skip_terminal=skip_terminal)



    def get_nb_of_connected_nbrs_with_non_zero_bonding_proba(self, site_index,
                                                             exclude_index=None,
                                                             indent=0):
        """
        Get ratio between the number of neighbors with non-zero clustering probability
        (based on central atom and neighbor types) and the target coordination number

        Args:
            exclude_index: int or None:
                Use to avoid double-counts when exploring a second shell
        """
        site = self.structure.sites[site_index]
        site_type = self.get_atom_type(site)
        site_target_CN = self.get_target_CN_from_index(site_index)
        n_connected_nbrs_with_non_zero_bonding_proba = 0
        for N_index in site.properties['connected_neighbors']:
            if N_index == exclude_index:
                continue
            N_type = self.get_atom_type(self.structure.sites[N_index])
            N_type_index = self.get_atom_type_index
            clustering_proba = self.get_clustering_proba_from_types(
                site_type, N_type)
            if clustering_proba > 0:
                n_connected_nbrs_with_non_zero_bonding_proba += 1

        applied_indent = indent * '\t'
        self.print(f"{applied_indent}{site_type}{site_index} with target CN {site_target_CN} "
                   f"has {n_connected_nbrs_with_non_zero_bonding_proba} neighbors with "
                   f"non-zero clustering probabilty (based on types)",
                   verb_th=4)

        return n_connected_nbrs_with_non_zero_bonding_proba


    def pick_atom_to_relocate(self):
        """
        Pick the next atom to relocate with meaningful probabilities

        Probabilities take into account that:

        * a site with a complete coordination sphere composed of atoms types
          to which it can be bonded (center-neighbor clustering_proba > 0)
          should be left untouched (p1 = 0) and sites with a large number of
          missing or incompatible-type neighbors should have high probability (high p1).

        * This evaluation (evaluation of missing or incompatible-type neighbors)
          is then also propagated to the next-neighbors, which should also be left untouched
          if complete.

        The final probability is p = p1 * p2

        """
        self.print('Function pick_atom_to_relocate().', verb_th=3)
        # Initialize propabilities

        self.print(f"\nCalculating picking probabilities of all {self.structure.num_sites} "
                   f"sites in the structure.", verb_th=2)
        p = np.ones(self.structure.num_sites)
        for site_index, site in enumerate (self.structure.sites):
            site_type = self.get_atom_type(site)
            target_CN = site.properties['target_coord_number']

            self.print(f"{site_type}{site_index} connected neighbors:", verb_th=3)

            # Set p to 0 if sphere is complete, highest for most-incomplete
            # spheres
            p[site_index] *= 1 - (self.get_nb_of_connected_nbrs_with_non_zero_bonding_proba(site_index,
                                                                                            indent=1)
                                  / target_CN)

            if np.isclose(p[site_index], 0):  # no need to explore first-neighbors-shell completion.
                self.print(f"\t{site_type}{site_index} has {target_CN} connected neighbor(s) "
                           f"with non-0 bonding probability for a taget CN of {target_CN}. "
                           f"Setting p[{site_index}] = 0.", verb_th=3)
                continue

            # Account for completion of nearest-neighbors coordination spheres
            # excluding the current site_index from the count
            cumul_target_CN = 0
            cumul_nb_of_nbrs = 0
            for connected_site_index in site.properties['connected_neighbors']:
                connected_site = self.structure.sites[connected_site_index]
                connected_site_type = self.get_atom_type(connected_site)
                self.print(f"\t{site_type}{site_index} - {connected_site_type}{connected_site_index}: "
                           f"connected_neighbors: {connected_site.properties['connected_neighbors']} "
                           f"({len(connected_site.properties['connected_neighbors'])}/"
                           f"{connected_site.properties['target_coord_number']})", verb_th=4)

                if site_index in connected_site.properties[
                        'connected_neighbors']:  # should always be a check (just a double-check)
                    cumul_target_CN += connected_site.properties[
                        'target_coord_number'] - 1
                    # count connected neighbors excluding site_index
                    cumul_nb_of_nbrs += self.get_nb_of_connected_nbrs_with_non_zero_bonding_proba(
                                            connected_site_index, exclude_index=site_index, indent=2)
                else:
                    self.print(('WARNING: in pick_atom_to_relocate: site_index'
                                ' {} not in connected_site.connected_neighbors'
                                ': {}').format(site_index,
                               connected_site.properties['connected_neighbors']
                               ))

            if cumul_target_CN > 0:
                p[site_index] *= 1 - (cumul_nb_of_nbrs / cumul_target_CN)
            else:
                self.print(f"\tCumulated target CN of {site_type}{site_index} neighbors is 0. "
                           f"p[{site_index}] picking probability kept to {p[site_index]:.3f}",
                           verb_th=4)

            # other priority checks ? prioritize bigger atoms ?

        # Normalize probabilities:
        p_sum = np.sum(p)
        if p_sum > 0:
             p /= p_sum
        else:
             print('WARNING: picking probabilities in pick_atom_to_relocate '
                   'are all 0. Find out why.')
             return None
        picked_atom_index = self.rng.choice(np.arange(len(p)), p=p)
        picked_atom_type = self.get_type_from_index(picked_atom_index)
        picked_atom_type = self.get_type_from_index(picked_atom_index)
        picked_atom_current_CN = self.get_current_CN_from_index(picked_atom_index)
        picked_atom_target_CN = self.get_target_CN_from_index(picked_atom_index)

        self.print(f"Picking site_to_relocate {picked_atom_type}{picked_atom_index} "
                   f"with current/target CN {picked_atom_current_CN}/{picked_atom_target_CN}.",
                   verb_th=1)

        self.print("from the following probabilities", verb_th=3)
        if self.verbosity >= 3:
            for index, proba in enumerate(p):
                self.print('  {}: {}, p = {:.3f}'.format(index,
                           self.get_type_from_index(index), proba), verb_th=3)

        return picked_atom_index

    def get_current_CN_from_index(self, site_index):
        return len(self.structure.sites[site_index].properties["connected_neighbors"])


    def pick_site_to_complete(self, site_to_relocate_index=None,
                              new_site_type=None, new_site_CN=None):
        """
        Pick the site that will temptatively be completed with the atom to relocate or add

        Remove site_to_relocate
        eliminate sites with complete shells
        calculate probabilities according to priorities:
            1. p = 0 if shell is aleady complete
            1. favor nearly-complete shell
               p = CN/target_CN
            2. favor type according to target proba
               p = p * clustering_proba (average A-B and B-A ?)
            3. take current global clustering proba into account :
               p = 0 if current_global_clustering_proba(A-B) >
               clustering_proba(A-B)
               p = p *(target-current)/target clustering_proba(AB) otherwise
            4. favor atom size :
               p = p * sum(bond_lengths)/max(sum(bond_lengths))

        Chose between site_to_relocate_index (if )
        """

        """
        self.print(('Function pick_site_to_complete(site_to_relocate_index={}'
                    .format(site_to_relocate_index)), verb_th=2)
        """
        if site_to_relocate_index is not None:
            new_site_index = site_to_relocate_index
            new_site_type = self.get_type_from_index(new_site_index)
        elif new_site_type is None or new_site_CN is None:
            raise ValueError("Set either site_to_relocate_index or (new_site_type, new_site_CN)")
        else:
            new_site_index = None  # Not yet attributed

        # Calculate picking probability as a function of index
        p = np.ones(self.structure.num_sites)
        for site_index, site in enumerate (self.structure.sites):
            if (site_index == new_site_index) or \
                    site.properties['is_shell_complete']:
                p[site_index] = 0.0
            else:
                # favor sites whose shell is closest to completion
                p[site_index] *= (len(site.properties['connected_neighbors']) /
                                 site.properties['target_coord_number'])
                site_type = self.get_atom_type(site)
                clustering_proba = self.get_clustering_proba_from_types(
                    site_type, new_site_type)
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
                        ).format(new_site_index, new_site_type))
            # Check whether all sites have been completed.
            return None
        picked_atom_index = self.rng.choice(np.arange(len(p)), p=p)

        picked_atom_type = self.get_type_from_index(picked_atom_index)
        picked_atom_current_CN = self.get_current_CN_from_index(picked_atom_index)
        picked_atom_target_CN = self.get_target_CN_from_index(picked_atom_index)

        self.print(f"Picking site_to_complete {picked_atom_type}{picked_atom_index} "
                   f"with current/target CN {picked_atom_current_CN}/{picked_atom_target_CN}.",
                   verb_th=2)

        if self.verbosity >= 3:
            print('using the following probabilities:')
            for index, proba in enumerate(p):
                print(f"  {self.get_type_from_index(index)}: {index}, p = {proba:.3f}")

        return picked_atom_index


    def relocate_atom(self, site_index):
        """
        TOBECOMPLETED
        """
        # Pick site-to-complete to relocate site_index onto
        # Designate site_to_complete as A and site_to_relocate as X for convenience
        X_index = site_index
        X_type = self.get_type_from_index(X_index)
        X_target_CN = self.get_target_CN_from_index(X_index)

        A_index = self.pick_site_to_complete(X_index)
        if A_index is None:
            self.print(f"{X_type}{X_index} could not be relocated because picking "
                       f"picking probabilities of all sites-to-complete are 0.",
                       verb_th=1)
            return 1, 'Picking probabilities of all sites-to-complete are 0.'
        A_type = self.get_type_from_index(A_index)
        A = self.structure.sites[A_index].coords
        A_target_CN = self.structure.sites[A_index].properties[
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
            # Use a neighbor search rather than indexes to account for
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

                    if X_target_CN == 1: # No angle to consider in this case
                        AXN_0 = None
                    else:
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

                    L_Nbrs = self.structure.get_neighbors(self.structure.sites[N_index],
                                                          self.get_max_bond_length_for_type(N_type,
                                                          include_tol=True))

                    for L_nbr in L_Nbrs:
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

                            if N_target_CN == 1:  # No angle to consider in this case
                                XNL_0 = None
                            else:
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

        self.print(f"\nStarting SLSQP minimization to relocate {X_type}{X_index} " 
                   f" in the neighborhood of {A_type}{A_index} with:\n"
                   f"  - initial coordinates: {X_0}\n"
                   f"  - constraints: {constraints}\n"
                   f"  - bounds: {bounds}\n\n",
                   verb_th=3)

        # MINIMIZE
        result = minimize(fun, X_0, constraints=constraints, bounds=bounds,
                          method='SLSQP', 
                          options={'disp': True if self.verbosity >= 3 else False})
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

            self.print(f"Site X {X_type}{X_index}) after relocation: {self.structure.sites[X_index]}, "
                       f"{self.structure.sites[X_index].properties}", verb_th=2)

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

            X_current_CN = self.get_current_CN_from_index(X_index)
            A_currrent_CN = self.get_current_CN_from_index(A_index)

            self.print(f"Site X {X_type}{X_index} (current/target CN: {X_current_CN}/{X_target_CN}) "
                       f"was successfully relocated in the neighborhood of site {A_type}{A_index} "
                       f" (current/target CN: {A_currrent_CN}/{A_target_CN})), at cartesian "
                       f"position {X}.", verb_th=1)

            return 0, f"Atom {X_type}{X_index} relocation was successful."  # meaning sucess

        else:
            self.print(f"No optimum found the relocation of site {X_type}{X_index} as "
                       f"a neighbor of site {A_type}{A_index}.", verb_th=1)
            return 2, 'Minimization failure: {}'.format(result)  # Meaning minimization error

    def get_site_indexes_by_type(self, types=None, to_single=False):
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
        site_indexes_by_type = [list(self.structure.indices_from_symbol(t)) for t in
                         types]
        if to_single and len(site_indexes_by_type) == 1:
            [site_indexes_by_type] = site_indexes_by_type
        return site_indexes_by_type

    def get_target_CN_proba(self, atom_type):
        return self.species_properties[atom_type]["coord_proba"]


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
        site_indexes_by_type = self.get_site_indexes_by_type(types=types)
        for type_index, atom_type in enumerate(types):
            coord_proba = []
            for CN, proba0 in enumerate(self.species_properties[atom_type][
                'coord_proba']):
                # count number of atom with CN
                coord_proba.append(len(
                    [i for i in site_indexes_by_type[type_index] if
                     len(self.structure.sites[i].properties[
                         'connected_neighbors']) == CN])
                                   / len(site_indexes_by_type[type_index]))
            coord_proba_by_type.append(coord_proba)

        # Convert list of list to simple list if a single type is requested
        if len(types) == 1 and to_single:
            [coord_proba_by_type] = coord_proba_by_type

        return coord_proba_by_type

    def is_potentially_terminal_type(self, atom_type):
        """ True if the first (CN=1) coordination probability of the considered type is > 0 """
        return True if self.get_target_CN_proba(atom_type)[1] > 0 else False

    def is_strictly_terminal_type(self, atom_type):
        """
        True if the first (CN = 1) coordination probability of the considered type is 1
        and all others (CN > 1) 0.
        """
        target_CN_proba = self.get_target_CN_proba(atom_type)
        target_CN_proba_other_than_1 = [tcnp for i, tcnp in enumerate(target_CN_proba)
                                        if i != 1]
        if (np.isclose(target_CN_proba[1], 1)
            and np.all([np.isclose(v, 0) for v in target_CN_proba_other_than_1])):
            return True
        else:
            return False

    def get_potentially_terminal_types(self):
        potentially_terminal_types =  [t for t in self.system["atom_types"]
                                       if self.is_potentially_terminal_type(t)]
        self.print(f"Potentially terminal types in the considered system: {potentially_terminal_types}",
                   verb_th=2)
        return potentially_terminal_types


    def get_strictly_terminal_types(self):
        strictly_terminal_types =  [t for t in self.system["atom_types"]
                                       if self.is_strictly_terminal_type(t)]
        self.print(f"Stricktly terminal types in the considered system: {strictly_terminal_types}",
                   verb_th=2)
        return strictly_terminal_types

    def get_target_CN_from_index(self, site_index):
        return self.structure.sites[site_index].properties["target_coord_number"]


    def get_statistics(self):
        """
        Get a dict of statistics for the considered system
        """

        nb_of_complete_sites = self.get_nb_of_complete_sites()
        site_indexes_by_type = self.get_site_indexes_by_type()
        nb_of_sites_by_type = [len(site_indexes_by_type[i]) for i,s in
                               enumerate(site_indexes_by_type)]
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


def main(seed=None, input_file='input.json', add_terminal_atoms_last=True,
         verbosity=1, abs_bond_length_tol=0.0,
         rel_bond_length_tol=0.1, max_attempts=20,
         max_iterations_step1=1000, max_iterations_step2=500,
         max_iterations_step3=500,
         numeric_tolerance=1e-5, search_radius=3.0,
         first_atom_intern_coords=(0.5, 0.5, 0.5), visualizer='vesta',
         export_format='poscar'):

    cnbd = ceramicNetworkBuilderData(seed=seed, input_file=input_file,
                                     add_terminal_atoms_last=add_terminal_atoms_last,
                                     verbosity=verbosity,
                                     abs_bond_length_tol=abs_bond_length_tol,
                                     rel_bond_length_tol=rel_bond_length_tol,
                                     numeric_tolerance=numeric_tolerance,
                                     search_radius=search_radius,
                                     visualizer=visualizer,
                                     export_format=export_format,
                                     max_iterations_step1=max_iterations_step1,
                                     max_iterations_step2=max_iterations_step2,
                                     max_iterations_step3=max_iterations_step3,
                                     max_attempts=max_attempts)

    cnbd.print_versions()
    cnbd.save_sample_json_input_file()

    nb_of_atoms = sum(cnbd.system['nb_of_atoms_by_type'])
    remaining_atoms_by_type = cnbd.system['nb_of_atoms_by_type']
    nb_of_atoms_by_type = np.asarray(cnbd.system['nb_of_atoms_by_type'])

    cnbd.initialize_structure(first_atom_intern_coords)

    cnbd.print(
        "************************************************************************\n"
        "*                ceramicNeworkBuilder Step 1                           *\n"
        "************************************************************************\n"
        " Place atoms with strict constraints on \n"
        " - bond lengths : equal to expected value based on atom types \n"
        " - bond angles : equal to expected values based on coordination number \n"
        " - min. dist between non-bonded atoms: > bond_length + (abs/rel tol) \n"
        "************************************************************************", 
        verb_th=0
    )

    iteration_index = -1
    while cnbd.structure.num_sites < np.sum(nb_of_atoms_by_type):
        iteration_index += 1

        if iteration_index >= cnbd.max_iterations_step1:
            print(f'Maximum number of Step-1 iterations ({cnbd.max_iterations_step1}) reached.')
            break

        # TODO: add a stop based on number of remaining non-terminal atoms

        cnbd.print(f"\nStarting Step-1 iteration number {iteration_index + 1}/{cnbd.max_iterations_step1}.", 
                   verb_th=1)

        # pick first atom not-yet-treated
        # TODO: (?) if add_terminal_atoms_last is True, exclude strictly-terminal sites from sites_not_yet_treated ?

        if add_terminal_atoms_last:
            # Strictly-terminal types should be avoided.
            strictly_terminal_types = cnbd.get_strictly_terminal_types()

            # Check whether structure already contains strictly-terminal types, which
            # should not be the case in phase 1.
            site_indexes = cnbd.get_site_indexes_by_type(types=strictly_terminal_types, to_single=True)
            if len(site_indexes):
                cnbd.print(f"Site indexes of striclty-terminal type {strictly_terminal_types}: {site_indexes}")
                cnbd.print(f"Structure at iteration index {iteration_index}:\n{cnbd.structure}")
                raise ValueError(f"Structure contains strictly-terminal atoms of types "
                                 f"{strictly_terminal_types} during step 1. This should not be the case.")

        sites_not_yet_treated = [index for (index, site) in enumerate(cnbd.structure.sites)
                                 if (not site.properties['is_treated']) and
                                 (site.properties['treatment_attempts'] < cnbd.max_attempts)]

        cnbd.print(f"sites_not_yet_treated = {sites_not_yet_treated}.", verb_th=2)

        if len(sites_not_yet_treated):
            # Treat sites in order of creation until one type of atoms is exhausted
            remaining_atoms = cnbd.get_remaining_atoms_by_type()
            if min(remaining_atoms) > 0:
                current_site_index = sites_not_yet_treated[0]
            else:  # end then randomly
                [current_site_index] = cnbd.rng.choice(sites_not_yet_treated, size=1)
        else:
            print('All sites have been treated. A new random site must be added.')

            # Add a new atom from scratch at a random position
            site = cnbd.add_random_site(skip_terminal=cnbd.add_terminal_atoms_last)

            if site is not None:
                continue
            else:
                cnbd.print("Could not add a new random site. Exiting step 1.", verb_th=1)
                break


        current_site_type = cnbd.get_type_from_index(current_site_index)
        # Set coord based on probabilities (easiest: ignore other sites)i
        current_site_target_CN = cnbd.pick_coord_number_from_type(current_site_type,
                                                           skip_terminal=cnbd.add_terminal_atoms_last)
        cnbd.structure.sites[current_site_index].properties['target_coord_number'] = \
            current_site_target_CN
        
        cnbd.print(f"Current site [{current_site_index}] : {current_site_type} with target "
                   f"coordination number {current_site_target_CN}.", verb_th=1)

        cnbd.build_current_site_shell(current_site_index,
                                      skip_terminal=cnbd.add_terminal_atoms_last)

        cnbd.print(f"\nStructure at the end of iteration {iteration_index}:\n{cnbd.structure}\n",
                   verb_th=3)

    cnbd.update_all_connected_neighbors()
    cnbd.pick_missing_target_coord_numbers(skip_terminal=cnbd.add_terminal_atoms_last)

    # FOR TESTING PURPOSES: RESET VERBOSITY TO USER-VALUE
    cnbd.verbosity = verbosity

    cnbd.print(f"\nStructure at the end of Step 1:\n{cnbd.structure}\n", verb_th=1)

    cnbd.print(
        "\n"
        "************************************************************************\n"
        "*                   ceramicNeworkBuilder Step 2                        *\n"
        "************************************************************************\n"
        "Step 2: force atoms at positions matching \n"
        " - bond lengths : equal to expected value based on atom types \n"
        " - bond angles : equal to expected values based on coordination number\n"
        " - min. dist between non-bonded atoms: > bond_length + (abs/rel tol) \n"
        "************************************************************************", 
        verb_th=0
    )

    iteration_index = -1
    while 1:
        iteration_index += 1
        if iteration_index >= cnbd.max_iterations_step2:
            cnbd.print(('Maximum number of iterations for step 2: site '
                        'relocations ({}) has been reached.').format(
                       cnbd.max_iterations_step2))
            break

        cnbd.print(f"\nStarting Step-2 iteration number {iteration_index + 1}/"
                   f"{cnbd.max_iterations_step2}.", verb_th=1)

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

        current_site_type = cnbd.get_type_from_index(current_site_index)
        current_site_target_CN = cnbd.get_target_CN_from_index(current_site_index)
        current_site_CN = len(cnbd.structure.sites[current_site_index].properties["connected_neighbors"])

        if not exit_code :
            cnbd.print(f"\n{20*'*'}\nStructure after relocation-iteration # {iteration_index} "
                       f"({current_site_type}{current_site_index} with new/target CN "
                       f"{current_site_CN}/{current_site_target_CN}:\n{20*'*'}'\n{cnbd.structure}",
                       verb_th=3)
        else:
            cnbd.print(('relocate_atom function failed with exit_code {}: {}'
                        ).format(exit_code, exit_msg), verb_th=2)
            cnbd.print(f"Atom {current_site_type}{current_site_index} with current/target "
                       f"coordination number {current_site_CN}/{current_site_target_CN} "
                       f"could not be relocated. Trying to relocate another atom.", 
                       verb_th=1)

    # Rank candidate atoms-to-move by :
    #   (1) nb_of_missing_nbrs (descending order)
    #           = site.properties['target_coord_number'] - len(site.properties['connected_neighbors'])
    #       At first, selected sites will have 1 neighbor, and among those we will chose
    #   (2) nb_of_missing_nbrs within site.properties['connected_neighbors'] (exluding site-of-focus)
    # The idea is to deplete already-depleted sites.

    if cnbd.add_terminal_atoms_last:
        cnbd.print(f"\n{100 * '*'}\nStructure at the end of step 2\n{100 * '*'}\n"
                   f"{cnbd.structure}\n{100 * '*'}\n", verb_th=1)

        cnbd.print(
            "\n"
            "************************************************************************\n"
            "*                ceramicNeworkBuilder Step 3                           *\n"
            "************************************************************************\n"
            " Adding terminal atoms. \n"
            "************************************************************************\n"
        )

        iteration_index = 0
        while 1:
            iteration_index += 1

            # DEBUGGING:
            iteration_index += 1
            
            remaining_potentially_terminal_atoms_by_type = {
                t: n for t, n in cnbd.get_remaining_atoms_by_type(as_dict=True).items()
                if cnbd.is_potentially_terminal_type(t)
            }
            remaining_potentially_terminal_types = list(remaining_potentially_terminal_atoms_by_type.keys())
            n_remaining_potentially_terminal_atoms = np.sum(list(
                remaining_potentially_terminal_atoms_by_type.values()))

            if not n_remaining_potentially_terminal_atoms:
                cnbd.print(f"There is no more potentially-terminal site to add. Exiting loop.")
                break

            if iteration_index >= max_iterations_step3:
                cnbd.print(f"The maximum number of iterations ({max_iterations_step3}) in step 3 "
                            f"has been reached. Exiting loop.")
                break

            cnbd.print(f"\nStarting Step-3 iteration number {iteration_index + 1}/"
                       f"{cnbd.max_iterations_step3}.", verb_th=1)

            # Pick type to add:
            p = np.ones(len(remaining_potentially_terminal_types))
            for type_index, atom_type in enumerate(remaining_potentially_terminal_types):
                # Account for relative amounts
                p[type_index] *= remaining_potentially_terminal_atoms_by_type[atom_type] / n_remaining_potentially_terminal_atoms
                # Account for probability to be terminal
                target_CN_proba = cnbd.get_target_CN_proba(atom_type)
                p[type_index] *= target_CN_proba[1] / np.sum(target_CN_proba)

            new_site_type = cnbd.rng.choice(remaining_potentially_terminal_types, p=p)

            site_to_complete_index = cnbd.pick_site_to_complete(new_site_type=new_site_type, new_site_CN=1)

            if site_to_complete_index is None:
                continue

            site_to_complete_CN = cnbd.get_target_CN_from_index(site_to_complete_index)

            nbrs = cnbd.update_connected_neighbors(site_to_complete_index)

            # TODO: (?) use several attempts for each site_to_complete
            new_nbrs = cnbd.add_neighbor(site_to_complete_index, site_to_complete_CN, nbrs,
                                         skip_terminal=False)

            if len(new_nbrs) > len(nbrs):
                cnbd.update_all_connected_neighbors()

        print("Step 3 is done.")

    if cnbd.verbosity >= 1:
        cnbd.print(f"\n{100 * '*'}\nFinal structure\n{100 * '*'}\n"
                   f"{cnbd.structure}\n{100 * '*'}\n")
        print('Density = {:.3f} g/cm-3'.format(cnbd.structure.density))

    stat_dict = cnbd.get_statistics()
    stat_file = "statistics.json"
    with open("statistics.json", "w") as f:
        json.dump(stat_dict, f, indent=4)
    print(f"Statistics have been stored as file {os.path.abspath(stat_file)}")

    fig = cnbd.plot_statistics(stat_dict=stat_dict)

    # print('cnbd = ', cnbd)
    if cnbd.export_format.lower() == 'poscar':
        cnbd.export_vasp_poscar(dir_name=os.getcwd())

    cnbd.visualize()

    plt.show()


def get_cell_length_from_target_compo_and_density(compo: Composition, density: FloatWithUnit, default_density_unit="g cm^-3", length_unit="ang"):
    """
    Get cubic cell length from a pymatgen Composition and a target density
    """
    if isinstance(compo, str):
        # Get composition from formula
        compo = Composition(compo)

    if not isinstance(density, FloatWithUnit):
        density = FloatWithUnit(density, default_density_unit)

    volume = compo.weight / density
    cell_length = volume ** (1/3)

    return cell_length.to("ang")




# Set label of the group containing the structures on which analyses should be
# performed.
# TODO: insert a density option to set density in g/cm-3 to a specifed value
# by adjusting the cell length parameters.

@click.command('cli')
@click.option('-s', '--seed', default=None, type=int,
              help='Seed for random number generator.')
@click.option('-i', '--input_file', default='input.json', type=str,
              help='Input file name (default: \'input.json\'.')
@click.option('-l', '--add_terminal_atoms_last', is_flag=True,
              help='Whether terminal atoms should be added in the end (additional step 3).')
@click.option('-r', '--rel_bond_length_tol', default=0.1, type=float,
              help=('Relative contact tolerance in fraction of expected bond '
                    'length (default: 0.1).'))
@click.option('-m', '--max_iterations_step1', default=1000, type=int,
              help='Maximum number of iterations for step 1 (default: 1000).')
@click.option('-M', '--max_iterations_step2', default=500, type=int,
              help='Maximum number of iterations for step 2 (default: 500).')
@click.option('-N', '--max_iterations_step3', default=500, type=int,
              help=('Maximum number of iterations for step 3 (only used with ' \
                    'add_terminal_atoms_last, default: 500).'))
@click.option('-a', '--max_attempts', default=10, type=int,
              help='Maximum number of completion attempts per site (default: 10).')
@click.option('-V', '--visualizer', default='ase', type=str,
              help='Visualizer (vesta, ase)')
@click.option('-e', '--export_format', default='poscar', type=str,
              help='Export format: poscar')
@click.option('-v', '--verbosity', default=1, type=int,
              help='Verbosity level. Default is 1')
def cli(seed, input_file, add_terminal_atoms_last, rel_bond_length_tol,
        max_iterations_step1, max_iterations_step2, max_iterations_step3,
        max_attempts, visualizer, export_format, verbosity):
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
         add_terminal_atoms_last=add_terminal_atoms_last,
         rel_bond_length_tol=rel_bond_length_tol,
         max_iterations_step1=max_iterations_step1,
         max_iterations_step2=max_iterations_step2,
         max_iterations_step3=max_iterations_step3,
         max_attempts=max_attempts,
         visualizer=visualizer.lower(),
         export_format=export_format, verbosity=verbosity)


if __name__ == '__main__':
    cli()
