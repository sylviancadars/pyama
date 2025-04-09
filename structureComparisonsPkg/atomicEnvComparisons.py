# atomcEnvComparisons module
"""
Use ML kernels (SOAP) to compare atomic environments

Sylvian Cadars, Assil Bouzid and Firas Shuaib Mohammed,
IRCER, University of Limoges, CNRS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel  # can be used to calculate this similarity
# from dscribe.kernels import REMatchKernel
# from sklearn.preprocessing import normalize

from pyama.utils import get_ase_atoms
from pyama.utils import BaseDataClass

from ase.atoms import Atoms

from itertools import product
from copy import deepcopy
from time import perf_counter
from sys import getsizeof
import pandas as pd
import os
import json

plt.rcParams['svg.fonttype'] = 'none'

# Create a custom color map (here Red-Gold-Green)
colors = [(1, 0, 0), (1, 0.9, 0), (0, 0.50, 0)]  # Red, Gold, Green
cmap = LinearSegmentedColormap.from_list('RedGoldGreen', colors)

class AtomEnvComparisonsData(BaseDataClass):
    """
    A class to compare individual atomic environments based on SOAP (or other) kernel

    TO BE COMPLETED
    """
    def __init__(self, average_kernel_metric='polynomial', average_kernel_degree=4,
                 soap_parameters=None, species=None, verbosity=1):

        super().__init__(verbosity=verbosity,
                         print_to_console=True,
                         print_to_file=False)

        self.average_kernel_metric = average_kernel_metric
        self.average_kernel_degree = average_kernel_degree

        if not soap_parameters:
            self.soap_parameters = {
                'r_cut': None,
                'n_max': 4,
                'l_max': 4,
                'sigma': 0.5,  # WARNING: small values such as 0.1 give unreliable results (default is 1.0)
                'rbf': 'polynomial',
                'weighting': {
                    "function": "poly",
                    "r0": 8.0,
                    "c": 1,
                    "m": 1
                },
                'periodic': True,
                'compression': {'mode': 'crossover'},
                'sparse': False
            }
        else:
            self.soap_parameters = soap_parameters

        self.verbosity = verbosity

        if species:
            self.set_species(species)
            self.set_soap_descriptor(species=species)

        else:
            self.species = None
            self.soap_descriptor = None
        # TODO: add soap.weighting

    def set_species(self, atoms_or_atoms_list):
        """
        Set species (as a list) from single or list of ASE Atoms or other

        In principle the function should work for a list of pymatgen structures
        as well.

        Args:
            atoms_or_atoms_list: str, Atoms, list, tuple, set
                single or list of element names, ASE Atoms
        """
        # Make list if single element
        if not isinstance(atoms_or_atoms_list, (list, tuple, set)):
            atoms_or_atoms_list = [atoms_or_atoms_list]

        if isinstance(atoms_or_atoms_list[0], str):
            self.species = list(atoms_or_atoms_list)
        else:
            _species = set()
            for _ in atoms_or_atoms_list:
                atoms = get_ase_atoms(_) if not isinstance(_, Atoms) else _
                _species.update(set(atoms.get_chemical_symbols()))
            self.species = list(_species)

        self.print('species property set to {}'.format(self.species), verb_th=2)


    def set_soap_descriptor(self, species=None):
        """
        Create a SOAP descriptor

        TO BE COMPLETED

        Args:

        Returns:

        """
        if species is not None:
            self.set_species(species)

        if self.species is None:
            raise ValueError('species property should be provided to create the SOAP descriptor.')

        self.soap_descriptor = SOAP(species=self.species, **self.soap_parameters)

    def get_similary_maps_by_type(self, system_1, system_2, species=None,
                                  periodic_1=None, periodic_2=None,
                                  matching_sites=False):
        """
        Get a dictionary containing similary maps by types of elements between two systems

        Args:
            system_1: ASE Atoms, pymatgen (I)Structure or (I)Molecule, or file name
                First atomic system
            system_2: ASE Atoms, pymatgen (I)Structure or (I)Molecule, or file name
                Second atom system
            periodic_1: bool or None (default is None)
                Whether periodicity should be considered to build the SOAP descriptors
                for system_1. If None the periodicity of the Atoms object (Atoms.pbc)
                will be used.
            periodic_2: bool or None (default is None)
                Whether periodicity should be considered to build the SOAP descriptors
                for system_2. If None the periodicity of the Atoms object (Atoms.pbc)
                will be used.
            matching_sites: bool (default is False)
                if True similarities will only be calculated for matching sites.
                This works only if sites between structures are identical.

        Returns:
            A dictionary constructed as follows:
                similarities = {
                    atom_type_1: {
                        'atoms_1_indexes': array of shape (1, n)
                        'atoms_2_indexes': array of shape (1, m)
                        'map': similarity map of shape (n, m),
                    },
                    atom_type_2: {
                        ...
                }
            where n and m are the number of atoms of atom_type_1 in system_1 and system_2
        """
        atoms_1 = get_ase_atoms(system_1)
        atoms_2 = get_ase_atoms(system_2)
        if matching_sites and (len(atoms_1) != len(atoms_2)
                               or atoms_1.get_chemical_symbols() !=
                               atoms_2.get_chemical_symbols()):
            raise ValueError('system_1 and system_2 should be structures with matching '
                             'number of sites and indexing.')
        if species is not None:
            self.set_species(species)
        else:
            if self.species is None:
                self.set_species([atoms_1, atoms_2])
                self.print('species was set from atoms_1 and atoms_2 arguments'.format(
                    self.species), verb_th=3)
            else:  # update self.species with species in atoms_1 and/or atoms_2
                _species = set(self.species)
                for atoms in [atoms_1, atoms_2]:
                    _species.update(atoms.get_chemical_symbols())
                    self.print('_species updated to {}'.format(_species), verb_th=3)
                self.set_species(list(_species))

        if periodic_1 is None:
            periodic_1 = all(atoms_1.pbc)
        if periodic_2 is None:
            periodic_2 = all(atoms_2.pbc)

        features_list = []
        for periodic, atoms in zip([periodic_1, periodic_2], [atoms_1, atoms_2]):
            # Allow possibility to have non-periodic atoms_1 or atoms_2
            if ('periodic' in self.soap_parameters.keys()
                and self.soap_parameters['periodic'] == periodic):
                if self.soap_descriptor is None:
                    self.set_soap_descriptor()
                desc = self.soap_descriptor
            else:
                desc = SOAP(species=self.species, **self.soap_parameters)

            features_list.append(desc.create(atoms))

        [features_1, features_2] = features_list

        sim = AverageKernel(metric=self.average_kernel_metric,
                            degree=self.average_kernel_degree)
        similarities = {}
        for specie_index, specie in enumerate(self.species):
            similarities[specie] = {}
            indexes_and_atoms_1 = [(i, at) for i, at in enumerate(atoms_1) if at.symbol == specie]
            # TODO: skip equivalent sites (as identified by kernels)
            indexes_and_atoms_2 = [(i, at) for i, at in enumerate(atoms_2) if at.symbol == specie]
            similarities[specie]['map'] = np.zeros((len(indexes_and_atoms_1),
                                             len(indexes_and_atoms_2)))
            self.print('similarities[{}][\'map\'] is of shape {}'.format(
                specie, similarities[specie]['map'].shape), verb_th=2)
            similarities[specie]['atoms_1_indexes'] = [_[0] for _ in indexes_and_atoms_1]
            similarities[specie]['atoms_2_indexes'] = [_[0] for _ in indexes_and_atoms_2]
            for j, (index_1, atom_1) in enumerate(indexes_and_atoms_1):
                for i, (index_2, atom_2) in enumerate(indexes_and_atoms_2):
                    if index_1 == index_2 or not matching_sites:
                        sim_kernel = sim.create([features_1[index_1].reshape(1, -1),
                                                 features_2[index_2].reshape(1, -1)])
                        self.print((f'({j},{i}), similarity kernel between {specie}-{index_1} '
                                    f'and {specie}-{index_2} : {sim_kernel}'), verb_th=3)
                        similarities[specie]['map'][j, i] = sim_kernel[0][1]
                        self.print('similarities[{}][\'map\'][{}, {}] = {}'.format(
                            specie, j, i, similarities[specie]['map'][j, i]), verb_th=3)
        return similarities

    def plot_similarity_maps(self, similarities, title=None, system_1_name=None,
                             system_2_name=None, fig_size_inches=(9.6, 4.8),
                             cmap=None, subplots=None, show_plot=True):
        """
        Plot similarity maps as obtained with function get_similary_maps_by_type

        Args:
            cmap: cmap, str or None.
                Set to 'default' to use the default matplotlib map.
            subplots: tuple, list or None (default is None)
                if None (1, nb_of_species) subplots will be used
        """
        # TODO: adapt number of subplots to number of species
        if cmap == 'default':
            cmap = None
        elif cmap is None:
            colors = [(1, 0, 0), (1, 0.9, 0), (0, 0.75, 0)]  # Red, Gold, Green
            cmap = LinearSegmentedColormap.from_list('RedGoldGreen', colors)

        if not isinstance(system_1_name, str):
            system_1_name = 'system 1'
        if not isinstance(system_2_name, str):
            system_1_name = 'system 2'

        if subplots is None:
            fig, ax = plt.subplots(1, len(similarities.keys()))
        else:
            fig, ax = plt.subplots(subplots[0], subplots[1])

        # flatten ax to facilitate loop.
        ax_flat = ax.ravel()
        for specie_index, specie in enumerate(similarities.keys()):
            self.print('similarities[{}][map] has shape {}'.format(
                specie, similarities[specie]['map'].shape), verb_th=2)
            pcm = ax_flat[specie_index].imshow(similarities[specie]['map'], cmap=cmap)
            ax_flat[specie_index].set(title='simaliries between {} atoms'.format(specie),
                                 xlabel='index in {}'.format(system_2_name),
                                 ylabel='index in {}'.format(system_1_name))
            ax_flat[specie_index].set_xticks(np.arange(len(similarities[specie]['atoms_2_indexes'])),
                                        similarities[specie]['atoms_2_indexes'])
            ax_flat[specie_index].set_yticks(np.arange(len(similarities[specie]['atoms_1_indexes'])),
                                        similarities[specie]['atoms_1_indexes'])

            cb = fig.colorbar(pcm, ax=ax_flat[specie_index])

        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
        if show_plot:
            plt.show()
        return fig, ax

    @staticmethod
    def get_fractional_coordinates(atoms):
        return atoms.cell.scaled_positions(atoms.positions)

    def sort_atoms_by_fractional_coordinate(self, atoms, axis=2):
        """
        Sort atoms according to coordinates along a chosen crystallographic axis

        Args:
            atoms: ASE Atoms
                ASE system to sort
            axis: int (default: 2)
                axis along which sites should be sorted.
        Returns:
            Sorted copy of the considered ASE Atoms

        """
        frac_coords = self.get_fractional_coordinates(atoms)
        return atoms[frac_coords[:, axis].argsort(axis=0)]

    def plot_similarity_maps_to_multi_ref_systems(self, atomic_system,
            ref_systems, atomic_system_name=None, ref_systems_names=None,
            sort_by_frac_coord_along_axis=None,
            show_plot=True):
        """
        TO BE COMPLETED
        """
        atoms, _name = get_ase_atoms(atomic_system, return_description=True)
        if sort_by_frac_coord_along_axis is not None:
            if sort_by_frac_coord_along_axis in ['a', 'b', 'c']:
                axis=['a', 'b', 'c'].index(sort_by_frac_coord_along_axis)
            elif sort_by_frac_coord_along_axis in [0, 1, 2]:
                axis=sort_by_frac_coord_along_axis
            else:
                raise ValueError(('sort_by_frac_coord_along_axis should be '
                                  'None, 0, 1, 2, \'a\', \'b\' or \'c\'.'))
            atoms = self.sort_atoms_by_fractional_coordinate(atoms, axis=axis)
        if atomic_system_name is None:
            atomic_system_name = _name

        ref_systems_atoms = []
        if ref_systems_names is None:
            ref_systems_names = []
        for ref_system in ref_systems:
            _atoms, _name = get_ase_atoms(ref_system, return_description=True)
            ref_systems_atoms.append(_atoms)
            if len(ref_systems_names) < len(ref_systems_atoms):
                ref_systems_names.append(_name)

        species = set()
        similarities = {}
        for ref_syst_name, ref_syst_atoms in zip(ref_systems_names,
                                                 ref_systems_atoms):
            species.update(ref_syst_atoms.get_chemical_symbols())
            similarities[ref_syst_name] = self.get_similary_maps_by_type(
                ref_syst_atoms, atoms)

        self.print('species = {}'.format(species), verb_th=3)
        self.print('similarities = {}'.format(similarities), verb_th=3)

        # Find min and max silimarity values
        sim_min = 1.0
        sim_max= 0.0
        for ref_syst_name in similarities.keys():
            for specie in similarities[ref_syst_name].keys():
                self.print('Shape of similarities[\'{}\'][\'{}\'][\'map\'] = {}'.format(
                    ref_syst_name, specie,
                    similarities[ref_syst_name][specie]['map'].shape), verb_th=2)
                if np.min(similarities[ref_syst_name][specie]['map'].shape) > 0:
                    _max = np.max(similarities[ref_syst_name][specie]['map'])
                    _min = np.min(similarities[ref_syst_name][specie]['map'])
                    if _max > sim_max:
                        sim_max = _max
                    if _min < sim_min:
                        sim_min = _min

        fig, ax = plt.subplots(len(ref_systems_atoms), len(species))
        for ref_index, ref_syst_name in enumerate(similarities.keys()):
            for specie_index, specie in enumerate(similarities[ref_syst_name].keys()):
                pcm = ax[ref_index, specie_index].imshow(
                    similarities[ref_syst_name][specie]['map'], aspect='auto',
                    vmin=sim_min, vmax=sim_max, cmap=cmap)
                ax[ref_index, specie_index].set(
                    title='similarities with {} atoms in {}'.format(specie, ref_syst_name),
                    xlabel='index in {}'.format(atomic_system_name),
                    ylabel='Site index in {}'.format(ref_syst_name))
                ax[ref_index, specie_index].set_xticks(
                    np.arange(len(similarities[ref_syst_name][specie]['atoms_2_indexes'])),
                    similarities[ref_syst_name][specie]['atoms_2_indexes'])
                ax[ref_index, specie_index].set_yticks(
                    np.arange(len(similarities[ref_syst_name][specie]['atoms_1_indexes'])),
                    similarities[ref_syst_name][specie]['atoms_1_indexes'])

                cb = fig.colorbar(pcm, ax=ax[ref_index, specie_index])

        fig.set_size_inches(9.6, 9.6)
        if show_plot:
            plt.show()

        return fig, ax

    def plot_similarity_map_to_multi_ref_systems(self, atomic_system,
            ref_systems, atomic_system_name=None,
            sort_by_frac_coord_along_axis=None,
            atomic_system_sites=None, return_map_dict=True,
            sort_by_type=False, size_inches=[4.8, 4.8], show_plot=True):
        """
        Plot similarities to a selection of sites in multiple systems

        ref_systems should be a list of dicts organized as follows:
        [
            {
                'name': 'REF_0_NAME',
                'file': 'REF_0_FILE_NAME',
                'sites': LIST_OF_PICKED_SITE_INDEXES'
            },
            ...
        ]
        """
        # make a copy of ref_systems
        _ref_systems = ref_systems.copy()
        atoms, _name = get_ase_atoms(atomic_system)
        if sort_by_frac_coord_along_axis is not None:
            if sort_by_frac_coord_along_axis in ['a', 'b', 'c']:
                axis=['a', 'b', 'c'].index(sort_by_frac_coord_along_axis)
            elif sort_by_frac_coord_along_axis in [0, 1, 2]:
                axis=sort_by_frac_coord_along_axis
            else:
                raise ValueError(('sort_by_frac_coord_along_axis should be '
                                  'None, 0, 1, 2, \'a\', \'b\' or \'c\'.'))
            atoms = self.sort_atoms_by_fractional_coordinate(atoms, axis=axis)
        if atomic_system_name is None:
            atomic_system_name = _name
        ref_systems_atoms = []
        ref_sites = []
        for ref_index, ref_system in enumerate(_ref_systems):
            _atoms, _name = get_ase_atoms(ref_system['file'])
            _ref_systems[ref_index]['atoms'] = _atoms
            if ref_system['name'] is None:
                _ref_systems[ref_index]['names'] = _name

            if ref_system['sites'] is None:  # consider all sites
                _ref_systems[ref_index]['sites'] = list(range(len(_atoms)))

        if atomic_system_sites is None:
            atomic_system_sites = list(range(len(atoms)))

        sites = [{'index': i, 'type': atom.symbol}
                 for i, atom in enumerate(atoms)]

        # Define a list of ref sites as [{ref_name: '', site_index: i, site_type: '']}
        ref_sites = []
        similarities_by_type = {}
        for ref_syst in _ref_systems:
            for site_index in ref_syst['sites']:
                ref_sites.append({
                    'syst_name': ref_syst['name'],
                    'index': site_index,
                    'type': ref_syst['atoms'][site_index].symbol
                })
            similarities_by_type[ref_syst['name']] = self.get_similary_maps_by_type(
                ref_syst['atoms'], atoms)
        species = set([ref_site['type'] for ref_site in ref_sites])

        similarities = np.empty((len(sites), len(ref_sites)))
        similarities.fill(np.nan)

        self.print('species = {}'.format(species), verb_th=3)
        self.print('similarities_by_type = {}'.format(similarities), verb_th=3)
        self.print('similarity matrix will have shape')

        # fill the map if types match
        for i, site in enumerate(sites):
            for j, ref_site in enumerate(ref_sites):
                if site['type'] == ref_site['type']:
                    """
                    DEBUGGING
                    print('site: ', site)
                    print('ref_site: ', ref_site)
                    print(similarities_by_type[
                        ref_site['syst_name']][site['type']]['map'].shape)
                    """
                    # TODO: find index of ref_site['index'] and site['index'] in map (only )
                    _ = similarities_by_type[
                        ref_site['syst_name']][site['type']]
                    self.print('map indexes ({}, {})'.format(
                        _['atoms_1_indexes'].index(ref_site['index']),
                        _['atoms_2_indexes'].index(site['index'])))
                    similarities[i, j] = _['map'][
                        _['atoms_1_indexes'].index(ref_site['index']),
                        _['atoms_2_indexes'].index(site['index'])]

        fig, ax = plt.subplots()
        # TODO ? define vmin and vmax ?
        pcm = ax.imshow(similarities, aspect='auto',
                        cmap=cmap)
        cb = fig.colorbar(pcm, ax=ax, label='Similarity')
        ax.set_xticks(np.arange(len(ref_sites)),
                      ['{}-{}({})'.format(site['syst_name'], site['type'],
                                           site['index'])
                       for site in ref_sites],
                      rotation='vertical')
        ax.set_yticks(np.arange(len(sites)),
                      ['{}({})'.format(site['type'],
                                          site['index'])
                       for site in sites])
        ax.set(title='Similarities between site in {} and reference systems.'.format(
                atomic_system_name), xlabel='Sites in reference systems',
                ylabel='Sites in {}'.format(atomic_system_name))
        fig.set_size_inches(size_inches)
        fig.tight_layout()

        if show_plot:
            plt.show()

        return fig, ax

    def get_similarities_between_matching_sites(self, system_1, system_2, species=None,
                                                periodic_1=None, periodic_2=None):
        """
        Get similarities between corresponding sites in structures with identical site indexing

        Args:

        Returns:

        """
        full_sim =  self.get_similary_maps_by_type(system_1, system_2, species=species,
            periodic_1=None, periodic_2=None, matching_sites=True)
        similarities = {}
        for atom_type, sim_by_type in full_sim.items():
            similarities[atom_type] = {
                'indexes': sim_by_type['atoms_1_indexes'],
                'similarities': np.diag(sim_by_type['map'])
            }
        return similarities

    def get_multisystem_similary_maps_by_type(self, systems, species=None,
                                              periodic=True):
        """
        Get similarity maps by type and associated site mappings between mutiple structures

        TODO: add the possibility to merge similar sites (beyond a specified similarity
              thesrhold). The mapping should then include a list of equivalent sites.


        'system_indexes': -np.ones(nb_of_sites_by_type[atom_type]),
                'site_indexes_in_syst': -np.ones(nb_of_sites_by_type[atom_type]),
                'map':
        Args:

        Returns:
            multisyst_sim_by_type: dict
                A dictionary constructed as follows:
                    multisyst_sim_by_type = {
                        atom_type_1: {
                            'system_indexes': array of shape (1, n)
                            'site_indexes_in_syst': array of shape (1, n)
                            'map': similarity map of shape (n, n),
                        },
                        atom_type_2: {
                            ...
                    }
                where n and m are the number of atoms of atom_type_1 in
                all considered systems.
        """
        if not all([isinstance(s, Atoms) for s in systems]):
            systems = [get_ase_atoms(s) for s in systems]

        print('systems = ', systems)

        if not species:
            self.set_species(systems)
        else:
            self.set_species(species)

        self.print(f'species = {self.species}', verb_th=2)

        self.set_soap_descriptor()

        # TODO ? Merge equivalent sites in each structure

        # Calculate number of sites and initiazlize global similarities:
        nb_of_sites_by_type = {atom_type: 0 for atom_type in self.species}
        for syst_1_index, atoms_1 in enumerate(systems):
            for atom_type in self.species:
                nb_of_sites_by_type[atom_type] += atoms_1.symbols.count(atom_type)

        self.print(f'The total number of sites by atom types in all {len(systems)} '
                   f'considered systems is :\n{nb_of_sites_by_type}')

        multisyst_sim_by_type = {
            atom_type: {
                'system_indexes': -np.ones(nb_of_sites_by_type[atom_type],
                                           dtype=int),
                'site_indexes_in_syst': -np.ones(nb_of_sites_by_type[atom_type],
                                                 dtype=int),
                'map': -np.ones((nb_of_sites_by_type[atom_type],
                                 nb_of_sites_by_type[atom_type]))
            } for atom_type in self.species
        }

        # Initialize first site_index_1_by_type and first site_index_1_by_type
        site_index_1_by_type = {atom_type: 0 for atom_type in self.species}
        site_index_2_by_type = {atom_type: 0 for atom_type in self.species}
        for syst_1_index, atoms_1 in enumerate(systems):
            # Fill multisyst_sim_by_type 'system_indexes' and 'site_indexes_in_syst'
            for atom_type in self.species:
                i_0 = site_index_1_by_type[atom_type]
                indexes_in_syst_1 = [i for i, atom in enumerate(atoms_1)
                                     if atom.symbol == atom_type]

                multisyst_sim_by_type[atom_type]['system_indexes'][
                    np.arange(i_0,i_0+len(indexes_in_syst_1))] = syst_1_index

                multisyst_sim_by_type[atom_type]['site_indexes_in_syst'][
                    np.arange(i_0,i_0+len(indexes_in_syst_1))] = indexes_in_syst_1

            for syst_2_index, atoms_2 in enumerate(systems):
                if syst_2_index > syst_1_index:
                    break
                _verbosity = self.verbosity
                self.verbosity = 1
                similarities = self.get_similary_maps_by_type(atoms_1, atoms_2,
                    species=self.species, periodic_1=periodic, periodic_2=periodic)
                self.verbosity = _verbosity

                for atom_type in self.species:
                    i_0 = site_index_1_by_type[atom_type]
                    j_0 = site_index_2_by_type[atom_type]
                    indexes_in_syst_1 = [i for i, atom in enumerate(atoms_1)
                                         if atom.symbol == atom_type]
                    indexes_in_syst_2 = [i for i, atom in enumerate(atoms_2)
                                         if atom.symbol == atom_type]

                    row_indexes = np.arange(i_0,i_0+len(indexes_in_syst_1))
                    col_indexes = np.arange(j_0,j_0+len(indexes_in_syst_2))
                    block_indexes = np.ix_(row_indexes, col_indexes)
                    self.print(f'row_indexes of shape {row_indexes.shape}: {row_indexes}',
                               verb_th=3)
                    self.print(f'col_indexes of shape {col_indexes.shape}: {col_indexes}',
                               verb_th=3)
                    self.print(f'block_indexes of length {len(block_indexes)}: {block_indexes}',
                               verb_th=3)

                    self.print(('similarity[{}][\'map\'] of shape {} will be used '
                                'to update multisyst_sim_by_type[{}][\'map\'] block '
                                ' of size {} starting with indexes {}').format(atom_type,
                                    similarities[atom_type]['map'].shape, atom_type,
                                    len(block_indexes), (i_0, j_0)),
                               verb_th=2)

                    # Fill multisyst_sim_by_type 'map'
                    multisyst_sim_by_type[atom_type]['map'][block_indexes
                        ] = similarities[atom_type]['map']

                    if syst_2_index != syst_1_index:
                         # Fill with symetric block with transpose
                         multisyst_sim_by_type[atom_type]['map'][
                        np.ix_(np.arange(j_0,j_0+len(indexes_in_syst_2)),
                               np.arange(i_0,i_0+len(indexes_in_syst_1)))
                        ] = similarities[atom_type]['map'].T

                    self.print('multisyst_sim_by_type[{}][\'map\'] =\n{}'.format(
                        atom_type, multisyst_sim_by_type[atom_type]['map']), verb_th=3)

                    site_index_2_by_type[atom_type] += len(indexes_in_syst_2)

            # Increment site_index_1_by_type and re-initialize site_index_2_by_type
            for atom_type in self.species:
                site_index_1_by_type[atom_type] += atoms_1.symbols.count(atom_type)
                site_index_2_by_type[atom_type] = 0

        return multisyst_sim_by_type

    def plot_mutisyst_similarity_maps(self, multisyst_sim_by_type, title=None,
            fig_size_inches=(9.6, 4.8), cmap=None, subplots=None, show_plot=True):
        """
        Plot muti-system similarity maps by type as obtained with get_multisystem_similary_maps_by_type

        Args:
            multisyst_sim_by_type: dict
                Dictionary as obtained with get_multisystem_similary_maps_by_type.
                (see description therein).
            title: str or one (default is None).
                If None a title will be set automatically.
            fig_size_inches: tuple, list or None
                Figure size in inches (default is (9.6, 4.8))
            cmap: cmap, str or None (default is None)
                Set to 'default' to use the default matplotlib map.
            subplots: tuple, list or None (default is None)
                if None (1, nb_of_species) subplots will be used
            show_plot: bool (default is True)
                Whether plit should be shown or simply used to create
                fig and ax variables.
        """
        # TODO: adapt number of subplots to number of species
        if cmap == 'default':
            cmap = None
        elif cmap is None:
            colors = [(1, 0, 0), (1, 0.9, 0), (0, 0.75, 0)]  # Red, Gold, Green
            cmap = LinearSegmentedColormap.from_list('RedGoldGreen', colors)

        if subplots is None:
            fig, ax = plt.subplots(1, len(multisyst_sim_by_type.keys()))
        else:
            fig, ax = plt.subplots(subplots[0], subplots[1])

        # flatten ax to facilitate loop.
        ax_flat = ax.ravel()
        for specie_index, specie in enumerate(multisyst_sim_by_type.keys()):
            self.print('multisyst_sim_by_type[{}][map] has shape {}'.format(
                specie, multisyst_sim_by_type[specie]['map'].shape), verb_th=2)
            pcm = ax_flat[specie_index].imshow(multisyst_sim_by_type[specie]['map'],
                                               cmap=cmap)
            ax_flat[specie_index].set(title='simaliries between {} atoms'.format(specie),
                                 xlabel='system-site indexes',
                                 ylabel='system-site indexes')
            tick_labels = [f'{i}-{j}' for i, j in
                     zip(multisyst_sim_by_type[specie]['system_indexes'],
                         multisyst_sim_by_type[specie]['site_indexes_in_syst'])]
            n_sites = len(multisyst_sim_by_type[specie]['system_indexes'])
            ax_flat[specie_index].set_xticks(np.arange(n_sites), labels=tick_labels)
            ax_flat[specie_index].set_yticks(np.arange(n_sites), labels=tick_labels)

            cb = fig.colorbar(pcm, ax=ax_flat[specie_index])

        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
        if show_plot:
            plt.show()
        return fig, ax


def expand_dict(params):
    # Find all keys that have lists as values
    list_keys = [key for key, value in params.items()
                 if isinstance(value, (list, tuple))]

    # Find all keys that have single values
    single_keys = [key for key, value in params.items()
                   if not isinstance(value, (list, tuple))]

    # Generate all combinations of list values
    list_combinations = product(*(params[key] for key in list_keys))

    # Create a list to store the expanded dictionaries
    expanded_dicts = []

    # Iterate over all combinations
    for combination in list_combinations:
        # Create a new dictionary for this combination
        new_dict = params.copy()

        # Update the new dictionary with the current combination
        for key, value in zip(list_keys, combination):
            new_dict[key] = value

        # Add the new dictionary to the list
        expanded_dicts.append(new_dict)

    return expanded_dicts


def test_soap_parameters(systems, change_parameters, ref_soap_parameters=None,
                         data_path='atomicEnvComparisons_data', 
                         fig_file_basename='test_soap_param_fig',
                         csv_file_name='test_soap_parameters.csv', 
                         show_similarity_plots=True, verbosity=1):
    """
    Test the influence of SOAP parameters on a list of structures

    Accuracy is evaluated by comparison with a set of
    highly-accurate ref_soap_parameters.

    The studied parameters are given in the change_parameters
    dictionary, which may contain for each parameters a single
    differing from the reference parameters, or a list/tuple
    of values, in which case similarities betweenall sites in
    all structures will be measured for all combinations of
    parameters.

    Parameters located in nested dicts of the SOAP parameters
    will be considered as well (with a depth of only one and
    assuming that parameters in different nested dictionaries
    do not have similar names).
    
    Args:
        systems: list or tuple of ASE Atoms, pymaten Structures, coord. files

        change_parameters: dict
            Dictionary of parameters and values that change from
            the reference set of parameters (ref_soap_parameters).
            Single values or list/tuples may be used for every
            considered paremeter.
        ref_soap_parameters: dict (default is None)
            Set of parameters that should be used as the
            high-accuracy reference t evaluate performance
            and accuracy.
        verbosity: int (default is 1)
            Verbosity level

    Returns:
        TOBECOMPLETED
    """
    atoms_list = [get_ase_atoms(system) for system in systems]

    if not ref_soap_parameters:
        aecd = AtomEnvComparisonsData()
        ref_soap_parameters = aecd.soap_paramaters

    change_param_combinations = expand_dict(change_parameters)

    if verbosity >= 2:
        print('change_param_combinations:\n', change_param_combinations)
        print()

    species = set()
    for atoms in atoms_list:
        species = species.union(atoms.symbols.species())
    species = list(species)

    if verbosity >= 1:
        print(f'Species = {species}')

    soap_param_list = []
    similarities_data = {atom_type: {} for atom_type in species}
    
    fixed_param_description = ', '.join(
        [f'{k}: {v}' for k, v in change_parameters.items()
         if not isinstance(v, (list, tuple))])
             
    change_param_descriptions = []
    is_data_dict_initialized = False
    data_index_by_type = {atom_type: 0 for atom_type in species}
    for param_set_index, change_param in enumerate(change_param_combinations):

        change_param_descriptions.append(
            ', '.join([f'{k}: {v}' for k, v in change_param.items()
                       if isinstance(change_parameters[k], (list, tuple))]))
        
        if verbosity >= 3:
            print('change_param for param_set_index {} with description {}:\n{}'.format(
                param_set_index, change_param_descriptions[-1], change_param))
                     
        soap_param = deepcopy(ref_soap_parameters)
        for k, v in change_param.items():
            if k in soap_param.keys():
                soap_param[k] = v
            else:
                for ref_param_key, ref_param_val in soap_param.items():
                    if isinstance(ref_param_val, dict) and k in ref_param_val.keys():
                        soap_param[ref_param_key][k] = v

        if verbosity >= 3:
            print('soap_param:\n', soap_param)

        soap_param_list.append(soap_param)
        tic = perf_counter()
        aecd = AtomEnvComparisonsData(soap_parameters=soap_param, species=species,
                                      verbosity=1)
        
        if verbosity >= 2:
            print('Similarities will be calculated with SOAP parameters:\n', 
                  aecd.soap_parameters)
        
        multisyst_sim_by_type = aecd.get_multisystem_similary_maps_by_type(
            atoms_list, species=species)
        sim_calc_time = perf_counter() - tic
        
        n_features = aecd.soap_descriptor.get_number_of_features()

        for atom_type in species:
            mssbt = multisyst_sim_by_type[atom_type]
            size = mssbt['map'].size
            shape = mssbt['map'].shape
            if len(soap_param_list) == 1:
                similarities_data[atom_type] = {
                    'system_indexes': mssbt['system_indexes'],
                    'site_indexes_in_syst': mssbt['site_indexes_in_syst'],
                    'all_sim': mssbt['map'].flatten().reshape((1, size)),
                }
            else:
                similarities_data[atom_type]['all_sim'] = np.concatenate(
                    (similarities_data[atom_type]['all_sim'],
                    mssbt['map'].flatten().reshape((1, size))))

            if verbosity >= 3:
                print(f'multisyst_sim_by_type[{atom_type}][\'map\'] has '
                      f'shape {shape} and size {size}.')
                print(f'similarities_data[{atom_type}][\'all_sim\'] now has '
                      'shape {}'.format(similarities_data[atom_type]['all_sim'].shape))

            if not is_data_dict_initialized:
                data_dict = {k: [v] for k, v in change_param.items()}
                data_dict['atom_type'] = [atom_type]
                data_dict['sim_calc_time_s'] = [sim_calc_time]
                data_dict['n_features'] = [n_features]
                data_dict['index_by_type'] = [data_index_by_type[atom_type]]
                is_data_dict_initialized = True
            else:
                [data_dict[k].append(v) for k, v in change_param.items()]
                data_dict['atom_type'].append(atom_type)
                data_dict['sim_calc_time_s'].append(sim_calc_time)
                data_dict['n_features'].append(n_features)
                data_dict['index_by_type'].append(data_index_by_type[atom_type])

            # increment data_index_by_type
            data_index_by_type[atom_type] += 1
            
            if verbosity >= 3:
                print(f'Similarities have been treated for {atom_type} atoms '
                      f'with param_set_index {param_set_index}, data_dict has '
                      f'been updated to:\n{data_dict}\n')

    if verbosity >= 2:
        print(f'fixed_param_description: {fixed_param_description}')
        print('change_param_descriptions:\n', change_param_descriptions)
        print('soap_param_list:\n', soap_param_list)
    
    # Store similarities_data[atom_type]['all_sim'] in data_path
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        
    for atom_type in species:
        file_name = os.path.abspath(os.path.join(
            data_path, f'similarities_{atom_type}.npy'))
        np.save(file_name, similarities_data[atom_type]['all_sim'])

    # Calculate similarities with reference parameters:
    aecd = AtomEnvComparisonsData(soap_parameters=ref_soap_parameters, species=species,
                                  verbosity=1)
    if verbosity >= 2:
        print('Similarities will be calculated with the reference SOAP parameters:\n', 
              aecd.soap_parameters)
    ref_multisyst_sim_by_type = aecd.get_multisystem_similary_maps_by_type(
            atoms_list, species=species)

    # adapt subplots to nb of species   
    num_plots = len(species)
    # Determine the number of rows and columns
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    # Create the subplots
    fig, ax = plt.subplots(num_rows, num_cols, 
                           figsize=(num_cols * 4, num_rows * 4))

    
    data_dict.update({param: [] for param in ['resid_err', 'RMSE_to_ref', 
                                              'linreg_a', 'linreg_b', 
                                              'RMSE_to_linreg']})
    ax_flat = ax.ravel()
    
    ref_sim = {}
    for type_index, atom_type in enumerate(species):
        ref_sim[atom_type] = ref_multisyst_sim_by_type[atom_type]['map'].flatten()
    
    for i in range(len(soap_param_list)):
        for type_index, atom_type in enumerate(species):
            
            """
            if plot_relative_errors:
                x = 1 - ref_sim
                y = x - similarities_data[atom_type]['all_sim'][i]
                y = np.divide(y, x, out=np.full_like(y, np.nan),
                              where=~np.isclose(x, 0, atol=1e-5))
            """
            x = ref_sim[atom_type]
            y = similarities_data[atom_type]['all_sim'][i]

            series, results = np.polynomial.Polynomial.fit(x, y, 1, full=True)
            # print(f'Polynomial fit series: {series}')
            (b, a) = series.convert().coef
            residual_errors = results[0][0]
            rmse = np.sqrt(np.mean((x - y)**2))
            rmse_to_linreg = np.sqrt(np.mean(((a * x + b) - y)**2))

            data_dict['resid_err'].append(residual_errors)
            data_dict['RMSE_to_ref'].append(rmse)
            data_dict['RMSE_to_linreg'].append(rmse_to_linreg)
            data_dict['linreg_a'].append(a)
            data_dict['linreg_b'].append(b)

            ax_flat[type_index].plot(x, y, label=change_param_descriptions[i],
                    marker='o', markersize=3, ls='none')
    
            color = ax_flat[type_index].lines[-1].get_color()
    
            min_max_x = np.array((np.min(x),np.max(x))) 
            ax_flat[type_index].plot(min_max_x, a* min_max_x + b, label='linreg',
                    marker='none', ls='-', color=color)
    
        ax_flat[type_index].set(xlabel='Site similarities with ref SOAP parameters',
            ylabel='Site similarities with selected param.',
            title=f'{atom_type} atoms, ' + fixed_param_description)
        ax_flat[type_index].legend()

    # WARNING: svg can lead to quickly lead to huge structure files.
    for ext in ['png']:
        fig_file_name = os.path.abspath(os.path.join(data_path, f'{fig_file_basename}.{ext}'))
        plt.savefig(fig_file_name)
        print(f'Figure saved as {fig_file_name}.')

    df = pd.DataFrame(data_dict)
    if verbosity >= 2:
        print('test_soap_parameters Dataframe:\n', df)
    
    csv_file_name = os.path.abspath(os.path.join(data_path, csv_file_name))
    
    df.to_csv(csv_file_name)
    print(f'Data table saved as {csv_file_name}.')
    
    # Store metadata json file:
    file_name = os.path.abspath(os.path.join(data_path, 'soap_parameters.json'))
    with open(file_name, 'w') as f:
        json.dump({
                    "ref_soap_parameters": ref_soap_parameters,  
                    "change_parameters": change_parameters,                    
                  }, f, indent=4)
    
    if show_similarity_plots:
        plt.show()

    return df, csv_file_name, fig, ax
