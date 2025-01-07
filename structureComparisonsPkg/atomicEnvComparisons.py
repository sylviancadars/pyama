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

plt.rcParams['svg.fonttype'] = 'none'

# Create a custom color map (here Red-Gold-Green)
colors = [(1, 0, 0), (1, 0.9, 0), (0, 0.50, 0)]  # Red, Gold, Green
cmap = LinearSegmentedColormap.from_list('RedGoldGreen', colors)

class AtomEnvComparisonsData(BaseDataClass):
    """
    A class to compare individual atomic environments based on SOAP (or other) kernel

    TO BE COMPLETED
    """
    def __init__(self, average_kernel_metric='polynomial', average_kernel_degree=3,
                 soap_r_cut=5.0, soap_n_max=5, soap_l_max=5, soap_sigma=1.0,
                 soap_weighting={'function': 'poly', 'c': 1, 'm': 1, 'r0': 5.0},
                 soap_periodic=True, soap_rbf='polynomial',
                 soap_compression={'mode': 'crossover'},
                 soap_sparse=False, species=None, verbosity=1):
        super().__init__(verbosity=verbosity,
                         print_to_console=True,
                         print_to_file=False)

        self.average_kernel_metric = average_kernel_metric
        self.average_kernel_degree = average_kernel_degree
        self.soap_r_cut = soap_r_cut
        self.soap_n_max = soap_n_max
        self.soap_l_max = soap_l_max
        self.soap_sigma = soap_sigma
        self.soap_weighting = soap_weighting
        self.soap_periodic = soap_periodic
        self.soap_rbf = soap_rbf
        self.soap_compression = soap_compression
        self.soap_sparse = soap_sparse
        self.verbosity = verbosity

        if species is not None:
            self.species = self.set_species(species)
        else:
            self.species = None
        # TODO: add soap.weighting
        self.soap_descriptor = None

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

        self.soap_descriptor = SOAP(species=self.species, r_cut=self.soap_r_cut,
                                    n_max=self.soap_n_max, l_max=self.soap_l_max,
                                    sigma=self.soap_sigma, weighting=self.soap_weighting,
                                    periodic=self.soap_periodic, rbf=self.soap_rbf,
                                    compression=self.soap_compression,
                                    sparse=self.soap_sparse)

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
        
        print(self.species)
        
        if periodic_1 is None:
            periodic_1 = all(atoms_1.pbc)
        if periodic_2 is None:
            periodic_2 = all(atoms_2.pbc)

        features_list = []
        for periodic, atoms in zip([periodic_1, periodic_2], [atoms_1, atoms_2]):
            if periodic == self.soap_periodic:
                if self.soap_descriptor is None:
                    self.set_soap_descriptor()
                desc = self.soap_descriptor
            else:
                desc = SOAP(species=self.species, r_cut=self.soap_r_cut,
                    n_max=self.soap_n_max, l_max=self.soap_l_max, sigma=self.soap_sigma,
                    periodic=periodic, rbf=self.soap_rbf, weighting=self.soap_weighting,
                    compression=self.soap_compression, sparse=self.soap_sparse)

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
                    print('site: ', site)
                    print('ref_site: ', ref_site)
                    print(similarities_by_type[
                        ref_site['syst_name']][site['type']]['map'].shape)
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

        TO BE SET USING get_similary_maps_by_type WITH matching_sites=True
        similarities[atom_type] could be simplified (one indexing by type and a vector of
        similarity value rather than a map.
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




