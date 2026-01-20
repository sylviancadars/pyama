"""
Module dedicated to the calculation of distances for structure comparisons

@author : Sylvian Cadars, Institut de Recherche sur les Céramiques, CNRS,
Université de Limoges

Last updated : 2020/09/02
"""
from pymatgen.core.structure import IStructure, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
from time import perf_counter
from scipy.spatial.distance import pdist, squareform, cosine
from multiprocessing import cpu_count
from plotly import express as px
from pandas import DataFrame
from dscribe.descriptors import SOAP, ValleOganov
from dscribe.kernels import REMatchKernel

from sklearn.preprocessing import normalize
from sklearn.manifold import MDS

from pyama.utils import get_ase_atoms, get_pymatgen_structure, BaseDataClass


class distanceMatrixData(BaseDataClass) :
    """
    Class designed to calculate and store information on distance matrices between structures

    Detailed description

    Properties :
    - Dmatrix : N by N ndarray with N the number of structures
    - IDs : array of IDs of the structures used to calculate Dmatrix
    - R : distance vector used to calculate the distance matrix
    - sigma : Gaussian smearing factor used to calculate the distance matrix
    - distanceMethod : 'cosine'
    - fingerprintMethod : 'OganovValle2009'
    - saveFileName : name of file where distance matrix data should be stored (in pickle format : extension should be .pkl)

    """

    def __init__(self, IDs:list=[], descriptions:list=[], distanceMethod:str='cosine',
                 fingerprintMethod:str='OganovValle2009', sigma:float=0.02,
                 R=None, Rmax:float=6.0, Rsteps:float=512, print_performance=False,
                 verbosity=1, **kwargs) :
        """
        Class initializer with

        Parameters
        ----------
        IDs : list, optional
            list of IDs numbers (int) associated with structures for which
            fingerprints and/or distances shall be calculated. May be used to
            link structures to databases. The default is [].
        descriptions : list, optional
            list of descriptions (str) of structures for which fingerprints
            and/or distances shall be calculated. Descriptions will be used as
            tick labels in cosine matrix plots. Default is [], in which case
            IDs (if available) will be used instead.
        distanceMethod : str, optional
            Method used to calculate the distance between structures based on
            their fingerprints. The default is 'cosine'.
        fingerprintMethod : str, optional
            Methoid used to calculate structure fingerprint. The default is
            'OganovValle2009', which corresponds to g(r)-1 (-> 0 for large r
            values).
        sigma : float, optional
            Gaussian smearing factor to avoid numerical errors fingerprint
            calculations. The default is 0.1.
        R : TYPE, optional
            Vector of radial distances. The default is [], in which case Rmax,
            Rstep and/or Rsteps values will be used to calculate R.
        Rmax : float, optional
            DESCRIPTION. The default is 6.0.
        Rsteps : float, optional
            DESCRIPTION. The default is 512.
        print_performance: bool (default is False)
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        super().__init__(verbosity=verbosity,
                         print_to_console=True,
                         print_to_file=False)

        self.IDs = IDs
        self.descriptions=descriptions
        self.distanceMethod = distanceMethod
        self.fingerprintMethod = fingerprintMethod
        self.sigma = sigma

        # Define or read R vector
        if R is None:
            R = []

        if len(R)>1 :
            self.R = R
            self.R = np.resize(R,(len(R),))
            self.Rmax = R[-1]
            self.Rsteps = len(R)
        else :
            # Define radius vector from 0 to cutoff R_max
            self.R = Rmax*(1/(Rsteps-1))*np.arange(Rsteps)
            self.Rmax = Rmax
            self.Rsteps = Rsteps
        if 'saveFileName' in kwargs :
            self.saveFileName = kwargs['saveFileName']
        else :
            self.saveFileName = 'tmp.pkl'
        self.print_performance = print_performance

    
    def calculate_all_partial_RDFs(self, structure, showPlot=False,
                                       rel_y_shift=0.25, n_jobs=1, 
                                       **vo_kwargs):
        """
        Calculate all partial radial distributions functions for a structure

        Args
            structure: pymatgen Structure object

            showPlot: bool (default is False)
                Only for debugging purposes. Program will interrupt until each
                figure is closed.

            rel_y_shift: float (default is 0.25)
                relative y shift beween different partials in fraction of the
                amplitude range of the first partial.

        TODO: offer the possibility to return figures.

        Returns:
            - partials: numpy.array of floats
                (len(types), len(types), len(R))
            - types: list of types of atoms in the same order
                 as in partials.
                to get a specifi partial A-B RDF use:
                partials[type.index(A), type.index(B), :]
        """
        if self.print_performance:
            tic = perf_counter()
        R = self.R
        sigma = self.sigma
        Rmax = self.Rmax
        DELTA = R[1]-R[0] # discretization step
        V = structure.volume # cell volume

        # Sort a copy of structure by atom types
        struct = structure.copy()
        struct.sort()
        # Get list of elements
        types = [i.name for i in struct.types_of_species]
        nb_of_atoms_of_type = [len(struct.indices_from_symbol(t)) for t in types]
        
        atoms = get_ase_atoms(structure)

        vo = ValleOganov(species=types, function="distance", 
                         n=len(self.R), sigma=self.sigma, r_cut=self.Rmax, 
                         **vo_kwargs)        
        tic_create_descriptor = perf_counter()
        descriptor = vo.create(atoms, n_jobs=n_jobs)
        tac_create_descriptor = perf_counter()

        all_partials = np.zeros((len(types), len(types), len(R)))
        
        for type_ind_A, type_A in enumerate(types):
            A_indexes = struct.indices_from_symbol(type_A)
            N_A = nb_of_atoms_of_type[type_ind_A]
            for type_ind_B, type_B in enumerate(types[:type_ind_A+1]):
                
                all_partials[type_ind_A, type_ind_B, :] = vo.get_location([type_A, type_B])
                all_partials[type_ind_B, type_ind_B, :] = all_partials[type_ind_B, type_ind_A, :]

        if self.print_performance:
            print(('Execution of function calculate_all_partial_RDFs for {} took '
                   '{:.1f} ms (including {:.1f} ms to create descriptor '
                   'get_all_neighbors()).').format(structure.formula,
                  1000*(perf_counter()-tic),
                  1000*(tac_create_descriptor-tic_create_descriptor)))

        if showPlot:
            fig, ax = plt.subplots()
            legend=[]
            tot_y_shift=0
            y_shift = rel_y_shift*(max(all_partials[0,0,:])-
                                   min(all_partials[0,0,:]))
            for i in range(all_partials.shape[0]):
                for j in range(all_partials.shape[1]):
                    ax.plot(R,all_partials[i,j,:] + tot_y_shift)
                    tot_y_shift += y_shift
                    legend.append(types[i] + '-' + types[j])
            ax.set(xlabel='R (Angstroms)', ylabel='g_AB(R)',
                   title=struct.formula + ' - Partial radial distribution functions')
            ax.legend(legend)

            return all_partials, types, fig, ax
        else:
            return all_partials, types


    def calculate_all_reduced_partial_RDFs(self, structure, showPlot=False):
        """
        Calculate reduced partial RDFs for types of atoms in a structure

        Args:
            structure: pymatgen Structure object
            showPlot: bool
                whether partial PDFs should be shown. For debugging purposes
                only. The user will have to close the plot for the program to
                continue

        Returns:
            G_AB: numpy array of dimension (len(types), len(types), len(R))
            types: atom types in the order used in G_AB
        if showPlot is True
            fig: figure handle
            ax:  axes handle
        """
        rho_0 = structure.num_sites/structure.volume  # Number of atoms/Angstrom^3
        if not showPlot:
            g_AB, types = self.calculate_all_partial_RDFs(structure,
                                                          showPlot=showPlot)
        else:
            g_AB, types, fig, ax = self.calculate_all_partial_RDFs(structure,
                                                                   showPlot=True)
        G_AB = 4*np.pi*rho_0*self.R*(g_AB-1)

        if showPlot:
            return G_AB, types, fig, ax
        else:
            return G_AB, types

    def calculate_cosine_distance(self, structure_1: Structure, structure_2: Structure,  
                                  show_plot: bool=False, system_names: list=None, 
                                  n_jobs=1, **vo_kwargs):
        """
        Docstring for calculate_cosine_distance
        
        :param self: Description
        :param structure_1: Description
        :type structure_1: Structure
        :param structure_2: Description
        :type structure_2: Structure
        :param show_plot: Description
        :type show_plot: bool
        :param system_names Description
        :type system_names: list
        :param vo_kwargs: Description
        """
        atoms_1 = get_ase_atoms(structure_1)
        atoms_2 = get_ase_atoms(structure_2)
        species = set.union(set(atoms_1.get_chemical_symbols()), 
                            set(atoms_2.get_chemical_symbols()))
        vo = ValleOganov(species=species, function="distance", 
                         n=len(self.R), sigma=self.sigma, r_cut=self.Rmax, 
                         **vo_kwargs)      
        tic_create_descriptor = perf_counter()
        descriptor = vo.create([atoms_1, atoms_2], n_jobs=n_jobs)

        print(f"2-structures descriptor (features) of shape {descriptor.shape}.")

        cosine_distance = squareform(pdist(descriptor, metric='cosine'))[0][1]

        if show_plot:
            shift = 0.0
            legend=[]
            fig, ax = plt.subplots()

            for A_index, A_type in enumerate(species):
                for B_index, B_type in enumerate(species):
                    if B_index > A_index:
                        continue
                    partial_1 = descriptor[0][vo.get_location((A_type, B_type))]
                    partial_2 = descriptor[1][vo.get_location((A_type, B_type))]
                    
                    ax.plot(self.R, partial_1 + shift, color='blue', 
                            label=f"System 1 {A_type}-{B_type}")
                    ax.plot(self.R, partial_2 + shift, color='red', 
                            label=f"System 2 {A_type}-{B_type}")
                    ax.plot(self.R, partial_2 - partial_1 + shift, color="black", 
                            label=f"{A_type}-{B_type} difference")
                    shift += 10.0

            ax.set(xlabel='r (A)', ylabel='Intensity')
            title = f"distance {cosine_distance:.3f}; partial PDFs"
            if system_names:
                title = ' vs '.join(system_names) + ': ' + title
            ax.set_title(title)
            ax.legend()
            plt.show()
            
            return cosine_distance, fig, ax

        else:
            return cosine_distance

    def set_stucture_ids(self, structures, structureIDs=[]):
        """
        Set structureIDs property based on a list of atomic systems

        Each system may be ASE Atoms or Pymatgen Structure

        Args:
            structures: list or tuple
                List of systems used to set associated structureIDs
            structureIDs: list ot tuple
                If not set, IDs will correspond to structure indexes in
                structures argument.

        """
        if not structureIDs or not(len(structureIDs)):
            try :
                if len(self.IDs) != len(structures):
                    # Setting default structure IDs between 1 and N where N
                    # is the number of considered structures
                    self.IDs = np.arange(len(structures))+1
                # Else : using the internal list of structure IDs : self.IDs
            except AttributeError:
                self.IDs = np.arange(len(structures))+1
        elif len(structureIDs) != len(structures):
            raise ValueError('Array or list of IDs should be the same size as the list of structures.')
        else:
            self.IDs = structureIDs

    def calculate_distance_matrix(self, structures, species=None, structureIDs=[], 
                                  n_jobs=None, return_plot=True, show_plot=False, 
                                  figure_title=None, **vo_kwargs):
        """
        Calculate cosine distance (between 0 and 1) matrix for a list of structures
        
        Calculation is based on the Valle Oganov (2009) distance descriptors
        as implemented in Dscribe.

        Args:
            structures: List of Structure
                 
            structureIDs: Description
        """
        # Adapt to accept list of ASE Atoms or Pymatgen Structure instances
        self.set_stucture_ids(structures=structures, structureIDs=structureIDs)
        self.Dmatrix = get_distance_matrix_from_valle_oganov_dscribe(
            structures, species, function="distance", sigma=self.sigma, 
            n=len(self.R), r_cut=self.Rmax, n_jobs=n_jobs, 
            return_plot=False, show_plot=False)
        
        if return_plot:
            print('Dmatrix property has been updated.')
            fig, ax = self.plot_distance_matrix(figure_title=figure_title)
            if show_plot:
                fig.show()
            return self.Dmatrix, fig, ax
        else:
            return self.Dmatrix
    
    def plot_distance_matrix(self,figure_title:str='',tickLabels:list=[],
                             axesLabel:str='',tickLabelsFontSize:int=0,
                             xticklabelsRotation=45, cmap=None) :
        """
        Plot distance matrix

        Args:
            figure_title: str, OPTIONAL
                If specified, will insert figure_title+' : ' before title
            tickLabels: list, OPTIONAL
                default is [] in wich case structure IDs (if any) or structure
                indexes will be used.
            axesLabel: str, OPTIONAL
                Default is ''.
            tickLabelsFontSize: int, OPTIONAL
                will change font size of tick labels (e.g. set to 6 or less if
                matrix size is equal to 50).
            xticklabelsRotation: float or {'horizontal','vertical'}, OPTIONAL
                Based on matplotlib.text.Text rotation option. Default is 45.
            cmap: str, matplotlib.colors.LinearSegmentedColormap or None
                if None a red-gold-green map will be used. If 'default' the default
                matplotlib cmap (blue-green-yellow) will be used.

        Returns:
            fig : figure.Figure
                distance matrix figure handle
            ax :  axes handle
        """
        if not figure_title:
            title = 'Distance matrix between structures'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = 'Cosine distance matrix between structures'

        if len(tickLabels) == 0 or len(tickLabels) != len(self.IDs):
            if len(self.descriptions) == len(self.IDs):
                tickLabels = self.descriptions
                if len(axesLabel) == 0:
                    axesLabel = 'Structure descriptions'
            else :
                tickLabels = [str(ID) for ID in self.IDs]
                if len(axesLabel) == 0:
                    axesLabel = 'Structure ID'

        fig, ax =  plot_distance_matrix(distance_matrix=self.Dmatrix,
            structure_names=tickLabels, system_name=figure_title,
            axesLabel=axesLabel, tickLabelsFontSize=tickLabelsFontSize,
            xticklabelsRotation=xticklabelsRotation, cmap=cmap)

        return fig, ax

    def sort_structures_by_distance_to_ref(self, structures,
                                           ref_struct_or_index=0,
                                           ref_struct_description=None,
                                           return_indexes=True):
        """
        Sort structure by increasing distance to a reference structure.

        Reference structure may be in list or an external pymatgen structure

        Args:
            structures: list or tuple
                list of pymatgen structure objects
            ref_struct_or_index: int or (I)Structure (default is 0)
                reference structure index in list or

        Returns:
            sorted_structures:
                reference structure placed first, then other structures
            indexes_in_orig_struct:
                Indexes of the sorted structures in the original structure list.
                If the reference structure is external the length of
                indexes_in_orig_struct
            indexes_in_sortde_structures
        """
        structures_with_ref = structures.copy()
        if isinstance(ref_struct_or_index, (Structure, IStructure)):
            # In this case the reference structure is considered as external and
            # will be appended to the end of structures_with_ref
            ref_struct = ref_struct_or_index
            structures_with_ref.append(ref_struct)
            ref_struct_index = len(structures_with_ref) - 1
            is_ref_external = True
            indexes_in_orig_struct = list(range(len(structures))) + [-1]
        elif isinstance(ref_struct_or_index, int):
            # Reference structure is internal
            if ref_struct_or_index < 0 or ref_struct_or_index >= len(structures):
                raise ValueError(('ref_struct_or_index should be between 0 and'
                                 'the number of structure ({})').format(
                                 len(structures)))
            ref_struct = structures_with_ref[ref_struct_or_index]
            ref_struct_index = ref_struct_or_index
            is_ref_external = True
            indexes_in_orig_struct = list(range(len(structures)))
        else:
            raise TypeError(('ref_struct_or_index should be a pymatgen '
                             '(I)Structure or a valid (< {}) structure index.'
                             ).format(len(structures)))
        sorted_structures = [ref_struct]
        distances_to_ref = []
        for struct_index, struct in enumerate(structures_with_ref):
            if struct_index != ref_struct_index:
                distances_to_ref.append(self.calculate_cosine_distance(ref_struct,
                                                                       struct))
            else:
                distances_to_ref.append(0.0)
        sorted_indexes = list(np.argsort(np.array(distances_to_ref)))
        sorted_structures = [structures_with_ref[i] for i in sorted_indexes]
        indexes_in_orig_struct = [indexes_in_orig_struct[i] for i in sorted_indexes]

        if return_indexes:

            return sorted_structures, sorted_indexes, indexes_in_orig_struct
        else:
            return sorted_structures
    

    def save_data_to_file(self,saveFileName='') :
        """
        Store distanceMatrixData object as a pickle file.
        """

        if len(saveFileName) == 0 :
            saveFileName = self.saveFileName

        with open(saveFileName, 'wb') as f:
            # TODO: check performance of pickle
            # pickle.dump(self, f)
            self.saveFileName = saveFileName
        print('distanceMatrixData saved in file : ',saveFileName)


    def select_distant_structures(self, structures, n_structures, initial_selection=None,
                                  distance_threshold=0.01, distance_method='max_min', 
                                  structure_ids=None, 
                                  structure_descriptions=None, 
                                  sorting_property_values=None, sorting_property_weight=0., 
                                  sorting_order='ascending', 
                                  sorting_property_label='property', 
                                  n_jobs=None, seed=None, 
                                  mds_plot_sizeref=0.01, mds_plot_sizemin=3, mds_plot_opacity=0.7, 
                                  mds_plot_fixed_size=10, description=None):
        """
        Select distant structures, possibly with a penalty associated with a property
        
        The method used [0, 1] cosine distances as described by Valle and Oganov, and 
        as implemented in the Dscribe library. Distances can be maximied with a 
        maxmin (each structure as facr as possible to the closest already-selected) or 
        maxmean (maximum avearage distance to all already-selected structures) algorithm, 
        and can be balanced with sorting_property_values associated with each structure 
        that should either be minimized ('ascending' sorting_order) or maximized ('descending' 
        sorting_order), using a sorting_property_weight w_p.

        Args:
            structures: list or tuple
                List of structures, ideally in ASE Atoms format. Pymatgen structures
                or file names are accepted and will be automatically converted to 
                ASE Atoms (which can be a bit long for a large number of structures)
            n_structures: int
                Number of structures to select.
            initial_selection: list, str or None (default is None)
                List of preselected structure IDs or pre-selection mode, 
                including:
                    * 'good_structures': the best 10 structures in goodPOSCARS
                    * 'best_structure': the best structure (according to fitness)
                    * 'lowest_energy': the lowest_energy structure
                If None, the first structure is chosen randomly.
            distance_thresded hold: float (default is 0.01)
                Minimum allowed distance between the considered structure and already-selected structures.
                Structures discared based on thsi criterion will be identified as "discarded" in the 
                MDS plot (shown if show_plot is True).
            distance_method: str (default is 'max_min')
                Choose method between 'max_average' (maximize global distance to all others at each step) 
                or "max_min" (maximize distance to closest strutcure at each step).
            structure_ids: list, tuple or None (default is None)
                IDs associated with the provided structures (should be have the same length)
                If None, (zero-based) indexes will be used as IDs.
            structure_descriptions: list, tuple or None (default is None)
                List of str corresponding to descriptions of the provided structures.
                If None, descriptions will be set using the structure IDs and compositions.  
            sorting_property_values: list, tuple or array (default is None)
                Values that will be used in combination with distances to select the structures.
                Should be used with sorting_property_weight > 0 (see definition below). 
                If None, distances between structures will be the only considered criterion. 
            sorting_order: str (default is 'ascending')
                Whether sorting_property_values should be sorted in ascending (as typically the case 
                for energies) or descending order (see sorting_property_weight definition)
            sorting_property_label: str (default is 'property')
                Name of the property balancing distances. Eg. "energy_per_atom", "bulk_modulus", etc...
            sorting_property_weight: float (default is 0.)
                Weight (w_p) associated with the property (p) balancing the distances between structures, 
                between 0 (no penalty, the default, in which case only distances matter) to 1 in 
                which case distance will not even matter and the property dominates entirely.
                Depending on sorting_order, the term that one tries to maximize may be :
                    'ascending': (1 - w_p) * (dist_average) + w_p * (1 - ((p - p_min) / (p_max-p_min))
                    'descending': (1 - w_p) * (dist_average) + w_p * (p - p_min) / (p_max-p_min)
                Typical case is when the property is the structure energy. In this case sorting_order
                should be 'ascending' and w_p values closer to 1 will favor stable structures.
            n_jobs: int (default is 1)
                Number of processors used to (re)calculate the full distance matrix.
            mds_plot_sizeref: float (default is 0.01)
                Marker size (diameter) factor reflecting relative energies in the MDS plot. 
            mds_plot_sizemin: int (default is 3)
                Minimum marker size (diameter) reflecting relative energies in the MDS plot.
            mds_plot_opacity: float (default is 0.7)
                Marker opacity in MDS plot.
            mds_plot_fixed_size: int (default is 10)
                Marker diameter in the plots when sorting_property_values are not provided.
            description: str or None (default is None)
                description of the considered set, used in the MDS plot title.

        Returns:
            results: dict
                dictionary containing information in the selected structures
            mds_plot_fig: plotly figure object
                Figure associated with the MDS plot.
        """
        if not n_jobs:
            n_jobs = cpu_count()

        # Convert structures to a list of ASE Atoms if this is not already the case.
        atoms_list = [get_ase_atoms(s) for s in structures]

        # TODO: first check whether matrix exists         
        self.calculate_distance_matrix(atoms_list, n_jobs=n_jobs)

        if structure_ids is None:
            structure_ids = list(range(len(atoms_list)))
        elif len(structure_ids) != len(atoms_list):
            raise(ValueError("structure_ids and structures should have the same length."))

        if structure_descriptions is None:
            structure_descriptions = [(f"{len(atoms)}-atom {atoms.get_chemical_formula()} structure "
                                      f"with ID {id}") for id, atoms in zip(structure_ids, atoms_list)] 
        elif len(structure_descriptions) != len(atoms_list):
            raise(ValueError("structure_descriptions and structures should have the same length."))

        # Initialize results dict
        results = {
            'sorting_property_weight': sorting_property_weight, 
        }

        if sorting_property_values is not None:
            if sorting_order.lower() in ['ascending', 'asc']:
                sorted_indexes = np.argsort(sorting_property_values)
            elif sorting_order.lower() in ['descending', 'desc']:
                sorted_indexes = np.argsort(-np.array(sorting_property_values))
            structure_indexes_by_sorting_property = np.argsort(sorted_indexes)
    
        struct_indexes = np.arange(len(atoms_list))
    
        # Process initial_selection:
        if isinstance(initial_selection, (list, tuple, np.ndarray)):
            selected_indexes = initial_selection
        elif isinstance(initial_selection, int):
            selected_indexes = [initial_selection]
        elif sorting_property_values and isinstance(
                initial_selection, str) and initial_selection.lower() == 'best':
            selected_indexes = sorted_indexes[:1]
        elif not initial_selection or (isinstance(initial_selection, str) 
                                       and initial_selection.lower() == 'random'):
            # Initialize sequence, save seed in results
            ssq = np.random.SeedSequence(seed)
            rng = np.random.default_rng(ssq)
            seed = ssq.entropy
            print('Random number generation:\nSeed = {}'.format(seed))
            # Make sure seed is returned
            results['seed'] = seed
            # Randomly pick one ID
            selected_indexes = [rng.choice(struct_indexes)]
        else:
            raise TypeError(f"initial_selection of type {type(initial_selection)} rather than "
                            f"allowed list, tuple, numpy ndarray, int or str types.")

        if distance_method in ['maximum_average', 'max_average', 'max_av']:
            dist_mthd_str = 'maximum average distance to all picked structures'
        elif distance_method in ['maximum_minimum', 'maxmin', 'max_min']:
            dist_mthd_str = 'maximum distance to closest picked structure'
        else:
            ValueError(f'{distance_method} not among allowed values (e.g. max_av)')

        if sorting_property_weight >= 0 and sorting_property_weight <= 1:
            w_p = sorting_property_weight
        else:
            raise ValueError(f"sorting_property_weight should be between 0 and 1.")

        # Make sure that selected IDs exist in self.IDs
        for index in selected_indexes:
                if index not in struct_indexes: 
                    raise ValueError(f"Selected index {index} is out of [0-{len(struct_indexes)}] range.")

        # Initialize lists in results based on selected indexes
        results['structure_indexes'] = list(selected_indexes)
        results["structure_ids"] = [structure_ids[i] for i in selected_indexes]
        results["structure_descriptions"] = [structure_descriptions[i] for i in selected_indexes]
        results['distance_contributions'] = [None] * len(selected_indexes)
        
        if sorting_property_values is not None:
            results[f"structure_indexes_by_{sorting_property_label}"
                    ] = structure_indexes_by_sorting_property[selected_indexes].tolist()
            sorting_property_values_ar = np.array(sorting_property_values, dtype="float")
            results[f'{sorting_property_label}_contributions'] = [None] * len(selected_indexes)
        else:
            sorting_property_values_ar = np.zeros(len(selected_indexes))
        
        def normalize(myarray):
            min_ar = np.min(myarray)
            max_ar = np.max(myarray)
            normalized_array = (myarray - min_ar) / (max_ar - min_ar)
            return normalized_array

        # Create sets of all, selected and remaining indexes
        all_indexes = set(struct_indexes)
        selected_indexes = set(selected_indexes)
        remaining_indexes = all_indexes - selected_indexes
        discarded_indexes = set()
        while 1:
            if len(selected_indexes) >= n_structures:
                self.print(f"The target number of selected structures ({len(selected_indexes)} out "
                           f"of {n_structures}) has been reached. exiting.")
                break
            
            if not len(remaining_indexes):
                self.print(f"Theere are no remaining structure. Exiting while loop with only "
                           f"{len(selected_indexes)} structures selected.")

            # Convert sets to arrays
            remaining_indexes_ar = np.array(list(remaining_indexes), dtype=int)
            selected_indexes_ar = np.array(list(selected_indexes), dtype=int)

            # Compute average (or min) distance to already-selected structures
            D = np.take(np.take(self.Dmatrix, remaining_indexes_ar, axis=0), 
                        selected_indexes_ar, axis=1)
            if distance_method in ['maximum_average', 'max_average', 'max_av']:
                dist_contrib = np.mean(D, axis=1)
            elif distance_method in ['maximum_minimum', 'maxmin', 'max_min']:
                dist_contrib = np.min(D, axis=1)
            else:
                ValueError(f'{distance_method} not among allowed values.')
            # Normalize over remaining structures
            dist_contrib = normalize(dist_contrib)
            
            # Calculate a fitness penalty normlized over remaining data
            prop_contrib = normalize(sorting_property_values_ar[remaining_indexes_ar])
            
            # Invert the contribution of the property depending on whether it should be
            # beneficial or detrimental
            if sorting_order.lower() in ['ascending', 'asc']:
                prop_contrib = 1 - prop_contrib

            # Find index maximizing average distance - fitness contribution
            # Term to maximize -> high w_p should favor low-fitness/energy structures
            best_index_in_remaining = np.argmax((1 - w_p) * dist_contrib + w_p * prop_contrib)
            best_index = remaining_indexes_ar[best_index_in_remaining]
            best_id = structure_ids[best_index]
            
            # Automatically discard a structure that would be identical to an already-selected 
            # structure within a given distance threshold.
            continue_while_loop = False
            for i in selected_indexes:
                d = self.Dmatrix[best_index, i]
                if d < distance_threshold:
                    self.print((f"The distance {d:.5f} between current best structure with index {best_index} "
                                f"and ID {best_id} and already-selected structure with index {i} and"
                                f"ID {id} is smaller than the threshlod distance {distance_threshold}. "
                                f"It will not be selected."), verb_th=1)
                    continue_while_loop = True
                    break
            
            if continue_while_loop:
                discarded_indexes.add(best_index)
                remaining_indexes.remove(best_index)
                continue
                
            results['structure_indexes'].append(best_index)
            results['structure_ids'].append(best_id)
            results["structure_descriptions"].append(structure_descriptions[best_index])
            results['distance_contributions'].append(dist_contrib[best_index_in_remaining]) 

            if sorting_property_values is not None:
                index_by_sorting_property = structure_indexes_by_sorting_property[best_index]
                results[f'structure_indexes_by_{sorting_property_label}'].append(index_by_sorting_property)
                results[f'{sorting_property_label}_contributions'].append(prop_contrib[best_index_in_remaining])
                self.print(f"Structure index {best_index} with ID {best_id}, ranked {index_by_sorting_property} "
                           f"by ({sorting_order}) {sorting_property_label} (with a {dist_mthd_str} of "
                           f"{dist_contrib[best_index_in_remaining]:.3f} and a normalized {sorting_property_label} "
                           f"contribution of {prop_contrib[best_index_in_remaining]:.3f}", 
                           verb_th=2)
            else:
                self.print(f"Structure index {best_index} with ID {best_id} with a {dist_mthd_str} of "
                           f"{dist_contrib[best_index_in_remaining]:.3f}", 
                           verb_th=2)
            
            selected_indexes.add(best_index)
            remaining_indexes.remove(best_index)

        # Compute average (or min) distance to all selected structures
        selected_indexes_ar = np.array(list(selected_indexes), dtype=int)
        D = np.take(np.take(self.Dmatrix, selected_indexes_ar, axis=0), 
                    selected_indexes_ar, axis=1)
        if distance_method.lower() in ['maximum_average', 'max_average', 'max_av']:
            dist_str = 'average_dist_to_other_selected'
            results[dist_str] = [np.mean(np.delete(D[i], i)) for i in range(D.shape[0])]
        elif distance_method.lower() in ['maximum_minimum', 'maxmin', 'max_min', 'max-min']:
            dist_str = 'average_dist_to_other_selected'
            results[dist_str] = [np.mean(np.delete(D[i], i)) for i in range(D.shape[0])]

        self.print(f"results = {results}", verb_th=2)
        results_df = DataFrame(results)
        self.print(f"distance-based structure selection is done:\n{results_df}", verb_th=1)

        # Compute MDS
        self.print("Computing multi-dimensional scaling (MDS)...")
        mds = MDS(n_components=2, dissimilarity='precomputed', 
                  metric=True, n_jobs=n_jobs)
        coords = mds.fit_transform(self.Dmatrix)

        structure_selection_statuses = []
        for i in range(len(atoms_list)):
            if i in selected_indexes:
                structure_selection_statuses.append('selected')
            elif i in discarded_indexes:
                structure_selection_statuses.append('discarded')
            else:
                structure_selection_statuses.append('unselected')

        # TODO: adapt in case no sorting property is specified
        # Create a DataFrame for Plotly
        df = DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'ID': structure_ids, 
            'description': structure_descriptions, 
            'is_selected': structure_selection_statuses
        })

        if not description:
            description_str = ""
        else:
            description_str += ""
        size_non_zero_add = 1e-6
        if sorting_property_values is not None:
            df[sorting_property_label] = sorting_property_values
            df[f"normalized_{sorting_property_label}"] = normalize(sorting_property_values) + size_non_zero_add
            df[f'index_by_{sorting_property_label}'] = structure_indexes_by_sorting_property
            size = f"normalized_{sorting_property_label}"
            title = (f"{description_str}MDS plot for {len(selected_indexes)} structures "
                     f"selected based on {dist_mthd_str}<br>"
                     f"with a {sorting_property_label} weight of {sorting_property_weight}.")
            hover_data = ['ID', 'description', sorting_property_label, 
                          f'index_by_{sorting_property_label}']
            marker_dict = dict(sizemode='diameter', sizeref=mds_plot_sizeref, 
                               sizemin=mds_plot_sizemin, opacity=mds_plot_opacity)
        else:
            size = None
            title = (f"{description_str}MDS plot for {len(selected_indexes)} structures "
                     f"selected based on {dist_mthd_str}<br>")
            hover_data = ['ID', 'description']
            marker_dict = dict(sizemode='diameter', size=mds_plot_fixed_size, 
                               opacity=mds_plot_opacity)

        # Plot with Plotly Express
        mds_plot_fig = px.scatter(
            df,
            x='x',
            y='y', 
            size=size,  # Control size by property
            color='is_selected',  # Color by selection status
            symbol='is_selected',  # Symbol by selection status
            title=title, 
            labels={'x': 'MDS Dimension 1', 'y': 'MDS Dimension 2'}, 
            hover_data=hover_data,  
            template='simple_white' 
        )

        # Customize symbol size range if needed
        mds_plot_fig.update_traces(
            marker=marker_dict,
            selector=dict(mode='markers')
        )
        
        self.print("Opening plotly (via browser)...")
        mds_plot_fig.show()

        df.to_csv('tmp.csv')

        return results, mds_plot_fig

# end of class distanceMatrixData


def get_distance_matrix_from_valle_oganov_dscribe(structures, species=None, 
                                                 function="distance",
                                                 sigma=0.1, n=100, r_cut=8.0, 
                                                 n_jobs=None, sparse=False, dtype="float32", 
                                                 distance_metric='cosine', return_plot=False, 
                                                 show_plot=False, 
                                                 structure_names=None, axesLabel:str='',
                                                 tickLabelsFontSize:int=0,
                                                 xticklabelsRotation=45, cmap=None,
                                                 verbosity=1, **vo_kwargs):
    """
    Get a distance matrix from a list of structures using the Valle Oganiv approach 
    as implemented in Dscribe
    
    
    Args:
        structures, 
        species: list or tuple (default is None)
            The chemical species as a list of atomic numbers or as a list of chemical symbols. 
            Notice that this is not the atomic numbers that are present for an individual system, 
            but should contain all the elements that are ever going to be encountered when 
            creating the descriptors for a set of systems. Keeping the number of chemical 
            species as low as possible is preferable.
            If None all species included in structures will be used.
        function: str (default is "distance")
            The geometry function The order (k=2 for "ditances" and k = 3 for "angles" tells 
            how many atoms are involved in the calculation and thus also heavily 
            influences the computational time.
        sigma: float (default is 0.1)
            Standard deviation of the gaussian broadening in Å.
        n: int (default: 100)
            Number of discretization points
        r_cut: float( default is 8.0)
            cutoff radius in Å.
        n_jobs: int: (default is None)
            number of processors used for the calculation of the distance matrix
            If None the number of processors will be obtained using the multiprocess module.
        distance_metric: str (default is 'cosine')
            Metric for the distance measurement. Default is a 0-1 cosine measurement.
            (see scipy pdist for other possibilities).
        return_plot: bool (default is False)
            Whether figure abd axes should be returned in addition to the distance matrix 
        show_plot: bool (default is False)
            Whether plot shall be shown automatically.
            If not it can be shown from the fig output with fig.show() 
        tickLabels: list, OPTIONAL
            default is [] in wich case structure IDs (if any) or structure
            indexes will be used.
        axesLabel: str, OPTIONAL
            Default is ''.
        tickLabelsFontSize: int, OPTIONAL
            will change font size of tick labels (e.g. set to 6 or less if
            matrix size is equal to 50).
        xticklabelsRotation: float or {'horizontal','vertical'}, OPTIONAL
            Based on matplotlib.text.Text rotation option. Default is 45.
        cmap: str, matplotlib.colors.LinearSegmentedColormap or None
            if None a red-gold-green map will be used. If 'default' the default
            matplotlib cmap (blue-green-yellow) will be used.
        verbosity: int (default is 1)
            Verbosity level
            
    Returns:
        distance_matrix: numpy.ndarray
            square matrix of dimension (N, N) where N = len(structures)
        fig: matplotlib.figure
            Figure handle (only if show_plot is True)
        ax: matplotlib.axes
            Axes handle
    """
    # Get a list of atoms and a list of pymatgen structures
    atoms_list = [get_ase_atoms(s) for s in structures]  
    
    if not n_jobs:
        n_jobs= cpu_count()

    # Define species
    if not species:
        species = list(set.intersection(*[set(atoms.get_chemical_symbols()) for atoms in atoms_list]))
        species.sort()
    
    vo = ValleOganov(species=species, function=function, sigma=sigma, 
                    n=n, r_cut=r_cut, **vo_kwargs)

    if verbosity >= 2:
        tic = perf_counter()
        print(f"Calculating Valle-Oganov descriptors (fingerprints) for {len(atoms_list)} "
              f"atomic structures using {n_jobs} CPUs...")

    # Calculate descriptors
    vo_descr_vect = vo.create(atoms_list, n_jobs=n_jobs)
    
    if verbosity >= 2:
        print(f"... took {perf_counter() - tic:.3f} s.")
    
    if verbosity >= 2:
        tic = perf_counter()
        print(f"Calculating distance matrix for {len(atoms_list)} atomic structures...")
    
    # Calculate distance matrix
    distance_matrix = squareform(pdist(vo_descr_vect, metric=distance_metric))
    
    if verbosity >= 2:
        print(f"... took {perf_counter() - tic:.3f} s.")
    
    if return_plot:
        pmg_structures = [get_pymatgen_structure(s) for s in structures]
        if structure_names is None:
            structure_names = ['{} # {}'.format(s.composition.reduced_formula, i)
                               for i, s in enumerate(pmg_structures)]
        fig, ax = plot_distance_matrix(distance_matrix, pmg_structures, structure_names,
                                       axesLabel=axesLabel, tickLabelsFontSize=tickLabelsFontSize,
                                       xticklabelsRotation=xticklabelsRotation, cmap=cmap)
        if show_plot:
            fig.show()
        
        return distance_matrix, fig, ax
    else:
        return distance_matrix


def get_distance_matrix_from_average_soap(structures, soap_weighting='poly',
                                          soap_n_max=8, soap_l_max=6, soap_r_cut=8.0,
                                          distance_metric='cosine', show_plot=False,
                                          structure_names=None, axesLabel:str='',
                                          tickLabelsFontSize:int=0,
                                          xticklabelsRotation=45, cmap=None,
                                          verbosity=1):
    """
    Get a distance matrix from a list of structures using the average SOAP kernel
    
    IMPORTANT: This measurement is very fast but of very poor sensivity.
        Method using the regularized entropy-matching kernel should 
        be preferred.

    Args:
        soap_wheighting: str, dict or None (default is 'poly')
            If 'poly', a polynomial weighting with a default set of parameters
            will be used.
            See weighting format in dscribe.descriptors.SOAP documentation.
            For polynomila wighting, use:
            soap_weithing={
                "function": "poly",
                "r0": 8.0,
                "c": 1,
                "m": 1}
        soap_r_cut: float (default is 8.0)
            SOAP cutoff radius. Will only be used if weighting is None.
        tickLabels: list, OPTIONAL
            default is [] in wich case structure IDs (if any) or structure
            indexes will be used.
        axesLabel: str, OPTIONAL
            Default is ''.
        tickLabelsFontSize: int, OPTIONAL
            will change font size of tick labels (e.g. set to 6 or less if
            matrix size is equal to 50).
        xticklabelsRotation: float or {'horizontal','vertical'}, OPTIONAL
            Based on matplotlib.text.Text rotation option. Default is 45.
        cmap: str, matplotlib.colors.LinearSegmentedColormap or None
            if None a red-gold-green map will be used. If 'default' the default
            matplotlib cmap (blue-green-yellow) will be used.
        verbosity: int (default is 1)
            Verbosity level
            
    Returns:
        distance_matrix: numpy.ndarray
            square matrix of dimension (N, N) where N = len(structures)
        fig: matplotlib.figure
            Figure handle (only if show_plot is True
        ax: matplotlib.axes
            Axes handle
    """

    # Define default weighting paremeters if relevant
    if isinstance(soap_weighting, str):
        if 'poly' in soap_weighting.lower():
            soap_weighting={"function": "poly", "r0": 8.0, "c": 1, "m": 1}
            # r_cut will not be used in this case
        # TODO: define other weighting methods here.
        else:
            raise ValueError('Unknown SOAP weighting method. See dscribe.descriptors.SOAP.')

    if verbosity >= 2:
        print('soap_weighting = {}'.format(soap_weighting))


    # Calculate average_sop for all individual structures
    # Calculate distance matrix with scipy.spatial.distance.pdist(X, metric='euclidean', *, out=None, **kwargs)

    # Define average_soap kernel
    species = set.intersection(*[set(structure.symbol_set) for structure in structures])
    soap_average = SOAP(species=species, average="inner", weighting=soap_weighting, n_max=soap_n_max,
                        l_max=soap_l_max, r_cut=soap_r_cut)
    aaa = AseAtomsAdaptor()
    av_soap_list = [soap_average.create(aaa.get_atoms(structure)) for structure in structures]
    av_soap_array = np.array(av_soap_list)

    distance_matrix = squareform(pdist(av_soap_array, metric=distance_metric))

    if show_plot:
        if structure_names is None:
            structure_names = ['{} # {}'.format(struct.composition.reduced_formula, i)
                               for i, struct in enumerate(structures)]
        fig, ax = plot_distance_matrix(distance_matrix, structures, structure_names,
                                       axesLabel=axesLabel, tickLabelsFontSize=tickLabelsFontSize,
                                       xticklabelsRotation=xticklabelsRotation, cmap=cmap)
        return distance_matrix, fig, ax
    else:
        return distance_matrix


def get_similarity_map_from_soap_rematchkernel(structures, 
                                               soap_weighting='poly', soap_n_max=5, soap_l_max=4, 
                                               soap_r_cut=6.0, soap_sigma=0.2, periodic=True, 
                                               verbosity=1):
    """
    Get a similarity map 
    
    Args:
        structures: list or tuple
            A list of structures
        soap_wheighting: str, dict or None (default is 'poly')
            If 'poly', a polynomial weighting with a default set of parameters
            will be used.
            See weighting format in dscribe.descriptors.SOAP documentation.
            For polynomila wighting, use:
            soap_weithing={
                "function": "poly",
                "r0": 8.0,
                "c": 1,
                "m": 1}
        soap_n_max: int (default is 5)
            ADD DEFINITION
        soap_l_max: int (default is 4)
            ADD DEFINITION
        soap_r_cut: float (default is 8.0)
            SOAP cutoff radius. Will only be used if weighting is None.
        tickLabels: list, OPTIONAL
            default is [] in wich case structure IDs (if any) or structure
            indexes will be used.
        axesLabel: str, OPTIONAL
            Default is ''.
        tickLabelsFontSize: int, OPTIONAL
            will change font size of tick labels (e.g. set to 6 or less if
            matrix size is equal to 50).
        xticklabelsRotation: float or {'horizontal','vertical'}, OPTIONAL
            Based on matplotlib.text.Text rotation option. Default is 45.
        cmap: str, matplotlib.colors.LinearSegmentedColormap or None
            if None a red-gold-green map will be used. If 'default' the default
            matplotlib cmap (blue-green-yellow) will be used.
        verbosity: int (default is 1)
            Verbosity level
            
    Returns:
        similarity_matrix: numpy.ndarray
            square matrix of dimension (N, N) where N = len(structures)
    """


    # Define default weighting paremeters if relevant
    if isinstance(soap_weighting, str):
        if 'poly' in soap_weighting.lower():
            soap_weighting={"function": "poly", "r0": 6.0, "c": 1, "m": 1}
            # r_cut will not be used in this case
        # TODO: define other weighting methods here.
        else:
            raise ValueError('Unknown SOAP weighting method. See dscribe.descriptors.SOAP.')

    if verbosity >= 2:
        print('soap_weighting = {}'.format(soap_weighting))

    aaa = AseAtomsAdaptor()
    
    # First we will have to create the features for atomic environments. Lets
    # use SOAP.
    species = set.intersection(*[set(structure.symbol_set) 
                                 for structure in structures])
    desc = SOAP(species=species, r_cut=soap_r_cut, n_max=soap_n_max, l_max=soap_l_max, 
                weighting=soap_weighting, sigma=soap_sigma, periodic=periodic, 
                compression={"mode":"off"}, sparse=False)
        
    features_list = []
    for i, structure in enumerate(structures):
        atoms = aaa.get_atoms(structure)
        features = normalize(desc.create(atoms))
        features_list.append(features)
        if verbosity >= 2:
            print('features has shape : {}'.format(features.shape))
        
    # Calculates the similarity with the REMatch kernel and a linear 
    # (or other) metric. result will be a full similarity matrix. 
    # Any metric supported by scikit-learn will work: e.g. a Gaussian.
    # re = REMatchKernel(metric="rbf", gamma=1, alpha=1, threshold=1e-6)    
    re = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
    similarity_matrix = re.create(features_list)
    
    if verbosity >= 2:
        print('Similarity matrix obtained with the regularized entropy matching Kernel:')
        print(similarity_matrix)
    
    return similarity_matrix


def get_distance_matrix_from_soap_rematchkernel(structures, 
        soap_weighting='poly', soap_n_max=5, soap_l_max=4, 
        soap_r_cut=6.0, soap_sigma=0.2, show_plot=False,
        periodic=True, structure_names=None, axesLabel:str='',
        tickLabelsFontSize:int=0, xticklabelsRotation=45, cmap=None,
        verbosity=1):
    """
    Get a distance matrix between structures with regularized 
    entropy matching kernel and SOAP
    
    TO BE COMPLETED.
    """
    similarity_matrix = get_similarity_map_from_soap_rematchkernel(structures, 
        soap_weighting=soap_weighting, soap_n_max=soap_n_max, 
        soap_l_max=soap_l_max, soap_r_cut=soap_r_cut, soap_sigma=soap_sigma,
        periodic=periodic, verbosity=verbosity)
    # TODO: deal with case where similarity_matrix = 1 (-> set distance to 0).
    distance_matrix = np.sqrt(2 - 2*similarity_matrix)
    
    if show_plot:
        if structure_names is None:
            structure_names = ['{} # {}'.format(struct.composition.reduced_formula, i)
                               for i, struct in enumerate(structures)]
        fig, ax = plot_distance_matrix(distance_matrix, structures, structure_names,
                                       axesLabel=axesLabel, tickLabelsFontSize=tickLabelsFontSize,
                                       xticklabelsRotation=xticklabelsRotation, cmap=cmap)
        return distance_matrix, fig, ax
    else:
        return distance_matrix


def plot_distance_matrix(distance_matrix=None, structures=None,
                         structure_names=None,
                         system_name=None, axesLabel:str='',
                         tickLabelsFontSize:int=0,
                         xticklabelsRotation=45, cmap=None):
    """
    Plot distance matrix

    Args:
        distance_matrix: numpy.darray or None
            Square matrix of distance between structures.
            If None a distance matrix will be calculated from SOAP descriptors
        structures: list or None
            Will be used to define names if system_name is None
        system_name: str, OPTIONAL
            If specified, will insert system_name + ' : ' before title
        tickLabels: list, OPTIONAL
            default is [] in wich case structure IDs (if any) or structure
            indexes will be used.
        axesLabel: str, OPTIONAL
            Default is ''.
        tickLabelsFontSize: int, OPTIONAL
            will change font size of tick labels (e.g. set to 6 or less if
            matrix size is equal to 50).
        xticklabelsRotation: float or {'horizontal','vertical'}, OPTIONAL
            Based on matplotlib.text.Text rotation option. Default is 45.
        cmap: str, matplotlib.colors.LinearSegmentedColormap or None
            if None a red-gold-green map will be used. If 'default' the default
            matplotlib cmap (blue-green-yellow) will be used.

    Returns:
        fig : figure.Figure
            distance matrix figure handle
    """

    if distance_matrix is None and structures is not None:
        distance_matrix = get_distance_matrix_from_average_soap(structures)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Distance matrix between structures'
    if isinstance(system_name, str):
        if len(system_name):
            title=system_name+': '+title

    nb_of_structures = distance_matrix.shape[0]

    if cmap is None:
        colors = [(1, 0, 0), (1, 0.9, 0), (0, 0.75, 0)]  # Red, Gold, Green
        cmap = LinearSegmentedColormap.from_list('RedGoldGreen', colors)
    elif cmap == 'default':
        cmap = None

    im = ax.imshow(distance_matrix, cmap=cmap)
    ax.set_aspect('equal')
    ax.set_title(title)

    if structure_names is None:
        if structures is not None:
            structure_names = ['{} # {}'.format(struct.composition.reduced_formula, i)
                               for i, struct in enumerate(structures)]
            if len(axesLabel) == 0:
                axesLabel = 'Structure'
        else:
            structure_names = [f'{i}' for i in range(nb_of_structures)]
            if len(axesLabel) == 0:
                axesLabel = 'Structure index'

    tickLabels = structure_names

    ax.set_xlabel(axesLabel)
    ax.set_ylabel(axesLabel)
    ax.set_xticks(0.5 + np.arange(nb_of_structures))
    ax.set_xticklabels(tickLabels, rotation=xticklabelsRotation,
                       va='top', ha='right')
    ax.set_yticks(0.5 + np.arange(nb_of_structures))
    ax.set_yticklabels(tickLabels, va='bottom', ha='right')
    if tickLabelsFontSize > 0 :
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(tickLabelsFontSize)
    plt.colorbar(im, ax=ax, orientation='vertical',label='Distance')

    return fig, ax


def get_partial_from_valle_oganov_dscribe(system, type_pair, species=None, 
                                          function='distance', n=100, 
                                          sigma=0.1, r_cut=8.0,  
                                          return_plot=False, show_plot=False, 
                                          use_reduced=False, 
                                          description=None, **vo_kwargs):
    
    if len(type_pair) != 2:
        raise ValueError('type_pair should be a list of lenght 2.')
    
    pair_name = type_pair[0]
    for elmt in type_pair[1:]:
        pair_name += f"-{elmt}"
        
    atoms = get_ase_atoms(system)
    
    if not species:
        species = list(set(atoms.get_chemical_symbols()))
        species.sort()
        
    vo = ValleOganov(species, function, n, sigma, r_cut, **vo_kwargs)    
    descriptor = vo.create(atoms)
    
    r = np.linspace(vo.grid["min"], vo.grid["max"], vo.grid["n"])
    rho_0 = len(atoms) / atoms.get_volume()
    
    g_AB = descriptor[vo.get_location(type_pair)]
    G_AB = 4 * np.pi * rho_0 * r * (g_AB - 1)
    
    partial = {
        'pair': list(type_pair), 
        'pair_name': pair_name, 
        'r': r, 
        'partial': g_AB, 
        'reduced_partial': G_AB, 
        'xlabel': 'r (Å)', 
        'ylabel': f"{pair_name} partial RDF", 
        'ylabel_reduced': f"{pair_name} reduced partial RDF", 
    }
    
    if return_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        y = partial['reduced_partial'] if use_reduced else partial['partial']
        ax.plot(partial['r'], y, label=f"{pair_name} partial")
        if not description:
            description = f"{atoms.get_chemical_formula()} {pair_name} partial pair distribution function"
        
        ylabel = partial['ylabel'] if use_reduced else partial['ylabel']
        ax.set(title=description, xlabel=partial['xlabel'], ylabel=ylabel)

        if show_plot:
            fig.show()

        return partial, fig, ax
        
    else:
        return partial
 

