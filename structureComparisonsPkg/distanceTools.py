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

from dscribe.descriptors import SOAP, ValleOganov
from dscribe.kernels import REMatchKernel

from sklearn.preprocessing import normalize

from pyama.utils import get_ase_atoms, get_pymatgen_structure


class distanceMatrixData() :
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
                 **kwargs) :
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
                                   rel_y_shift=0.25):
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

        if self.print_performance:
            tic_pmg_get_nbrs = perf_counter()
        all_nbrs = struct.get_all_neighbors(Rmax)
        if self.print_performance:
            tac_pmg_get_nbrs = perf_counter()

        len_R = len(R)
        all_partials = np.zeros((len(types), len(types), len(R)))
        for type_ind_A, type_A in enumerate(types):
            A_indexes = struct.indices_from_symbol(type_A)
            N_A = nb_of_atoms_of_type[type_ind_A]
            for A_index in A_indexes:
                for type_ind_B, type_B in enumerate(types[:type_ind_A+1]):
                    N_B = nb_of_atoms_of_type[type_ind_B]
                    B_nbr_indexes = [i for i, nbr in enumerate(all_nbrs[A_index])
                                     if nbr.species.elements[0].name == type_B]

                    R_ij = [all_nbrs[A_index][B_nbr_index].nn_distance for
                            B_nbr_index in B_nbr_indexes]
                    len_R_ij = len(R_ij)
                    R_ij_tab = np.matmul(np.reshape(R_ij,(len_R_ij,1)) ,
                                         np.ones((1,len_R)))
                    R_tab = np.matmul(np.ones((len_R_ij,1)) ,
                                      np.reshape(R,(1,len_R)))
                    delta_ij = (Rmax/(sigma*np.sqrt(2*np.pi)*len_R))*np.exp(
                        -0.5*np.square((R_tab-R_ij_tab)/sigma))
                    g_AB_ij = np.divide(delta_ij , 4*np.pi*np.square(R_ij_tab)*
                                        N_A*N_B*DELTA/V)
                    all_partials[type_ind_A, type_ind_B, :] = np.add(
                        all_partials[type_ind_A, type_ind_B, :],
                        np.sum(g_AB_ij,axis=0))

        # Symmetrize all_partials
        for type_ind_A, type_A in enumerate(types):
            for type_B in types[type_ind_A+1:]:
                type_ind_B = types.index(type_B)
                all_partials[type_ind_A, type_ind_B, :] = all_partials[
                    type_ind_B, type_ind_A, :]

        if self.print_performance:
            print(('Execution of function calculate_all_partial_RDFs for {} took '
                   '{:.1f} ms (including {:.1f} ms for pymatgen '
                   'get_all_neighbors()).').format(structure.formula,
                  1000*(perf_counter()-tic),
                  1000*(tac_pmg_get_nbrs-tic_pmg_get_nbrs)))

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


    def calculate_partial_RDF(self,structure,specieA,specieB,**kwargs) :
        """
        Calculate pairwise fingerprint function as defined in Oganov A.R. and Valle M. J. Chem. Phys. 130(10), 2009

        Implementation based on equation (3) in :
        Oganov, Artem R., et Mario Valle. « How to Quantify Energy Landscapes of Solids ». The Journal of Chemical Physics 130, nᵒ 10 (14 mars 2009): 104504. https://doi.org/10.1063/1.3079326.

        Should only be used in the case where ony one specific partial RDF
        is needed. Even when only 2 types of species are present in the system
        it is much more efficient to use calculate_all_partial_RDFs (or
        calculate_all_reduced_partial_RDFs).

        args :
            - structure : object of pymatgen.core.structure Structure class
            - speciesA : first species name as a string (eg. 'Al', 'N', etc.)
            - speciesB : second species name as a string (eg. 'Al', 'N', etc.)
            - sigma : Gaussian smearing factor (standard deviation)
            - Rmax : cutoff radius for neighbor search and maximum value of array of R values (starting at 0)
            - Rsteps : number of bins for R discretization

        Optional arguments :
            - showPlot : (default = False)
            - R : R array (should be ordered with constant step)
        returns :
            g_AB(R) where b_AB(R) is the partial RDF between (A,B) species
            calculated for the vector of radial distances R.
        """

        print('Calculating {}-{} partial RDF.'.format(specieA, specieB))

        tic = perf_counter()
        intermediate_tics = []
        performance = {}
        perf_steps = ['reduced_struct', 'get_neighbors', 'R_ij', 'R_ij_tab', 'R_tab',
                      'delta_ij', 'g_AB_ij', 'g_AB']
        for s in perf_steps:
            performance[s] = []

        # import copy

        R = self.R
        sigma = self.sigma
        Rmax = self.Rmax

        DELTA = R[1]-R[0] # discretization step

        # Parameters to get from structure
        V = structure.volume # cell volume

        # TODO: Create a copy of the structure containing only A and B atomic types (only A in case where B = A)

        # AInCellIndexes = [index for index,site in enumerate(structure.sites) if specieA in site.specie.symbol]
        # Can be obtained directly with structure.indices_from_symbol : structure.indices_from_symbol('As'))

        tic2 = perf_counter()
        reduced_struct = IStructure.from_sites([
            structure.sites[index] for index in
            set(structure.indices_from_symbol(specieA)) |
            set(structure.indices_from_symbol(specieB))])

        performance['reduced_struct'] = [perf_counter() - tic2]

        AInCellIndexes = list(reduced_struct.indices_from_symbol(specieA))
        AIndexesSet = set(AInCellIndexes)
        N_A = len(AInCellIndexes)

        if specieA != specieB:
            BInCellIndexes = list(reduced_struct.indices_from_symbol(specieB))
            BIndexesSet = set(BInCellIndexes)
            N_B = len(BInCellIndexes)
        else:
            N_B = N_A
            BInCellIndexes = AInCellIndexes
            BIndexesSet = AIndexesSet

        g_AB = np.zeros((len(R),))

        tic2 = perf_counter()
        all_nbrs = reduced_struct.get_all_neighbors(Rmax)
        performance['get_neighbors'] = [perf_counter() - tic2]

        intermediate_tics = [perf_counter()]
        # Pick sites of type A

        g_AB = np.zeros((len(R),))
        for index, nbrs in enumerate(all_nbrs):
            if reduced_struct.sites[index].species.elements[0].name == specieA:
                # Calculate R_ij distances and reshape R_ij and R as tables of dimension(len(R_ij),len(R))
                intermediate_tics.append(perf_counter())
                R_ij = [nbr.nn_distance for nbr in nbrs if
                        nbr.species.elements[0].name == specieB]
                intermediate_tics.append(perf_counter())
                R_ij_tab = np.matmul(np.reshape(R_ij,(len(R_ij),1)) ,
                                     np.ones((1,len(R))) )
                intermediate_tics.append(perf_counter())
                R_tab = np.matmul( np.ones((len(R_ij),1)) ,
                                  np.reshape(R,(1,len(R))) )
                intermediate_tics.append(perf_counter())
                # The Gaussian smearing normalization takes R into account to ensure that sum(delta_ij)~=1
                delta_ij = (Rmax/(sigma*np.sqrt(2*np.pi)*len(R)))*np.exp(
                    -0.5*np.square((R_tab-R_ij_tab)/sigma))
                intermediate_tics.append(perf_counter())
                g_AB_ij = np.divide(delta_ij , 4*np.pi*np.square(R_ij_tab)*
                                    N_A*N_B*DELTA/V )
                intermediate_tics.append(perf_counter())
                g_AB = np.add(g_AB,np.sum(g_AB_ij,axis=0))
                intermediate_tics.append(perf_counter())
                for i, s in enumerate(perf_steps[2:]):
                    performance[s].append(intermediate_tics[i+1]-intermediate_tics[i])

        if self.print_performance:
            print(('Execution of function calculate_partial_RDF for {}-{} took '
                   '{:.3f} ms.').format(specieA, specieB,
                                        1000*(perf_counter()-tic)))
            for k, v in performance.items():
                print('Calculations of {} took {:.3f} ms.'.format(k,
                                                                  1000*sum(v)))

        if 'showPlot' in kwargs :
            try :
                if kwargs['showPlot'] == True :
                    fig, ax = plt.subplots()
                    ax.plot(R,g_AB)
                    ax.set(xlabel='R (Angstroms)',
                            ylabel='g_'+specieA+'_'+specieB+'(R)',
                            title='Partial RDF g_'+specieA+'_'+specieB+'(R)')
                    plt.show()
            except :
                raise ValueError('showPlot should be a True/False boolean.')
        return g_AB


    def calculate_reduced_partial_RDF(self,structure,specieA,specieB,**kwargs) :
        """
        Calculate pairwise partial reduced radial distribution function G_AB(r)

        Reduced partial RDF function G_AB(r) is obtained from parial RDF
        g_AB(r) using the expression :
            G_AB(r) = 4*pi*rho_0*r*(g_AB(r)-1)
        with rho_0 the atomic density (??? UNIT ???)

        args :
            - structure : object of pymatgen.core.structure Structure class
            - speciesA : first species name as a string (eg. 'Al', 'N', etc.)
            - speciesB : second species name as a string (eg. 'Al', 'N', etc.)

        Optional arguments :
            - sigma : Gaussian smearing factor (standard deviation)
            - R : R array (should be ordered with constant step)
            - Rmax : cutoff radius for neighbor search and maximum value of array of R values (starting at 0)
            - Rsteps : number of bins for R discretization
            - showPlots : (default = False)

        returns :
            G_AB(R),R where G_AB(R) is the partial RDF between (A,B) species
            calculated for the vector of radial distances R.
        """
        # TODO : retrieve system density in the appropriate unit
        rho_0 = structure.num_sites/structure.volume  # Number of atoms/Angstrom^3
        g_AB = self.calculate_partial_RDF(structure,specieA,specieB,**kwargs)
        G_AB = 4*np.pi*rho_0*self.R*(g_AB-1)
        return G_AB

    def calculate_fingerprint_AB_component(self,structure,specieA,specieB,**kwargs) :
        """
        Calculate pairwise fingerprint function as defined in Oganov A.R. and Valle M. J. Chem. Phys. 130(10), 2009

        For a definition see equation (3) in :
        Oganov, Artem R., et Mario Valle. « How to Quantify Energy Landscapes
        of Solids ». The Journal of Chemical Physics 130, nᵒ 10 (14 mars 2009):
        104504. https://doi.org/10.1063/1.3079326.

        F_AB = g_AB -1
        where g_AB is the partial radial distribution function (RDF)

        args :
            - structure : object of pymatgen.core.structure Structure class
            - speciesA : first species name as a string (eg. 'Al', 'N', etc.)
            - speciesB : second species name as a string (eg. 'Al', 'N', etc.)
            - sigma : Gaussian smearing factor (standard deviation)
            - Rmax : cutoff radius for neighbor search and maximum value of array of R values (starting at 0)
            - Rsteps : number of bins for R discretization

        Optional arguments :
            - showPlots : (default = False)
            - R : R array (should be ordered with constant step)
        returns :
            F_AB(R),R where F_AB(R) is the (A,B) element of the fingerprint matrix
        """

        showPlot = False
        if 'showPlot' in kwargs :
            try :
                if kwargs['showPlot'] == True :
                    showPlot = True
                    # Setting kwargs['showPlot'] to false to avoid plotting g_AB
                    kwargs['showPlot'] = False

            except :
                raise ValueError('showPlot should be a True/False boolean.')

        if self.fingerprintMethod == 'OganovValle2009' :
            F_AB = self.calculate_partial_RDF(structure,specieA,specieB,**kwargs) - 1.0

        # TODO : insert other types of fingerprint calcuations here
        # elif self.fingerprintMethod == 'MEHODNAME' :
        # CALCULATE F_AB

        if showPlot == True :
            fig, ax = plt.subplots()
            ax.plot(self.R,F_AB)
            ax.set(xlabel='R (Angstroms)',
                    ylabel='F_'+specieA+'_'+specieB+'(R)',
                    title ='Fingerprint F_'+specieA+'_'+specieB+'(R) based on '+self.fingerprintMethod)
            plt.show()

        return F_AB


    def calculate_cosine_distance(self,structure1,structure2, systemName:int='',
                                  showPlot=False, **kwargs):
        """
        Calculate "distance" between two structures (i.e. 1-similarity) of
        identical compositions

        If self.distanceMethod = 'cosine', the method uses the cosine distance
        matrix as defined in Oganov, A.R., and Valle M., J. Chem. Phys. 2009,
        130 (10), 104504. (https://doi.org/10.1063/1.3079326).

        Implementation using calculate_all_partial_RDFs(), based on pymatgen
        get_all_neighbors() method. This is more efficient than using
        calculate_partial_RDF() separately on all pairs of atoms.

        Args :
            - structure1 : pymatgen.core.structure Structure object
            - structure2 : pymatgen.core.structure Structure object
            - showPlot: bool (default is False)

        Returns:
            float: distance between 0 (identical structures) and 1 (infinitely-
            different structures.
        """

        # TODO: extend the use of the method to systems having identical atom types
        # with different proportions (use set()
        # Get compositions using symple elements (e.g. 'Al') as opposed to species (e.g. 'Al3+')
        """
        compo1 = structure1.composition.remove_charges()
        compo2 = structure2.composition.remove_charges()
        if not compo1.reduced_formula == compo2.reduced_formula :
            print('Two structures used in calculate_cosine_distance have different reduced compositions. Distance set to 1')
            return 1.0
        """

        (partials_1, types_1) = self.calculate_all_partial_RDFs(structure1)
        (partials_2, types_2) = self.calculate_all_partial_RDFs(structure2)
        # structures have previously been ordered
        if types_1 != types_2:
            print('Two structures used in calculate_cosine_distance have different'
                  'compositions. Distance set to 1')
            return 1.0

        if self.print_performance:
            tic = perf_counter()
        listOfAtomTypes = types_1
        nbOfAtomTypes = len(listOfAtomTypes)
        compo1 = structure1.composition.remove_charges()
        compo2 = structure2.composition.remove_charges()

        if self.distanceMethod == 'cosine' :
            # Calculate fingerprint component weights (must be the same for both structures)
            w_AB = np.zeros((nbOfAtomTypes,nbOfAtomTypes))
            sumOfNANB = 0
            for AIndex,AatomType in enumerate(listOfAtomTypes):
                for BIndex,BatomType in enumerate(listOfAtomTypes[:AIndex+1]) :
                    w_AB[AIndex][BIndex] = compo1[AatomType]*compo1[BatomType]
                    if AIndex != BIndex:
                        w_AB[BIndex][AIndex] = w_AB[AIndex][BIndex]
                    sumOfNANB += compo1[AatomType]*compo1[BatomType]
            w_AB /= sumOfNANB

            # This may use a lof of memory. NOt sure it needs to be stored.
            term1 = 0.0
            term2 = 0.0
            term3 = 0.0
            # F1 = np.zeros((nbOfAtomTypes,nbOfAtomTypes,len(R)))
            # F2 = np.zeros((nbOfAtomTypes,nbOfAtomTypes,len(R)))

            if showPlot == True :
                shift = 0.0
                legend=[]
                fig, ax = plt.subplots()

            for AIndex,AatomType in enumerate(listOfAtomTypes) :
                for BatomType in listOfAtomTypes[:AIndex+1]:
                    BIndex = listOfAtomTypes.index(BatomType)
                    F1_AB = partials_1[AIndex, BIndex, :] - 1.0
                    F2_AB = partials_2[AIndex, BIndex, :] - 1.0

                    if showPlot == True :
                        ax.plot(self.R,F1_AB+shift)
                        legend.append('F1_'+AatomType+'_'+BatomType)
                        ax.plot(self.R,F2_AB+shift)
                        legend.append('F2_'+AatomType+'_'+BatomType)
                        shift += 10.0

                    if BIndex == AIndex :
                        term1 += np.sum(F1_AB*F2_AB*w_AB[AIndex][BIndex])
                        term2 += np.sum(np.square(F1_AB)*w_AB[AIndex][BIndex])
                        term3 += np.sum(np.square(F2_AB)*w_AB[AIndex][BIndex])
                    elif BIndex < AIndex:
                        term1 += 2*np.sum(F1_AB*F2_AB*w_AB[AIndex][BIndex])
                        term2 += 2*np.sum(np.square(F1_AB)*w_AB[AIndex][BIndex])
                        term3 += 2*np.sum(np.square(F2_AB)*w_AB[AIndex][BIndex])
            distance = 0.5*(1-term1/(np.sqrt(term2)*np.sqrt(term3)))

        if self.print_performance:
            print(('Calculation of distance between structures took {:.1f} ms'
                   ).format(1000 * (perf_counter() - tic)))

        # TODO : insert here other definitions of distances between structures
        # elif self.distanceMethod == 'DISTANCE_METHOD_NAME'
            # ENTER DISTANCE CALCULATION HERE

        if showPlot == True :
            ax.set(xlabel='R (Angstroms)', ylabel='Intensity')
            title = 'distance = '+str(distance)+' ; fingerprint components'
            if systemName != '' :
                title=systemName+' : '+title
            ax.set_title(title)
            ax.legend(legend)
            plt.show()

        return distance

    def set_stucture_ids(self, structures,structureIDs=[]):
        """
        TO BE COMPLETED
        """
        if len(structureIDs) == 0:
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


    def calculate_distance_matrix(self,structures,structureIDs=[],
                                  systemName:str='',
                                  show_plot=False, **kwargs) :
        """
        Calculate a matrix of distances among a list of structures

        In default class settings the method uses cosine distances (between 0 and 1) between structure fingerprints as defined in Oganov A.R. and Valle M. J. Chem. Phys. 2009, 130(10), 104504 (https://doi.org/10.1063/1.3079326.), equations (3), (6b) and (7).
        Other implementations may be inserted in the future.

        Args :
        - structures : list of pymatgen.core.structure Structure objects.
        - structureIDs : list or array of IDs associated with the different
          structures. May by used to connect them to a database or
          uspexStructuresData objects.
        - systemName : str, OPTIONAL
            Will add systemName+' : ' at beginning of title if different from ''
        - show_plot : bool, OPTIONAL


        Optional keyword arguments :
        - saveToFile : (default = False) = True : saving matrix and associated data to file self.distMatrixDataSaveFile
        - saveFileName : saving matrix and associated data to file saveFileName

        Returns :
            - fig, ax if show_plot
            - self with distance matrix stored as self.distMatrix and structure IDs as self.IDs
        """

        self.set_stucture_ids(structures=structures, structureIDs=structureIDs)

        self.Dmatrix = np.zeros((len(self.IDs),len(self.IDs)))
        for index1,ID1 in enumerate(self.IDs) :
            for index2,ID2 in enumerate(self.IDs) :
                if ID2 == ID1 :
                    self.Dmatrix[index1][index2] = 0
                elif ID2 < ID1 :
                    self.Dmatrix[index1][index2] = self.calculate_cosine_distance(
                        structures[index1],structures[index2],systemName=systemName)
                    self.Dmatrix[index2][index1] = self.Dmatrix[index1][index2]

        if 'saveFileName' in kwargs :
            saveFileName = kwargs['saveFileName']
        else :
            saveFileName = self.saveFileName

        if 'saveToFile' in kwargs :
            if kwargs['saveToFile'] == True :
                try :
                    with open(saveFileName, 'wb') as f:
                        pickle.dump(self, f)
                    self.saveFileName = saveFileName
                except :
                    raise ValueError

        if show_plot:
            fig, ax = self.plot_distance_matrix(systemName=systemName)
            return fig, ax

    # End of method calculate_distance_matrix

    def plot_distance_matrix_old(self,systemName:str='',tickLabels:list=[],
                                 axesLabel:str='',tickLabelsFontSize:int=0,
                                 xticklabelsRotation=45, cmap=None) :
        """
        Plot distance matrix (OLD METHOD, DEPRECATED)

        Args:
            systemName: str, OPTIONAL
                If specified, will insert systemName+' : ' before title
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = 'Cosine distance matrix between structures'
        if systemName != '' :
            title=systemName+': '+title


        if cmap is None:
            colors = [(1, 0, 0), (1, 0.9, 0), (0, 0.75, 0)]  # Red, Gold, Green
            cmap = LinearSegmentedColormap.from_list('RedGoldGreen', colors)
        elif cmap == 'default':
            cmap = None

        im = ax.imshow(self.Dmatrix, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_title(title)
        if len(tickLabels) == 0 or len(tickLabels) != len(self.IDs):
            if len(self.descriptions) == len(self.IDs):
                tickLabels = self.descriptions
                if len(axesLabel) == 0:
                    axesLabel = 'Structure descriptions'
            else :
                tickLabels = [str(ID) for ID in self.IDs]
                if len(axesLabel) == 0:
                    axesLabel = 'Structure ID'

        ax.set_xlabel(axesLabel)
        ax.set_ylabel(axesLabel)
        ax.set_xticks(0.5+np.arange(len(self.IDs)))
        ax.set_xticklabels(tickLabels,rotation=xticklabelsRotation,va='top',ha='right')
        ax.set_yticks(0.5+np.arange(len(self.IDs)))
        ax.set_yticklabels(tickLabels,va='bottom',ha='right')
        if tickLabelsFontSize > 0 :
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(tickLabelsFontSize)
        plt.colorbar(im, ax=ax, orientation='vertical',label='D_cosine')

        return fig, ax

    # End of classmethod plot_distance_matrix


    def plot_distance_matrix(self,systemName:str='',tickLabels:list=[],
                             axesLabel:str='',tickLabelsFontSize:int=0,
                             xticklabelsRotation=45, cmap=None) :
        """
        Plot distance matrix

        Args:
            systemName: str, OPTIONAL
                If specified, will insert systemName+' : ' before title
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
        title = 'Distance matrix between structures'
        if systemName != '' :
            title = systemName + ': ' + title

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
            structure_names=tickLabels, system_name=systemName,
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


# end of class distanceMatrixData


def get_distance_matrix_from_valle_oganov_dscribe(structures, species=None, 
                                                 function="distance",
                                                 sigma=0.1, n=100, r_cut=8.0, 
                                                 n_jobs=1, sparse=False, dtype="float32", 
                                                 distance_metric='cosine', show_plot=False,
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
        n_jobs: int: (default is 1)
            number of processors used for the calculation of the distance matrix
        distance_metric: str (default is 'cosine')
            Metric for the distance measurement. Default is a 0-1 cosine measurement.
            (see scipy pdist for other possibilities).
        sparse: bool (defalut is False)
            Whether result should be returned as a numpy sparse matrix.
        dtype: str (default is "float64")
            Numpy dtype of output distance matrix
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
    
    # Define species
    if not species:
        species = list(set.intersection(*[set(atoms.get_chemical_symbols()) for atoms in atoms_list]))
        species.sort()
        
    vo = ValleOganov(species=species, function=function, sigma=sigma, 
                    n=n, r_cut=r_cut, **vo_kwargs)

    if verbosity >= 2:
        tic = perf_counter()
        print(f"Calculating Valle-Oganov descriptors (fingerprints) for {len(atoms_list)} "
              f"atomic structures...")
    
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
    
    if show_plot:
        pmg_structures = [get_pymatgen_structure(s) for s in structures]
        if structure_names is None:
            structure_names = ['{} # {}'.format(s.composition.reduced_formula, i)
                               for i, s in enumerate(pmg_structures)]
        fig, ax = plot_distance_matrix(distance_matrix, pmg_structures, structure_names,
                                       axesLabel=axesLabel, tickLabelsFontSize=tickLabelsFontSize,
                                       xticklabelsRotation=xticklabelsRotation, cmap=cmap)
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
                                          show_plot=True, use_reduced=False, 
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
    
    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        y = partial['reduced_partial'] if use_reduced else partial['partial']
        ax.plot(partial['r'], y, label=f"{pair_name} partial")
        if not description:
            description = f"{atoms.get_chemical_formula()} {pair_name} partial pair distribution function"
        
        ylabel = partial['ylabel'] if use_reduced else partial['ylabel']
        ax.set(title=description, xlabel=partial['xlabel'], ylabel=ylabel)
        
        return partial, fig, ax
        
    else:
        return partial

    
    
    
    
    
    


