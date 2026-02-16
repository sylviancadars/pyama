"""
uspexDataManipulation Module

Sylvian Cadars, Assil Bouzid
Institute of Research on Ceramics (IRCER), University of Limoges, CNRS, France
sylvian.cadars@unilim.fr

Open a set of lowest-energy structures structures from a USPEX run and measure their mutual distances (i.e. similarity) using different method :
 - structure fingerprints and cosine distances as defined in Oganov AR et al., J. Chem. Phys. 130, 104504 (2009)
 - structure fingerprint and distances using pymatgen local_env tools as detailed in : https://wiki.materialsproject.org/Structure_Similarity
Compare results obtained with both methods

2D-crystal (or thin film) runs (-200) may now be processed with a more robust and adaptative parsing of the Individuals output file summarizing the amain information on explored structures. In addition to (static) properties such as IDs, enthalpies, etc., a new property that can accomodate any type of metadata found in the Individuals file is included: individuals_dict.

This new parsing is new implemented in the function parse_individuals_file, which may also be called externally (without instanciating  uspexStructuresData class):

from pyama.uspexAnalysesPkg.uspexDataManipulation import parse_individuals_file
"""

import os
import numpy as np
from scipy.stats import rankdata
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from ase.io import read
import matplotlib.pyplot as plt
import pickle
import structureComparisonsPkg.distanceTools as dt
import json
from warnings import warn
from pandas import DataFrame
import plotly.express as px
from sklearn.manifold import MDS
from multiprocessing import cpu_count

def parse_individuals_file(file_name, verbosity=1, skip_magmoment_type=True,
                           na_value = 1.0e6):
    """
    Parse the Individuals output file of USPEX

    Tested for USPEX version 10.4 on fixed-composition bulk (calculationType 300)
    and "2D-crystal" or thin-film calculations (calculationType -200)

    Parsed data are recovered in the form of a dictionary (which can then be
    passed into a pandas dataframe, for example).

    Args:
        file_name: str
            path to the Individuals file to be parsed
        verbosity: int (default is 1)
            verbosity level
        skip_magmoment_type: bool (default is True)
            Should be set to True unless  magnetic moments are used in the calculation.
            This is critical for "2D-crystal" or thin-film calculations (calculationType -200)
            because (at least in USPEX v. 10.4) this property appears in the head of the
            Individual files but no corresponding values are parsed.

    Returns:
        individuals_dict: dict
            Dictionnary with keys corresponding to the properties listed in
            USPEX Individuals output file and values will be lists or
            numpy arrays containing the corresponding property values
            for each structure. Each list of array will have the number
            of structures as first dimension length.
    """
    # TODO: define default value in case of N/A is found in Individuals file
    # currently we use na_value
    if verbosity >= 1:
        print('Parsing data from file : ', file_name)
    with open(file_name, 'r') as f:
        lines = f.readlines()

    nb_of_header_lines = 2

    property_names = lines[0].split()

    # TODO: adapt in the case of magnetic calculations
    if skip_magmoment_type and 'Magmoment-Type' in property_names:
        property_names.remove('Magmoment-Type')

    if verbosity >= 2:
        print('property_names = {}'.format(property_names))

    individuals_dict = {prop: [] for prop in property_names}
    # TODO: read units ?
    individuals_dict['nb_of_structures'] = len(lines)-nb_of_header_lines
    if verbosity >= 2:
        print('nb_of_structures = ', individuals_dict['nb_of_structures'])

    for struct_index, line in enumerate(lines[nb_of_header_lines:]):
        values = []
        for str1 in line.split('['):
            list1 = str1.strip().split(']')
            if len(list1) > 1:
                values.append(list1[0].strip().split())
                [values.append(str2) for str2 in list1[1].strip().split()]
            else:
                [values.append(str2) for str2 in list1[0].strip().split()]

        if verbosity >= 2 and not struct_index:
            print('values = {}'.format(values))

        if len(values) != len(property_names):
            raise ValueError(('Unexpected number of parsed values in file {}, line {}: {}.'.format(
                file_name, struct_index + nb_of_header_lines, line)))

        for key, val in zip(property_names, values):
            individuals_dict[key].append(val)


    if verbosity >= 3:
        print('individuals_dict = {}', individuals_dict)

    if 'N/A' in individuals_dict['Fitness']:
        # Cure N/A fitnesses in case all but N/A values are equal to Enthalpy values
        # (the default in fixed-composition runs)
        if all([individuals_dict['Fitness'][i] == individuals_dict['Enthalpy'][i]
                for i, v in enumerate(individuals_dict['Fitness']) if v != 'N/A']):
            individuals_dict['Fitness'] = individuals_dict['Enthalpy']
        # TODO: Automatically-correct fitness for variable-composition runs

    # Convert to appropriate formats:
    for prop in property_names:
        if prop in ['Gen', 'ID', 'SYMM']:
            individuals_dict[prop] = [int(val) for val in individuals_dict[prop]]
        if prop in ['Enthalpy', 'Volume', 'Density', 'Thickness', 'Surf_Area', 'Spec_surf_area',
                    'Fitness', 'Q_entr', 'A_order', 'S_order']:
            for index, val in enumerate(individuals_dict[prop]):
                if val != 'N/A':
                    individuals_dict[prop][index] = float(val)
                else:
                    individuals_dict[prop][index] = na_value
            # or directly convert in numpy array ?
        if prop in ['Composition', 'KPOINTS']:
            individuals_dict[prop] = [[int(val) for val in _list] for _list in individuals_dict[prop]]

    return individuals_dict


class uspexStructuresData():
    """
    Extract and analyze structures produced by a USPEX crystal structure prediction run
    """
    def __init__(self, fileOrDirectoryName, uspex_run_name=None, 
                 selectedStructures = 'all',
                 extractedPOSCARBaseName='ID-', extractedPOSCARExtention='.vasp',
                 r=None, sigma=0.02, r_max=8.0, r_steps=200, 
                 verbosity=1):
        """
        Calss initialization

        Args:
            fileOrDirectoryName: 'str'
                Path to the resultX USPEX directory or Individuals file therein
            uspex_run_name: str or None (default is None)
                Name designating the input file. If None, the basename of the 
                folder containing the resultsX folder (and the inputs) will be used.
            selectedStructures: str (default is 'all')
                Whether all or certain structure (e.g. 'good_structures') shall be
                considered. NOT YET IMPLEMENTED.
            extractedPOSCARBaseName: str (default is 'ID-')
                Base name of the exported (VASP-POSCAR format) files that
                will contain the atomic coordinates of explored structures
            extractedPOSCARExtention: str (default is '.vasp')
                Base name of the exported (VASP-POSCAR format) files that
                will contain the atomic coordinates of explored structures.
            r: list or numpy ndarray (default is None)
                Vector of radial distances (in Å) used to measure distances between structures.
            sigma: float (default is 0.02)
                Gaussian broadening (in Å) used in partial radial distribution functions to 
                measure distances between structures.
            r_max: float (default is 8.0)
                Maximum radial distance (in Å) used in partial radial distribution functions to 
                measure distances between structures.
            r_steps: int (default is 200)
                Number of points in partial radial distribution functions used to 
                measure distances between structures.
            verbosity: int (default is 1)
                verbosity level from 0 to 3 (hard-core debugging).
                
        Returns:

        """
        self.verbosity = verbosity

        # Initialization of distance matrix data
        self.distance_data  = dt.distanceMatrixData(R=r, sigma=sigma,
            Rmax=r_max, Rsteps=r_steps)

        self.extractedPOSCARBaseName =  extractedPOSCARBaseName
        self.extractedPOSCARExtention = extractedPOSCARExtention

        if selectedStructures == 'all' :
            self.load_uspex_structure_data(fileOrDirectoryName)
        elif selectedStructures == 'best' :
            print('selectedStructures = ', selectedStructures)
            #TODO : read only best structures from file
        else :
            errorMsg = 'Invalid input parameter selectedStructures = ' + selectedStructures + ' for class uspexStructuresData.'
            raise ValueError(errorMsg)

        if not uspex_run_name:
            self.uspex_run_name = self.resultsDirectoryName.split(os.path.sep)[-2]
        else:
            self.uspex_run_name = uspex_run_name

        self.print("Extracting POSCAR files and loading pymatgen structures.")
        
        self.pmg_structures = self.get_structures_from_IDs(self.IDs, extract_poscar_files=True,
                                                           use_initial=False)
        
        self.print("Loading ASE structures from extracted POSCAR files.")
        self.set_ase_atoms_list()

        # initilize distance data with provided paremeters
        self.set_distance_parameters(r=r, sigma=sigma, r_max=r_max, r_steps=r_steps)
        self.distance_matrix = None  # To be defined when necessary

        # End of initialization function

    def as_dict(self):
        """
        Write most class instance attributes as a dict ready for export as json

        numpy arrays will be converted to lists.
        distance_matrix attribute contained np.nan values which will
        be converted to 'NaN' upon json dump.
        conversion back to np array with np.array(json.loads(json_list))
        will convert these to float('nan') values.
        """
        mydict = {}
        print('uspexStructuresData as_dict method: IMPLEMENTATION IN PROGRESS')

        attributes = [
            'extractedPOSCARBaseName', 'extractedPOSCARExtention',
            'IDs', 'generationNumbers', 'creationMethods',
            'numbersOfAtomsOfEachType', 'numbersOfAtoms',
            'enthalpies', 'volumes', 'densities',
            'fitnesses', 'kpoints', 'symmGroupNb',
            'Q_entr', 'A_order', 'S_order', 'distance_matrix',
            'individuals_dict',
        ]
        for attr_name in attributes:
            attribute = self.__getattribute__(attr_name)
            if isinstance(attribute, np.ndarray):
                mydict[attr_name] = attribute.tolist()
            elif isinstance(attribute, (int, str, list, float)):
                mydict[attr_name] = attribute

            # TODO: implement distance_data to dict conversion
            # Requires that distanceMatrixData aso has an as_dict method
        return mydict

    def to_json(self, json_file_name=None):
        """
        Create  JSON file from uspexStructuresData instance.
        """
        if json_file_name is None:
            json_file_name = self.json_file_name
        with open(json_file_name, 'w') as f:
            json.dump(f, self.as_dict())
        self.json_file_name = os.path.abspath(json_file_name)
        self.print('JSON file {} has been created from uspexStructuresData instance.'.format(
            self.json_file_name), verb_th=2)
        return self.json_file_name

    def update_distance_matrix_from_json(self):
        with open(self.json_file_name, 'w') as f:
            mydict = json.load(f)
        self.distance_matrix = np.array(mydict['distance_data'])

    def print(self, text, verb_th=1):
        """
        print if verbosity >= threshold
        """
        if self.verbosity >= verb_th:
            print(text)


    def set_distance_parameters(self, r = None, sigma=0.02, r_max=8.0, r_steps=512):
        """
        Re-initilize distance data with provided paremeters

        Args:
            r = None, sigma=None, r_max=None, r_steps=None
        """
        # Initialization of distance matrix data
        self.distance_data  = dt.distanceMatrixData(R=r, sigma=sigma,
            Rmax=r_max, Rsteps=r_steps)

    def load_uspex_structure_data(self, fileOrDirectoryName,
                                  set_volumes_and_densities=False, **kwargs) :
        """
        Reading structure IDs, enthalpies, generation number, volumes, etc. from Individuals file

        Parameters :
            fileOrDirectoryName : Name of a results[X] directory containing the output files of a uspex run. The directory should contain a "Individuals" file.

        Optional parameters :
            enthalpy : enthaly will be retrieved  from file.

            if none of the above is mentioned the function will return all parameters listed in the Individuals file.
        """
        targetFileName = 'Individuals'
        if not os.path.exists(fileOrDirectoryName) :
            errorMsg = fileOrDirectoryName + ' does not correspond to an existing file or directory.'
            raise ValueError(errorMsg)
        if os.path.isdir(fileOrDirectoryName) :
            self.resultsDirectoryName = os.path.normpath(fileOrDirectoryName)
            fileName = self.resultsDirectoryName + os.path.sep + targetFileName
            print('fileName = ',fileName)
            if not os.path.isfile(fileName) :
                errorMsg = 'No file Individuals in directory ' + fileOrDirectoryName + '.'
                raise ValueError(errorMsg)
        elif os.path.isfile(fileOrDirectoryName) :
            head, tail = os.path.split(fileOrDirectoryName)
            if not tail == targetFileName :
                print('WARNING : input file name is not : ',targetFileName)
            fileName = os.path.normpath(fileOrDirectoryName)
            name, self.resultsDirectoryName = os.path.split(os.path.abspath(fileName))

        self.uspexStructuresDataDir = self.resultsDirectoryName+os.path.sep+'uspexStructuresDataDir'
        self.extractedPOSCARSDirectoryName = self.uspexStructuresDataDir+os.path.sep+'extractedPOSCARS'
        if 'saveFileName' in kwargs :
            self.saveFileName = kwargs['saveFileName']
        else :
            # Setting default saveFileName
            self.saveFileName = self.uspexStructuresDataDir+os.path.sep+'savedData.pkl'

        self.print('Reading uspex-structure data from file : {}'.format(fileName), verb_th=1)

        individuals_dict = parse_individuals_file(fileName, verbosity=self.verbosity)

        """ Gen   ID    Origin   Composition    Enthalpy    Thickness   Surf_Area   Spec_surf_area   Fitness   KPOINTS  SYMM  Q_entr A_order S_order Magmoment-Type
        Gen   ID    Origin   Composition    Enthalpy   Volume  Density   Fitness   KPOINTS  SYMM  Q_entr A_order S_order
        """

        self.nbOfStructures = individuals_dict['nb_of_structures']
        self.generationNumbers = np.asarray(individuals_dict['Gen'])
        self.IDs = np.asarray(individuals_dict['ID'])
        self.creationMethods = individuals_dict['Origin']
        self.numbersOfAtomTypes = np.empty(self.nbOfStructures,dtype=int)
        self.numbersOfAtomsOfEachType = np.asarray(individuals_dict['Composition'])
        self.numbersOfAtoms = np.sum(self.numbersOfAtomsOfEachType, axis=1)
        self.enthalpies = np.asarray(individuals_dict['Enthalpy'])
        if 'Volume' in individuals_dict.keys():
            self.volumes = np.asarray(individuals_dict['Volume'])
        else:
            # TODO: set volumes from structures
            self.volumes = None
        if 'Density' in individuals_dict.keys():
            self.densities = np.asarray(individuals_dict['Density'])
            # TODO: set densities from structures
        else:
            self.densities = None
        self.fitnesses = np.asarray(individuals_dict['Fitness'])
        self.kpoints = np.asarray(individuals_dict['KPOINTS'])
        self.symmGroupNb = np.asarray(individuals_dict['SYMM'])
        self.Q_entr = np.asarray(individuals_dict['Q_entr'])
        self.A_order = np.asarray(individuals_dict['A_order'])
        self.S_order = np.asarray(individuals_dict['S_order'])

        self.individuals_dict = individuals_dict

        if (self.volumes is None or self.densities is None
            ) and set_volumes_and_densities:
            self.set_all_volumes_and_densities()

    # end of method get_uspex_structures_data

    def set_all_volumes_and_densities(self):
        """
        Load individual structures to retrieve volume and density in case
        """
        structures = self.get_structures_from_IDs(self.IDs)
        self.volumes = np.zeros(self.nbOfStructures,dtype=float)
        self.densities = np.zeros(self.nbOfStructures,dtype=float)
        for i, structure in structures:
            self.volumes[i] = structure.volume
            self.densities[i] = float(structure.density)

    def get_volumes_from_IDs(self, IDs):
        """
        get a list of volumes in Angstrom^3 from IDs
        """
        structures = self.get_structures_from_IDs(IDs)
        return [float(structure.volume) for structure in structures]

    def get_densities_from_IDs(self, IDs):
        """
        get a list of densities in g.cm-3 from IDs
        """
        structures = self.get_structures_from_IDs(IDs)
        return [float(structure.density) for structure in structures]


    def get_structure_IDs(self,**kwargs) :
        """
        Generate array of structure IDs from an USPEX run based on different criteria

        Used without an argument the function will return the IDs of the most stable structure obtained with the run. Other keywords may be used to obtain other structure IDs (N lowest-energy structures, all structures from a given generation, etc.)

        Args :
            IDs=[list or array of integer structure ID numbers]
            bestStructures=N : N structures with the lowest fitness criterion (most-commonly the enthalpy)
            mostStableStructures=N : N structures with the lowest enthalpy
            withinXeVFromBest=X : all structures with enthalpy less than X eV from lowest-enthalpy structure
            worstStructures=N : N structures with the highest fitness criterion (most-commonly the enthalpy)

        Returns :
            numpy array of integer numbers with the IDs of the requested structures
        """
        #**** SETTING DEFAULT VALUES FOR OPTIONAL FUNCTION INPUT ARGUMENTS *****
        # Here we use a list only if a unique structure is searched for to proceed the same way no matter what the number of requested input structure is

        # Setting structure selection to the best structure
        selectedStructureIDs = [np.argmin(self.fitnesses)]

        #**** READING AND PROCESSING OPTIONAL FUNCTION INPUT ARGUMENTS *******
        if 'bestStructures' in kwargs :
            # TODO : test value
            nbOfSelectedStructures = int(kwargs['bestStructures'])
            # rank structures by fitness
            sortedIndexes = np.argsort(self.fitnesses)
            selectedStructureIDs = self.IDs[sortedIndexes[0:nbOfSelectedStructures]]

        if 'mostStableStructures' in kwargs :
            # TODO : test value
            nbOfSelectedStructures = int(kwargs['mostStableStructures'])
            # rank structures by enthalpy
            sortedIndexes = np.argsort(self.enthalpies)
            selectedStructureIDs = self.IDs[sortedIndexes[0:nbOfSelectedStructures]]

        if 'worstStructures' in kwargs :
            # TODO : test value
            nbOfSelectedStructures = int(kwargs['worstStructures'])
            # rank structures by fitness
            sortedIndexes = np.argsort(self.fitnesses)[::-1]
            selectedStructureIDs = self.IDs[sortedIndexes[:nbOfSelectedStructures]]

        print('selectedStructureIDs = ',selectedStructureIDs)
        return selectedStructureIDs


    def get_extracted_POSCAR_file_name(self,structureID, use_initial=False) :
        if use_initial:
            file_name = os.path.join(self.extractedPOSCARSDirectoryName,
                                     '{}{}_initial{}'.format(self.extractedPOSCARBaseName,
                                                             structureID, self.extractedPOSCARExtention))
        else:
            file_name = os.path.join(self.extractedPOSCARSDirectoryName,
                                     '{}{}{}'.format(self.extractedPOSCARBaseName,
                                                     structureID, self.extractedPOSCARExtention))
        return file_name


    def get_extracted_POSCAR_file_names(self,structureIDs, use_initial=False) :
        listOfExtractedPOSCARFIleNames = [self.get_extracted_POSCAR_file_name(ID, use_initial=use_initial)
                                          for ID in structureIDs]
        return listOfExtractedPOSCARFIleNames

    def set_ase_atoms_list(self, set_info=True, use_initial=False):
        """
        Get a list of ASE Atoms objects from extracted POSCARS and store as  
        
        This function assumes that POSCAR files have already been extracted.
        
        Args:
            TO BE COMPLETED
        """
        use_initial_str = " initial structure" if use_initial else ""
        self.print(f'Reading ase Atoms from extracted POSCAR{use_initial_str} files', verb_th=1)
        
        try:
            # DEBUGGING
            self.ase_atoms_list = []
            for file_name in self.get_extracted_POSCAR_file_names(self.IDs, use_initial=use_initial):
                self.print(f"file_name = {file_name}", verb_th=2)
                atoms = read(file_name)
                self.ase_atoms_list.append(atoms)
            """
            self.ase_atoms_list = [read(file_name) for file_name in 
                                   self.get_extracted_POSCAR_file_names(self.IDs, use_initial=use_initial)]
            """
        except FileNotFoundError as e:
            self.print((f"At least on extracted POSCAR file does not exist; {e}. "
                        f"Extracting all files at once."), verb_th=2)
            self.extract_all_poscars(use_initial=use_initial)
            self.ase_atoms_list = [read(file_name) for file_name in 
                                   self.get_extracted_POSCAR_file_names(self.IDs, use_initial=use_initial)]
        except Exception as e:
            raise ValueError(f"Exception found: {e}")

        self.print(f'{len(self.ase_atoms_list)} ASE-Atoms list created and stored in ase_atoms_list property.', 
                   verb_th=1)

        # write structure-specific information 
        if set_info:
            for index, atoms in enumerate(self.ase_atoms_list):
                self.set_ase_atoms_info(atoms, structure_index=index, 
                                        use_initial=use_initial)
        
        self.print('Structure-specific metadata added to info property of all ASE Atoms in ase_atoms_list.', 
                   verb_th=1)

    def set_ase_atoms_info(self, ase_atoms, structure_index=None, 
                           structure_id=None, use_initial=False):
        """
        Write structure-specific metadata information into ASE Atoms info property

        TODO: make sure in-place modification works

        :param self: Description
        :param ase_atoms: Description
        :param structure_index: Description
        :param structure_id: Description
        :param use_initial: Description
        """
        if not isinstance(structure_index, int) and not isinstance(structure_id, int):
            raise ValueError("set either structure_index or structure_id.")
        if structure_id:
            structure_index = self.get_structure_index_from_ID(structure_id)
        
        ase_atoms.info['origin'] = 'uspex'
        ase_atoms.info['uspex_run_name'] = self.uspex_run_name
        ase_atoms.info['ID'] = structure_id
        ase_atoms.info['is_initial_structure'] = use_initial
        if not use_initial:
            # properties related to relaxed strutcures only
            ase_atoms.info['enthapy'] = self.enthalpies[structure_index]
            ase_atoms.info['fitness'] = self.fitnesses[structure_index]
            ase_atoms.info['kpoints'] = self.kpoints[structure_index].tolist()
            ase_atoms.info['symmGroupNb'] = self.symmGroupNb[structure_index]
            ase_atoms.info['Q_entr'] = self.Q_entr[structure_index]
            ase_atoms.info['A_order'] = self.A_order[structure_index]
            ase_atoms.info['S_order'] = self.S_order[structure_index]
        
        return ase_atoms


    def get_ase_atoms_from_poscar_file(self, id, use_initial=False, extract_all_poscars=True):
        # Read from file (extract if necessary)
        file_name = self.get_extracted_POSCAR_file_name(id, use_initial=use_initial)  
        if not os.path.exists(file_name):
            if extract_all_poscars:
                self.extract_all_poscars(use_initial=use_initial)
            else:
                self.extract_POSCARS_from_IDs([id])

        ase_atoms = read(file_name, format='vasp')
        return ase_atoms

    def get_structures_from_IDs(self, structureIDs, extract_poscar_files=False,
                                use_initial=False, get_ase_atoms=False) :
        """
        Create a (list of) pymatgen Structure or ASE Atoms from a CSP run with USPEX.

        Structures with the requested IDs will be read from the gatheredPOSCARS file (the default) or alternatively from gatheredPOSCARS_order or goodStructures_POSCARS files.

        Args :
            structureIDs : array (or list of) of indexes

        Returns :
            selectedStructures = (list of) pymatgen.core.structure Structure class object(s)

        TODO :
            - read from other POSCAR files (faster parsing when looking for best structures)
            - allow reading symmetrized structures from symmetrized_structures.cif file. THe parsing should be adapted in this case.
        """
        structureIDs = self.make_list_if_single_element(structureIDs)
        
        if get_ase_atoms:
            selected_ase_atoms = []
            for id in structureIDs:
                index = self.get_structure_index_from_ID(id)
                if not use_initial:
                    if self.ase_atoms_list[index]['ID'] == id:
                        selected_ase_atoms.append(self.ase_atoms_list[index])
                    else:
                        raise ValueError(f"ID {id} does not match structure index {index} "
                                         f"(ID {self.ase_atoms_list[index]['ID']})")
                else:
                    selected_ase_atoms.append(
                        self.get_ase_atoms_from_poscar_file(id, use_initial=use_initial))
 
            [self.ase_atoms_list[i] for i in 
                                  self.get_indexes_from_IDs(structureIDs)]
            return selected_ase_atoms

        if not extract_poscar_files:
            if get_ase_atoms:
                warn('ASE Atoms will be more efficiently extracted if POSCAR files are extracted' \
                     'with extract_poscar_files')
                return [get_ase_atoms(poscar.structure) for poscar in 
                        self.get_poscars_from_IDs(structureIDs)]
            else:
                return [poscar.structure for poscar in self.get_poscars_from_IDs(structureIDs)]

        missingIDs = [ID for ID in structureIDs if not os.path.exists(
                      self.get_extracted_POSCAR_file_name(ID, use_initial=use_initial))]
        if len(missingIDs) > 0 :
            self.extract_POSCARS_from_IDs(missingIDs, use_initial=use_initial)

        if get_ase_atoms:
            selectedStructures = [read(fileName) for fileName in
                                  self.get_extracted_POSCAR_file_names(structureIDs, 
                                                                   use_initial=use_initial)]
        else:
            selectedStructures = [Structure.from_file(fileName) for fileName in
                                  self.get_extracted_POSCAR_file_names(structureIDs, 
                                                                       use_initial=use_initial)]

        return selectedStructures


    def get_structure_from_ID(self, structureID, extract_poscar_file=False,
                              use_initial=False, get_ase_atoms=False) :
        """
        Create a pymatgen Structure object(s) obtained from a crystal structure prediction run with USPEX.

        The structure with the requested ID will be read from the gatheredPOSCARS file (the default) 
        or alternatively from gatheredPOSCARS_order or goodStructures_POSCARS files.

        Args :
            structureID : index of considered structure. Accepts an array or list containing a single ID. A warning will be printed if an array or list of multiple elements. The function will in this case call get_structures_from_IDs and return a list of structures.

        Returns :
            selectedStructures = pymatgen.core.structure Structure class object(s) (or a list thereof, see above.)
        """
        if get_ase_atoms and not self.ase_atoms_list:
            self.set_all_ase_atoms()

        # Convert structureID to a single element to use in class method get_structures_from_IDs
        # and returns the first (and in principle only structure in list
        structureID = self.make_single_value_if_list_of_single_element(structureID)
        if not extract_poscar_file:
            if get_ase_atoms:
                struct_index = self.get_structure_from_ID(structureID)
                return self.ase_atoms_list[struct_index]

            # read individual structure from gatheredPOSCARS file
            try:
                [poscar] = self.get_poscars_from_IDs([structureID])
            except ValueError as e:
                raise ValueError('{} : poscar could not be loaded for structure ID {}'.format(
                    e, structureID))
            
            return poscar.structure
        else:
            try :
                if len(structureID) > 0 :
                    print('WARNING : structure ID has a length larger than 1. Function will return a list of structures rather than a structure')
                    structure = self.get_structures_from_IDs(structureID, use_initial=use_initial, 
                                                             get_ase_atoms=get_ase_atoms)
            except TypeError :
                # structureID is neither an array nor a list
                structure = self.get_structures_from_IDs([structureID], use_initial=use_initial, 
                                                         get_ase_atoms=get_ase_atoms)[0]
            return structure

    def get_poscars_from_IDs(self, structureIDs, use_initial=False) :
        """
        get individual pymatgen Poscar instances for selected structure IDs

        Args :
            structureIDs : list or numpy array of integer structure ID numbers

            use_initial: bool (default is False)
                initial (unrelaxed) structures will be extracted instead of final.

        Returns :
            list of Poscar instances
        """

        structureIDs = self.make_list_if_single_element(structureIDs)

        # Setting default POSCARS file
        POSCARSFile_basename = 'gatheredPOSCARS_unrelaxed' if use_initial else 'gatheredPOSCARS'
        POSCARSFile = os.path.join(self.resultsDirectoryName, POSCARSFile_basename)
        if not os.path.exists(POSCARSFile) :
            errorMsg = ('File : ',POSCARSFile,' does not exist.')
            raise ValueError(errorMsg)

        # convert structureIDs (which may be a list for example) to a numpy array
        structureIDs = np.asarray(structureIDs,dtype=int)

        self.print('Looking for structures with IDs {} in file {}'.format(
            structureIDs ,POSCARSFile), verb_th=2)
        with open(POSCARSFile, 'r') as f:
            lines = f.readlines()

        poscars = []
        reading_structure = False
        poscar_lines = []
        extracted_ids = []
        for line in lines:
            try:
                if line[0:2] == 'EA':
                    if reading_structure:
                        # finish reading a structure
                        try:
                            poscars.append(Poscar.from_str(''.join(poscar_lines)))
                        except AttributeError:  # Keep compatibility with older pymatgen versions
                            poscars.append(Poscar.from_string(''.join(poscar_lines)))
                        extracted_ids.append(currentStructID)
                        reading_structure = False
                        if len(poscars) >= structureIDs.size:
                            self.print('Poscar instances have been obtained from all '
                                       'requested  structures.', verb_th=2)
                            break
                    # start reading structure
                    currentStructID = int(line[2:].split()[0])
                    if currentStructID in structureIDs:
                        # Create POSCAR file and write first line
                        reading_structure = True
                        # Initialize poscar_lines
                        poscar_lines = []
            except IndexError:
                pass  # len(line) is smaller than 2
            if reading_structure :
                poscar_lines.append(line)
        if self.IDs[-1] in structureIDs and self.IDs[-1] not in extracted_ids:
            # In principle that last structure is not stored by the procedure above,
            # unless Individuals file contains less structures than the gatherPOSCARS file.
            # This situation can happen if structures with fitness value N/A are omitted
            # or simply deleted from the file.
            try:
                poscars.append(Poscar.from_str(''.join(poscar_lines)))
            except AttributeError:  # Keep compatibility with older pymatgen versions
                poscars.append(Poscar.from_string(''.join(poscar_lines)))
        return poscars

    # end of method get_structures_from_IDs

    def extract_all_poscars(self, use_initial=False):
        """
        Extract all poscars at once from approprate gathered_POSCARS file
        
        :param self: Description
        :param use_initial: Description
        """
        # Setting default POSCARS file
        POSCARSFile_basename = 'gatheredPOSCARS_unrelaxed' if use_initial else 'gatheredPOSCARS'
        POSCARSFile = os.path.join(self.resultsDirectoryName, POSCARSFile_basename)
        if not os.path.exists(POSCARSFile) :
            errorMsg = ('File : ',POSCARSFile,' does not exist.')
            raise ValueError(errorMsg)
        
        if not os.path.exists(self.uspexStructuresDataDir) :
            os.mkdir(self.uspexStructuresDataDir)

        # checking if extractedPOSCARS directory exists in results directory
        if not os.path.exists(self.extractedPOSCARSDirectoryName) :
            os.mkdir(self.extractedPOSCARSDirectoryName)

        listOfExtractedPOSCARFileNames = []
        self.print(f"Extracting individual POSCAR files from file {POSCARSFile}", 
                   verb_th=1)
        with open(POSCARSFile, 'r') as inputFile:
            count = 0
            extractedPOSCARFileName = None
            while True:
                count += 1
                line = inputFile.readline()
                self.print(f"Line {count}: {line}", verb_th=3)  # hard-core debugging
                if not line:  # end of file is reached
                    self.print(f'End of file {POSCARSFile} is reached.', verb_th=2)
                    self.print(f'Closing file {extractedPOSCARFileName} and exiting.', verb_th=2)
                    outputFile.close()
                    break
                    
                if len(line) >= 2 and line[0:2] == 'EA' :
                    self.print("Line starts with EA", verb_th=3)
                    if extractedPOSCARFileName:
                        print('Closing file {}'.format(extractedPOSCARFileName))
                        outputFile.close()
                        listOfExtractedPOSCARFileNames.append(extractedPOSCARFileName)
                    
                    currentStructID = int(line[2:].split()[0])
                    extractedPOSCARFileName = self.get_extracted_POSCAR_file_name(currentStructID, 
                                                                                  use_initial)
                    outputFile = open(extractedPOSCARFileName, 'w')
                    self.print(f"Now writing in file {extractedPOSCARFileName}", verb_th=2)
                    outputFile.write(line)
                else:
                    outputFile.write(line)
                                
        return listOfExtractedPOSCARFileNames


    def extract_POSCARS_from_IDs(self, structureIDs, use_initial=False) :
        """
        extract individual POSCAR (i.e. VASP atomic structure) files for selected structure IDs

        Relaxed structures obtained during a USPEX run are stored in a gatheredPOSCARS (and orther similar) files in the VASP POSCAR format. This method creates individual POSCAR files for each requested structure ID and stores them with the name :
        <usd.extractedPOSCARSDirectoryName><os.path.sep><usd.extractedPOSCARBaseName><ID><usd.extractedPOSCARExtention>
        where usd is uspexStructuresData object.
        Args :
            structureIDs : list or numpy array of integer structure ID numbers

            use_initial: bool (default is False)
                initial (unrelaxed) structures will be extracted instead of final.

        Returns :
            list of extracted POSCAR file names
        """

        structureIDs = self.make_list_if_single_element(structureIDs)

        # Setting default POSCARS file
        POSCARSFile_basename = 'gatheredPOSCARS_unrelaxed' if use_initial else 'gatheredPOSCARS'
        POSCARSFile = os.path.join(self.resultsDirectoryName, POSCARSFile_basename)
        if not os.path.exists(POSCARSFile) :
            errorMsg = ('File : ',POSCARSFile,' does not exist.')
            raise ValueError(errorMsg)

        # convert structureIDs (which may be a list for example) to a numpy array
        structureIDs = np.asarray(structureIDs,dtype=int)

        if not os.path.exists(self.uspexStructuresDataDir) :
            os.mkdir(self.uspexStructuresDataDir)

        # checking if extractedPOSCARS directory exists in results directory
        if not os.path.exists(self.extractedPOSCARSDirectoryName) :
            os.mkdir(self.extractedPOSCARSDirectoryName)

        listOfExtractedPOSCARFileNames = []
        # TODO :
        # - open and scan file looking for lines starting with EAXXX where XXX is the structure ID
        # - when requested and read IDs match : open structure with pymatgen
        # - append structure to selectedStructures list
        print('Looking for structures with IDs : ',structureIDs,' in file ',POSCARSFile)
        inputFile = open(POSCARSFile, 'r')
        count = 0
        extractedPOSCARFileName = None
        writingFile = False
        while True:
            count += 1
            line = inputFile.readline()
            try :
                if line[0:2] == 'EA' :
                    if writingFile :
                        print('Closing file {}'.format(extractedPOSCARFileName))
                        outputFile.close()
                        listOfExtractedPOSCARFileNames.append(extractedPOSCARFileName)
                        writingFile = False
                        if len(listOfExtractedPOSCARFileNames) == structureIDs.size :
                            print('All requested structures have been extracted.')
                            break
                    currentStructID = int(line[2:].split()[0])
                    # IDindex = np.where(structureIDs == int(line[2:].split()[0]))
                    for ID in structureIDs :
                        if ID == currentStructID :
                            # Create POSCAR file and write first line
                            writingFile = True
                            extractedPOSCARFileName = self.get_extracted_POSCAR_file_name(ID, use_initial)
                            outputFile = open(extractedPOSCARFileName, 'w')
            except IndexError :
                print('len(line) = ',len(line),' smaller than 2.')
            if writingFile :
                outputFile.write(line)

            # if line is empty end of file is reached
            if not line:
                if len(listOfExtractedPOSCARFileNames) < structureIDs.size :
                    print('End of file ',POSCARSFile,' reached although some of the requested structure POSCAR files have not been extracted.')
                break
        print('Closing file ',POSCARSFile)
        inputFile.close()

        return listOfExtractedPOSCARFileNames



    def visualize_structure(self,structureIDs,**kwargs) :
        """
        Visualize structure associated with the chosen ID(s) with VTK package
        """
        if 'visualizationProgram' in kwargs :
            import subprocess
            if kwargs['visualizationProgram'] == 'VESTA' :
                # TODO : the part here should be made platform-independent
                # Only works with executable full path on Windows
                executable = r'C:\Users\cadarp02\VESTA\VESTA-win64\VESTA.exe'
                listOfExtractedPOSCARFileNames = self.extract_POSCARS_from_IDs(structureIDs)
                sp = subprocess.run([executable]+listOfExtractedPOSCARFileNames,capture_output=True)

                # try :
                    # subprocess.run([executable]+listOfExtractedPOSCARFileNames)
                # except :
                    # print('Vesta program not found.')
        else :
            from pymatgen.vis.structure_vtk import StructureVis
            structVis = StructureVis()
            structVis.set_structure(self.get_structure_from_ID(structureIDs), reset_camera=True, to_unit_cell=True)
            structVis.show()

    def make_single_value_if_list_of_single_element(self, arrayOrList) :
        """
        Deal with particular case where a single element (ID, structure) is manipulated. For internal use.
        """
        try :
            [result] = arrayOrList
        except TypeError:
            # print('make_single_value_from_array_or_list : already a single element')
            result = arrayOrList
        except ValueError:
            if len(arrayOrList) > 1 :
                # print('Multiple elements in array or list. Returning input unchanged.')
                result = arrayOrList
        # print('make_single_value_if_list_of_single_element method : ',arrayOrList,' converted to ',result)
        return result

    def make_list_if_single_element(self, inputValues,printWarning=False) :
        """
        Deal with particular case where a single element (ID, structure) is manipulated. For internal use.
        """
        self.print('inputValues = {}'.format(inputValues), verb_th=3)
        try :
            if len(inputValues) == 1 :
                result = inputValues
            elif len(inputValues) > 0 :
                # list (or array) contains more than one element. returning it unchanged.
                if printWarning :
                    print('WARNING : input of method make_list_from_single_element() is a list or array of length greater than one. Returning input unchanged.')
                result = inputValues
            else :
                raise ValueError('len(inputValues) = 0')
        except TypeError :
            # structureID is neither an array nor a list
            result = [inputValues]
        self.print('make_list_if_single_element : {} converted to {}'.format(
            inputValues,result), verb_th=3)
        return result

    # End of staticmethod make_list_if_single_element


    def get_cosine_distance_matrix(self, structureIDs, systemName='',
                                   update_distance_matrix=True, **kwargs) :
        """
        Calculate cosine distance matrix between a set of structures designated by their IDs.

        The method implemented is based on Oganov, Artem R., et Mario Valle. « How to Quantify Energy Landscapes of Solids ». The Journal of Chemical Physics 130, nᵒ 10 (14 mars 2009): 104504. https://doi.org/10.1063/1.3079326. , with a structure fingerprint function defined in eq (3) and cosine distance defined in equations (6b) and (7) therein.

        TO BE COMPLETED
        """
        indexes = self.get_indexes_from_IDs(structureIDs)

        distance_matrix = np.empty(2*[len(structureIDs)])
        distance_matrix[:] = np.nan
        np.fill_diagonal(distance_matrix, 1.0)
        for i, (index_1, ID_1) in enumerate(zip(indexes, structureIDs)):
            for j, (index_2, ID_2) in enumerate(zip(indexes, structureIDs)):
                if j >= i:
                    break
                distance_matrix[i, j] = self.get_distance(ID_1, ID_2)
                distance_matrix[j, i] = distance_matrix[i, j]
                if update_distance_matrix:
                    self.distance_matrix[index_1, index_2] = distance_matrix[i, j]
                    self.distance_matrix[index_2, index_1] = distance_matrix[j, i]

        return distance_matrix

    def save_data_to_file(self,saveFileName='') :
        """
        Store uspexStructuresData object as a pickle file.
        """

        if len(saveFileName) == 0 :
            saveFileName = self.saveFileName

        with open(saveFileName, 'wb') as f:
            pickle.dump(self, f)
            self.saveFileName = saveFileName
        print('uspexStructuresData saved in file : ',saveFileName)


    def load_distance_matrix_from_file(self,saveDistMatrixFileName='') :
        """
        Load distance_data  attribute object of class distanceMatrixData from a pickle file.

        DEPRECATED.
        """
        print('DEPRECATED. DATA SHOULD BE SAVED AS JSON. NOT IMPLEMENTED YET.')
        if len(saveDistMatrixFileName) == 0 :
            saveDistMatrixFileName = self.saveDistMatrixFileName
        with open(saveDistMatrixFileName,'rb') as f:
            self.distance_data  = pickle.load(f)


    def get_indexes_from_IDs(self,structureIDs):
        """
        get structure indexes in database based on their iDs
        """
        indexes = [index for index,ID in enumerate(self.IDs) if ID in structureIDs]

        return indexes


    def get_ranks_from_IDs(self, structureIDs, one_based=True):
        """
        Get rank (from 1 to nbOfStrctures+1) based on fitness for selected IDs
        """
        structureIDs = self.make_list_if_single_element(structureIDs)
        # rank all structures based on fitnesses
        ranks = rankdata(self.fitnesses, method='min')
        # select structures associated with the selected IDs
        selectedRanks = ranks[self.get_indexes_from_IDs(structureIDs)]
        if one_based:
            selectedRanks += 1

        return selectedRanks


    def get_IDs_sorted_by(self, sort_by:str='fitness',
                          sort_order:str='ascending'):
        """
        Get IDs sorted by fitness (the default or other such as enthalpy_by_atom)

        Args:
            sort_by: str (default is 'fitness')
                parameter to use for sorting IDs.
                other options are :
                enthalp... or energ... : enthalpy by atom
            sort_order: str (default is 'ascending')

        returns sorted_ids
        """
        if sort_by == 'fitness':
            sorted_indexes = np.argsort(self.fitnesses)
        elif 'enthalp' or 'energ' in sort_by:
            sorted_indexes = np.argsort(self.enthalpies / self.numbersOfAtoms)
        else:
            raise ValueError('sort_by {} is not implemented.'.format(sort_by))

        if sort_order in ['descending', 'reverse']:
            sorted_indexes = sorted_indexes[::-1]

        sorted_IDs = self.IDs[sorted_indexes]

        return sorted_IDs


    def plot_enthalpies(self,structureIDs,enthalpyUnit:str='eV',
                        relativeEnthalpies:bool=True,systemName:str='',
                        **kwargs):
        """
        Draw a plot of USPEX structure final enthalpies vs rank

        Args :
            enthalpyUnit : str, OPTIONAL
                specify energy unit among the following : meVByAtom',
                'eVbyAtom','meV','eV' (the default)
            relativeEnthalpies : bool, OPTIONAL
                If true (the default) the y axis will be E-min(E)
                (default is False)
            systemName : str, OPTIONAL
                Adds systemName+' : ' at he beginning of plot title

        Optional keyword arguments :
            Any axis property may be used here and will be passed to plot axes.

        Returns :
            Figure handle
        """

        x = self.get_ranks_from_IDs(structureIDs)
        y = self.enthalpies[self.get_indexes_from_IDs(structureIDs)]
        yLabel = 'Enthalpy'
        yUnit = 'eV'
        try :
            if relativeEnthalpies == True:
                y = y - np.amin(self.enthalpies)
                yLabel = 'H - min(H)'
        except :
            print('Exception : using absolute enthalpies.')
        try :
            if enthalpyUnit == 'meV':
                y = 1000*y
                yUnit = 'meV'
            elif enthalpyUnit.lower() == 'meVByAtom'.lower():
                y = 1000*np.divide(y,self.numbersOfAtoms[self.get_indexes_from_IDs(structureIDs)])
                yUnit = 'meV/Atom'
            elif enthalpyUnit.lower() == 'eVByAtom'.lower():
                y = np.divide(y,self.numbersOfAtoms[self.get_indexes_from_IDs(structureIDs)])
                yUnit = 'eV/Atom'
            else :
                print('Unknown unit. Plotting enthalpies in eV')
        except :
            print('Exception : using eV as the enthalpy unit.')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'o', label='enthalpies')
        title = 'Structure enthalpies vs fitness rank'
        if systemName != '' :
            title=systemName+' : '+title
        ax.set_title(title)
        ax.set(xlabel='Structure rank (by fitness)',**kwargs)
        # TODO : change y axis label as a function of
        ax.set(ylabel=yLabel + ' (' + yUnit + ')')

        return fig, ax


    def printTable(self,structureIDs):
        """
        Print table with all relevant data in a convenient format

        Returns
        -------
        None.

        """
        # TODO : adapt
        indexes = self.get_indexes_from_IDs(structureIDs)
        rowStr = ''
        for index in indexes :
           rowStr+='{:5d}'.format(self.IDs[index])
           rowStr+='{:5d}'.format(index)
           rowStr+='{:5d}'.format(self.generationNumbers[index])
           rowStr+='{:12s}'.format(self.creationMethods[index])
           rowStr+='{:12s}'.format(self.creationMethods[index])
           rowStr+='{:8.4f}'.format(self.enthalpies[index])

           """
           # TODO : continue list...
            self.numbersOfAtomTypes[structureIndex] = len(lineSplit2)
            a = [int(s) for s in lineSplit2]
            self.numbersOfAtomsOfEachType.append(a)
            self.numbersOfAtoms[structureIndex] = np.sum(a)
            self.enthalpies[structureIndex] = float(lineSplit3[0])
            self.volumes[structureIndex] = float(lineSplit3[1])
            self.densities[structureIndex] = float(lineSplit3[2])
            self.fitnesses[structureIndex] = float(lineSplit3[3])
            self.kpoints[structureIndex:(structureIndex+1)][0:3] = [int(s) for s in lineSplit4 if s.isdigit()]
            self.symmGroupNb[structureIndex] = int(lineSplit5[0])
            self.Q_entr[structureIndex] = float(lineSplit5[1])
            self.A_order[structureIndex] = float(lineSplit5[2])
            self.S_order[structureIndex]
           """

    def initialize_distance_matrix(self):
        # Initialize matrix
        self.distance_matrix = np.empty((len(self.IDs), len(self.IDs)))
        self.distance_matrix[:] = np.nan
        np.fill_diagonal(self.distance_matrix, 0.0)

    def get_structure_index_from_ID(self, id):
        """ In case indexes and IDs do not match"""
        if not np.issubdtype(type(id), int):  #  isinstance(id, int) will not work of id is a numpy int
            raise TypeError('id should be an int (or numpy int64, int32, etc.).')
        [index],  = np.where(self.IDs == id)
        return index

    def calculate_cosine_distance_matrix(self, selectedIDs=None, r=None, 
                                         r_max=None, r_steps=None, sigma=None, 
                                         n_jobs=None, return_plot=False, 
                                         show_plot=False):
        if any([sigma, r_max, r_steps, r]):
            self.set_distance_parameters(r, sigma, r_max, r_steps)

        if not self.ase_atoms_list:
            self.set_all_ase_atoms()

        if selectedIDs:
            selected_indexes = self.get_indexes_from_IDs(selectedIDs)
            atoms_list = [self.ase_atoms_list[i] for i in selected_indexes]
        else:
            atoms_list = self.ase_atoms_list

        outputs = self.distance_data.calculate_distance_matrix(atoms_list, structureIDs=selectedIDs, 
                                                               n_jobs=n_jobs, return_plot=return_plot, 
                                                               show_plot=show_plot)
        print(outputs)
        if return_plot:
            (D, fig, ax) = outputs
        else:
            D = outputs
        
        # Update distance_matrix
        if not selectedIDs:
            self.distance_matrix = D
        else:
            if not self.distance_matrix:
                self.initialize_distance_matrix()
            for i, A_index in enumerate(selected_indexes):
                for j, B_index in enumerate(selected_indexes):
                    if j > i:
                        break
                    if j == i:
                        self.distance_matrix[A_index, A_index] = D[i, i]
                    elif i == j:
                        self.distance_matrix[A_index, B_index] = D[i, j]
                        self.distance_matrix[B_index, A_index] = D[j, i]

        if return_plot:
            if show_plot:
                fig.show()
            return fig, ax         
        

    def get_distance(self, ID_1, ID_2, update_distance_matrix=True):
        """
        Get distance between 2 structures from their IDs

        Cosine distance matrix based on two-atom fingerprints
        as defined by Oganov and Valle 2009.

        Args:
            ID_1: int
                UPSEX ID of first structure (one-based indexing)
            ID_2: int
                UPSEX ID of 2nd structure (one-based indexing)
            update_distance_matrix: bool (default is True)
                Whether self.distance_matrix should be updated

        Returns:
            distance between structures
        """
        if not self.ase_atoms_list:
            self.set_all_ase_atoms()
        
        i = self.get_structure_index_from_ID(ID_1)
        j = self.get_structure_index_from_ID(ID_2)
        self.print(('Calculating distances between structures {} and {} '
                    'with indexes ({}, {})').format(ID_1, ID_2, i, j), verb_th=3)
        if self.distance_matrix is None:
            if update_distance_matrix:
                self.initialize_distance_matrix()
        if np.isnan(self.distance_matrix[i, j]):
            distance = self.distance_data.calculate_cosine_distance(
                self.get_structure_from_ID(ID_1), self.get_structure_from_ID(ID_2))
            if update_distance_matrix:
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance
        else:
            distance = self.distance_matrix[i, j]
        self.print('Distance between structures IDs {} and {}: {:.3f}'.format(
            ID_1, ID_2, distance), verb_th=2)

        return distance

    
    def get_sorted_indexes_ids_and_energies(self, use_relative_energies=True):
        # Sort structures by enthalpy
        energies_by_atom = self.enthalpies / self.numbersOfAtoms
        sorted_indexes = np.argsort(energies_by_atom)
        sorted_energies_by_atom = energies_by_atom[sorted_indexes]
        if use_relative_energies:
            sorted_energies_by_atom -= np.min(energies_by_atom)
        sorted_ids = self.IDs[sorted_indexes]
        
        return sorted_indexes, sorted_ids, sorted_energies_by_atom


    def get_distance_matrix_sorted_by_energy(self, n_jobs=1, return_plot=True):
        """
        Get a distance matrix with indexes sorted by enrgy
        """
        # Check whether distance_matrix is already full
        if not isinstance(self.distance_matrix, np.ndarray) or np.any(np.isnan(self.distance_matrix)):
            # TODO: calculate missing datapoints only
            self.print('There are missing distances in the distance matrix. (Re)calculating '
                       'the full matrix.', verb_th=1)
            self.calculate_cosine_distance_matrix(n_jobs=n_jobs)

        sorted_indexes, sorted_ids, sorted_energies_by_atom = self.get_sorted_indexes_ids_and_energies()

        sorted_matrix = np.take(np.take(self.distance_matrix, sorted_indexes, axis=0), 
                                sorted_indexes, axis=1)
        
        if return_plot:
            fig, ax = dt.plot_distance_matrix(sorted_matrix, 
                                              structure_names=[str(id) for id in sorted_ids],
                                              system_name=self.uspex_run_name)
            
        sorted_matrix_dict = {
            'matrix': sorted_matrix, 
            'sorted_indexes': sorted_indexes, 
            'sorted_ids': sorted_ids, 
            'sorted_energies_by_atom': sorted_energies_by_atom, 
        }
        if return_plot:
            sorted_matrix_dict.update({'figure': fig, 'axes': ax})

        return sorted_matrix_dict


    def is_full_distance_matrix(self):
        """ Check whether distance matrix is full or not (i.e. contains NaN terms """
        # TODO: add the possibility to test a sub-matrix
        is_full = (isinstance(self.distance_matrix, np.ndarray) 
                   and not np.any(np.isnan(self.distance_matrix)))
        if not is_full:
            self.print('There are missing distances in the distance matrix.', verb_th=1)
        return is_full

    def get_distance_matrix_sorted_by_distance_to_ref(self, ref='best', n_jobs=1, 
                                                      return_plot=True):
        """
        Get a distance matrix with indexes sorted by energy
        """
        if ref == 'best':  # best based on fitness
            ref_id = self.get_good_structures_ids()[0]
        elif ref == 'lowest-energy':
            _, sorted_ids, _ = self.get_sorted_indexes_ids_and_energies()
            ref_id = sorted_ids[0]
        elif isinstance(ref, str):
            raise ValueError(f'Allowed ref string values are \'best\' and \'lowest-energy\'.')
        elif np.issubdtype(type(ref), int):
            ref_id = int(ref)
            if ref_id not in self.IDs:
                raise ValueError(f'ref value ({ref}) does not correspond to a valid uspex ID.')
        else:
            raise TypeError('Ref should be a str or (numpy) int.')

        if not self.is_full_distance_matrix():
            # TODO: calculate missing datapoints only
            self.calculate_cosine_distance_matrix(n_jobs=n_jobs)

        ref_index = self.get_structure_index_from_ID(ref_id)

        distances_to_ref = self.distance_matrix[ref_index]
        sorted_indexes = np.argsort(distances_to_ref).tolist()
        # Ensure ref_index comes first (in two structires are strictly identical)
        sorted_indexes.remove(ref_index)
        sorted_indexes.insert(0, ref_index)
        sorted_indexes = np.array(sorted_indexes)
        sorted_ids = self.IDs[sorted_indexes]
        sorted_distances_to_ref = distances_to_ref[sorted_indexes]

        sorted_matrix = np.take(np.take(self.distance_matrix, sorted_indexes, axis=0), 
                                sorted_indexes, axis=1)
        
        if return_plot:
            fig, ax = dt.plot_distance_matrix(sorted_matrix, 
                                              structure_names=[str(id) for id in sorted_ids],
                                              system_name=self.uspex_run_name + ", structures sorted by dist to ref.")
            
        sorted_matrix_dict = {
            'matrix': sorted_matrix, 
            'sorted_indexes': sorted_indexes, 
            'sorted_ids': sorted_ids, 
            'sorted_distances_to_ref': sorted_distances_to_ref, 
        }
        if return_plot:
            sorted_matrix_dict.update({'figure': fig, 'axes': ax})

        return sorted_matrix_dict

    def select_energy_distant_structures(self, nb_of_structures,
                                         max_calc_per_structure=10,
                                         pick_lowest_energy=True,
                                         pick_highest_energy=True,
                                         sigma=None, r_max=None, seed=None,
                                         show_full_distance_matrix=False):
        """
        Divide energy range in bins and maximize distance between picked
        structures in adjacent bins.

        Structures may be distinguished based on:
            * their energies (in eV/atom)
            * distance (Oganov and Vale, PDF)
        (1) fix an energy range for considered structures
        (2) divide energy range in nb_of_structures segments
        (3) randomly pick max_struct_per_energy_segment in segment
            (or all structures if less than this)
        (4) For each struct in segment:
            calculate distance to all previously-selected structures

        TODO: enforce the number of output structures in cases where scare
              high-energy or low-energy structures result in empty bins

        Args:
            TO BE COMPLETED

        Returns:
            seed: seed used for the random number generator.
        """
        ssq = np.random.SeedSequence(seed)
        rng = np.random.default_rng(ssq)
        seed = ssq.entropy
        print('Random number generation:\nSeed = {}'.format(seed))
        # Make sure seed is returned

        selected_ids = []

        # Sort structures by energy by atom
        sorted_indexes, sorted_IDs, sorted_E_rel = self.get_sorted_indexes_ids_and_energies()
        
        global_distance_matrix = np.zeros((len(sorted_IDs), len(sorted_IDs)))

        # TODO: initialize nb_of_bins to nb_of_structures and increase until
        # the nb_of_occupied_bins is equal to nb_of_structures
        
        nb_of_bins = nb_of_structures
        nb_of_occupied_bins = 0
        while 1:
            nb_of_occupied_bins = 0
            # split into energy segments
            hist, bin_edges = np.histogram(sorted_E_rel, bins=nb_of_bins)
            _ = np.digitize(sorted_E_rel, bin_edges, right=False)
            bin_indexes = np.where(_ == nb_of_bins + 1, nb_of_bins, _) - 1
            
            for i, bin_pop in enumerate(hist):
                IDs_in_bin = [ID for ID in sorted_IDs[bin_indexes == i]
                              if ID not in selected_ids]
                if len(IDs_in_bin) > 0:
                    nb_of_occupied_bins += 1
            
            if nb_of_occupied_bins >= nb_of_structures:
                self.print(f'Splitting the energy range in {nb_of_bins} yields '
                           f'{nb_of_occupied_bins} occupied bins, matching the '
                           f'targeted number of structures.', verb_th=1)
                break
            
            self.print(f'Only {nb_of_occupied_bins} out of {nb_of_bins} contained '
                       f'structures for a target of {nb_of_structures} structures. '
                       'Incrementing the number of bins.', verb_th=2)
            nb_of_bins += 1
            
        struct_index = 0
        for i, bin_pop in enumerate(hist):
            IDs_in_bin = [ID for ID in sorted_IDs[bin_indexes == i]
                          if ID not in selected_ids]
            if len(IDs_in_bin) > 0:
                self.print('Energy bin edges = {} to {} ev/atom abobe min'.format(
                    bin_edges[i], bin_edges[i + 1]), verb_th=2)
                if bin_pop == 1:
                    selected_ids.append(IDs_in_bin[0])
                    if len(selected_ids) > 1:
                        self.get_distance(selected_ids[-2], selected_ids[-1])
                    continue
                elif i == 0:  # Initial bin
                    if pick_lowest_energy:
                        selected_ids.append(IDs_in_bin[0])
                    else:
                        # Pick one ID randomly in bin
                        ID = rng.choice(IDs_in_bin, size=1)[0]
                        index_in_bin = list(IDs_in_bin).index(ID)
                        selected_ids.append(ID)
                        self.print(('Structure {} selected randomy from '
                                    'first energy bin of size {}.').format(
                            IDs_in_bin[index_in_bin], bin_pop), verb_th=1)
                elif pick_highest_energy and i == len(hist) - 1:
                    print('i = {} out of {}, bin_pop = {}'.format(i, len(hist), bin_pop))
                    print('IDs_in_bin = {}'.format(IDs_in_bin))
                    selected_ids.append(IDs_in_bin[-1])
                    self.print(('Highest-energy structure {} selected from '
                                'last energy bin of size {}.').format(IDs_in_bin[-1],
                        bin_pop), verb_th=1)
                    if len(selected_ids) > 1:
                        self.get_distance(selected_ids[-2], selected_ids[-1])
                else:
                    # pick max_calc_per_structure structures in bin or shuffle if
                    # bin size <= max_calc_per_structure
                    picked_ids = rng.choice(IDs_in_bin, replace=False,
                        size=min(max_calc_per_structure, len(IDs_in_bin)))

                    _d = np.zeros(len(picked_ids))
                    for k, picked_id in enumerate(picked_ids):
                        _d[k] = self.get_distance(picked_id, selected_ids[-1])
                    selected_ids.append(picked_ids[np.argmax(_d)])

            else:  # the current bin (with index i) is empty
                if i < nb_of_bins - 1:
                    # append lowest-energy structure from the first non-empty bin above
                    incr = 1
                    while i + incr < len(hist):
                        IDs_in_next_bin = [ID for ID in sorted_IDs[bin_indexes == i + incr]
                                           if ID not in selected_ids]
                        if len(IDs_in_next_bin):
                            selected_ids.append(IDs_in_next_bin[0])
                            break
                        incr += 1
                    self.print('No occupied bins have been found above the current one. '
                               'Check final number of structures.', verb_th=1)
                elif i > 0:
                    # append not-yet-picked highest-energy structure from bin below
                    incr  = 1
                    while i - incr >= 0:
                        IDs_in_next_bin = [ID for ID in sorted_IDs[bin_indexes == i - incr]
                                           if ID not in selected_ids]
                        if len(IDs_in_next_bin):
                            selected_ids.append(IDs_in_next_bin[-1])
                            break
                        incr += 1
                else:
                    # DEBUGGING
                    print('Problem ?')
                    print('i = {} ; bin_pop = {} ; struct_index = {}'.format(
                        i, bin_pop, struct_index))

        result = (selected_ids, seed)

        # TODO: enforce the exact number of structures by eliminating the structure with
        # the shortest 2 distances to others (because sortest distance yields 2 structures)

        if self.verbosity >= 2:
                self.print('Calculating distances between selected structures:\n {}'.format(
                    selected_ids), verb_th=2)
                selected_dist_matrix = self.get_cosine_distance_matrix(selected_ids)
                self.print(selected_dist_matrix, verb_th=2)

        if show_full_distance_matrix:
            
            sorted_distance_matrix = np.take(np.take(
                self.distance_matrix, sorted_indexes, axis=0), sorted_indexes, axis=1)
            self.print('sorted_distance_matrix = \n{}'.format(sorted_distance_matrix),
                       verb_th=2)
            fig, ax = plt.subplots()
            row_indexes, col_indexes = np.where(sorted_distance_matrix != np.nan)
            distances = [sorted_distance_matrix[i, j]
                         for i, j in zip(row_indexes, col_indexes)]
            ax.scatter(sorted_E_rel[row_indexes], sorted_E_rel[col_indexes], s=5,
                       c=sorted_distance_matrix)
            ax.set(xlabel='E - min(E) (eV/atom)', ylabel='E - min(E) (eV/atom)')

            return selected_ids, seed, fig, ax
        else:
            return selected_ids, seed
    # End of select_energy_distant_structures method

    def select_distant_structures(self, n_structures, initial_selection=None,
                                  r=None, sigma=None, r_max=None, r_steps=None, 
                                  distance_method='max_min',  
                                  fitness_weight=0., n_jobs=None, seed=None, 
                                  mds_plot_sizeref=0.01, mds_plot_sizemin=3, mds_plot_opacity=0.7, 
                                  mds_random_state=None, 
                                  energy_plot_sizeref=0.03, energy_plot_sizemin=4, energy_plot_opacity=0.4):
        """
        Select distant structures, possibly with an energy penalty to 
        favor lower-energy structures
        
        TODO: use distanceTools.distanceMatrixData.select_distant_structures with custom 
        initial_selection and enthalpy by atom. Make distance_matrix is stored and passed 
        if existing.

        Args:
            n_structures: int
                Number of structures to select. 
            initial_selection: list, str or None (default is None)
                List of preselected structure IDs or pre-selection mode, 
                including:
                    * 'good_structures': the best 10 structures in goodPOSCARS
                    * 'best_structure': the best structure (according to fitness)
                    * 'lowest_energy': the lowest_energy structure
                If None, the first structure is chosen randomly.
            distance_method: str (default is 'max_min')
                Choose method between 'max_average' (maximize global distance to all others at each step) 
                or "max_min" (maximize distance to closest strutcure at each step).
            fitness_weight: float (default is 0.)
                Weight of the fitness penalty between 0 (no penalty, the default, in which 
                case only distances matter) to 1 in which case distance will not even matter.  
                (1 - w_F) * (dist_average) + w_F * (1 - ((F - F_min) / (F_max-F_min))
                Values closer to 1 will favor good structures (i.e. stable structures if fitness 
                relates to energy).
            n_jobs: int (default is 1)
                Number of processors used to (re)calculate the full distance matrix.
            mds_plot_sizeref: float (default is 0.01)
                Marker size (diameter) factor reflecting relative energies in the MDS plot. 
            mds_plot_sizemin: int (default is 3)
                Minimum marker size (diameter) reflecting relative energies in the MDS plot.
            mds_plot_opacity: float (default is 0.7)
                Marker opacity in MDS plot.
            energy_plot_sizeref: float (default is 0.03)
                Marker size (diameter) factor reflecting distances in the energy plot.
            energy_plot_sizemin: int (default is 4)
                Minimum marker size (diameter) reflecting distances in the energy plot.
            energy_plot_opacity: float (default is 0.4)
                Marker opacity in the energy plot.

        Returns:
            results: dict
                dictionary containing information in the selected structures
            energy_plot_fig: plotly figure object
                Figure of the energy plot use energy_plot_fig.update_layout() or 
                energy_plot_fig.update_traces() to modify interactively.
            mds_plot_fig: plotly figure object
                Figure associated with the MDS plot.
        """
        if not n_jobs:
            n_jobs = cpu_count()

        updated_dist_param = False
        if r or sigma or r_max or r_steps:
            self.set_distance_parameters(r, sigma, r_max, r_steps)
            updated_dist_param = True

        # Test whether full matrix exists
        if updated_dist_param or not self.is_full_distance_matrix():
            self.print((f"Calculating full distance matrix with r_max = {self.distance_data.Rmax} and "
                        f"sigma = {self.distance_data.sigma}, please wait..."), verb_th=1)
            self.calculate_cosine_distance_matrix(n_jobs=n_jobs)

        # Initialize results dict
        results = {
            'fitness_weight': fitness_weight
        }

        # Process initial_selection:
        #   str: 'good_structures', 'best_structure', lowest_energy_structure'
        #   list of IDs (rather than indexes)
        if isinstance(initial_selection, (list, tuple, np.ndarray)):
            selected_ids = initial_selection
        elif isinstance(initial_selection, int):
            selected_ids = [initial_selection]
        elif isinstance(initial_selection, str):
            good = ['good', 'good_structures', 'good structures']
            best = ['best', 'best_structure', 'best structure']
            lowest_E = ['lowest-e', 'lowest_energy', 'lowest-energy', 'lowest energy']
            allowed_initial_selections = good + best + lowest_E
            if initial_selection.lower() in good:
                selected_ids = self.get_good_structures_ids()
            elif initial_selection.lower() in best:
                selected_ids = [self.get_good_structures_ids()[0]]
            elif initial_selection.lower() in lowest_E:
                selected_ids = [self.get_IDs_sorted_by(sort_by='enthalpy')[0]]
            else:
                raise ValueError(f"Allowed initial_selection string values are (case unsensitive) "
                                 f"{allowed_initial_selections}.")
        elif not initial_selection:
            # Initialize sequence, save seed in results
            ssq = np.random.SeedSequence(seed)
            rng = np.random.default_rng(ssq)
            seed = ssq.entropy
            print('Random number generation:\nSeed = {}'.format(seed))
            # Make sure seed is returned
            results['seed'] = seed
            # Randomly pick one ID
            selected_ids = [rng.choice(self.IDs)]
        else:
            raise TypeError(f"initial_selection of type {type(initial_selection)} rather than "
                            f"allowed list, tuple, numpy ndarray, int or str types.")

        if distance_method in ['maximum_average', 'max_average', 'max_av']:
            dist_mthd_str = 'maximum average distance to all picked structures'
        elif distance_method in ['maximum_minimum', 'maxmin', 'max_min']:
            dist_mthd_str = 'maximum distance to closest picked structure'
        else:
            ValueError(f'{distance_method} not among allowed values (e.g. max_av)')

        if fitness_weight >= 0 and fitness_weight <= 1:
            w_F = fitness_weight
        else:
            raise ValueError(f"fitness_weight should be between 0 and 1.")

        # Make sure that selected IDs exist in self.IDs
        for id in selected_ids:
                if id not in self.IDs: 
                    raise ValueError(f"ID {id} is not among listed IDs.")

        # Create sets of all, selected and remaining indexes
        all_indexes = set(np.arange(len(self.IDs)))
        selected_indexes = set(self.get_indexes_from_IDs(selected_ids))
        remaining_indexes = all_indexes - selected_indexes

        sorted_ids = list(self.get_IDs_sorted_by(sort_by="fitness"))

        # Initialize lists in results based on selected ids 
        results['structure_ids'] = list(selected_ids)
        results['structure_indexes'] = list(selected_indexes)
        results['structure_indexes_by_fitness'] = [sorted_ids.index(id) for id in selected_ids]
        results['structure_rank_by_fitness'] = [i + 1 for i in results['structure_indexes_by_fitness']]
        results['distance_contributions'] = [None] * len(selected_ids)
        results['fitness_contributions'] = [None] * len(selected_ids)
        
        def normalize(myarray):
            min_ar = np.min(myarray)
            max_ar = np.max(myarray)
            normalized_array = (myarray - min_ar) / (max_ar - min_ar)
            return normalized_array

        while 1:
            if len(selected_indexes) >= n_structures:
                self.print(f"The target number of selected structures ({len(selected_indexes)} out "
                           f"of {n_structures}) has been reached. exiting.")
                break
            
            # Convert sets to arrays
            remaining_indexes_ar = np.array(list(remaining_indexes), dtype=int)
            selected_indexes_ar = np.array(list(selected_indexes), dtype=int)

            # Compute average (or min) distance to already-selected structures
            D = np.take(np.take(self.distance_matrix, remaining_indexes_ar, axis=0), 
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
            fitness_contrib = normalize(self.fitnesses[remaining_indexes_ar]) 

            # Find index maximizing average distance - fitness contribution
            # Term to maximize -> high w_F should favor low-fitness/energy structures
            best_index_in_remaining = np.argmax((1 - w_F) * dist_contrib + w_F * (1 - fitness_contrib))
            best_index = remaining_indexes_ar[best_index_in_remaining]
            best_id = self.IDs[best_index]
            index_by_fitness = sorted_ids.index(best_id) + 1 # one-based
            results['structure_ids'].append(best_id) 
            results['structure_indexes'].append(best_index)
            results['structure_indexes_by_fitness'].append(index_by_fitness)
            results['structure_rank_by_fitness'].append(index_by_fitness + 1)
            results['distance_contributions'].append(dist_contrib[best_index_in_remaining]) 
            results['fitness_contributions'].append(fitness_contrib[best_index_in_remaining])
            self.print(f"Structure ID {best_id} (index {best_index}, ranked {index_by_fitness + 1}) "
                       f"with a {dist_mthd_str} of {dist_contrib[best_index_in_remaining]:.3f} "
                       f"and a normalized fitness contribution "
                       f"of {fitness_contrib[best_index_in_remaining]:.3f}", 
                       verb_th=2)
            
            selected_indexes.add(best_index)
            remaining_indexes.remove(best_index)

        # Compute average (or min) distance to all selected structures
        selected_indexes_ar = np.array(list(selected_indexes), dtype=int)
        D = np.take(np.take(self.distance_matrix, selected_indexes_ar, axis=0), 
                    selected_indexes_ar, axis=1)
        if distance_method.lower() in ['maximum_average', 'max_average', 'max_av']:
            dist_str = 'average_dist_to_other_selected'
            results[dist_str] = [np.mean(np.delete(D[i], i)) for i in range(D.shape[0])]
        elif distance_method.lower() in ['maximum_minimum', 'maxmin', 'max_min', 'max-min']:
            dist_str = 'average_dist_to_other_selected'
            results[dist_str] = [np.mean(np.delete(D[i], i)) for i in range(D.shape[0])]

        # store relative energies per atom of selected structures in results
        E_rel = self.enthalpies / self.numbersOfAtoms
        E_rel = E_rel - np.min(E_rel)
        results['relative_energies_per_atom'] = E_rel[selected_indexes_ar].tolist()

        ranks_by_fitness = self.get_ranks_from_IDs(self.IDs, one_based=True)

        d = []
        x = []
        y = []
        struct_1_ids = []
        struct_2_ids = []
        struct_1_ranks_by_fitness = []
        struct_2_ranks_by_fitness = []
        for i in selected_indexes_ar:
            for j in selected_indexes_ar:
                d.append(self.distance_matrix[i, j])
                x.append(E_rel[i])
                y.append(E_rel[j])
                struct_1_ids.append(self.IDs[i])
                struct_2_ids.append(self.IDs[j])
                struct_1_ranks_by_fitness.append(ranks_by_fitness[i])
                struct_2_ranks_by_fitness.append(ranks_by_fitness[j])

        energy_plot_fig = px.scatter(x=x, y=y, size=normalize(d), color=d, 
                           labels={'x': "E - min(E) (eV/atom)", "y": "E - min(E) (eV/atom)"}, 
                           template='simple_white', 
                           title=(f"{self.uspex_run_name} CSP run: {len(selected_indexes)} structures "
                                  f"selected based on {dist_mthd_str}<br>"
                                  f"with a fitness-penalty weight of {fitness_weight}."), 
                            hover_data={"x Structure ID": struct_1_ids, 
                                        "y Structure ID": struct_2_ids, 
                                        "x Structure rank": struct_1_ranks_by_fitness, 
                                        "x Structure rank": struct_2_ranks_by_fitness}) 

        energy_plot_fig.update_layout(xaxis=dict(constrain='domain'), 
                            yaxis=dict(scaleanchor="x", scaleratio=1))
        
        # Customize symbol size range if needed
        energy_plot_fig.update_traces(
            marker=dict(sizemode='diameter', sizeref=energy_plot_sizeref, sizemin=energy_plot_sizemin, 
                        opacity=energy_plot_opacity),
            selector=dict(mode='markers')
        )
        energy_plot_fig.show()

        results_df = DataFrame(results)
        if self.verbosity >= 1:
            print(f"distance-based structure selection is done:\n{results_df}")


        # Compute MDS
        self.print("Computing multi-dimensional scaling (MDS)...")
        mds = MDS(n_components=2, dissimilarity='precomputed', 
                  metric=True, n_jobs=n_jobs, random_state=mds_random_state)
        coords = mds.fit_transform(self.distance_matrix)

        relative_energies = self.enthalpies / self.numbersOfAtoms
        relative_energies /= min(relative_energies)

        # Create a DataFrame for Plotly
        df = DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'ID': self.IDs, 
            'normalized_fitness': normalize(self.fitnesses),
            'rank_by_fitness': ranks_by_fitness, 
            'relative_energy': relative_energies, 
            'is_selected': ['Selected' if i in selected_indexes else 'Unselected' 
                            for i in range(len(self.IDs))]
        })

        # TODO: plot in matrix (x=y=index_by_fitness)

        # Enforce cnostant ordering of symbol and colr sequence based on is_selected
        category_order = ["Unselected", "Selected"]

        # Plot with Plotly Express
        mds_plot_fig = px.scatter(
            df,
            x='x',
            y='y',
            size='normalized_fitness',  # Control size by property
            color='is_selected',  # Color by selection status
            symbol='is_selected',  # Symbol by selection status
            title=(f"{self.uspex_run_name} CSP run: {len(selected_indexes)} structures "
                   f"selected based on {dist_mthd_str}<br>"
                   f"with a fitness-penalty weight of {fitness_weight}."),
            labels={'x': 'MDS Dimension 1', 'y': 'MDS Dimension 2'},
            hover_data=['ID', 'normalized_fitness', 'rank_by_fitness', 'relative_energy'], 
            template='simple_white', 
            category_orders={"is_selected": category_order}
        )

        # Customize symbol size range if needed
        mds_plot_fig.update_traces(
            marker=dict(sizemode='diameter', sizeref=mds_plot_sizeref, sizemin=mds_plot_sizemin, 
                        opacity=mds_plot_opacity),
            selector=dict(mode='markers')
        )
        
        self.print("Opening plotly (via browser)...")
        mds_plot_fig.show()

        return results, energy_plot_fig, mds_plot_fig


    def get_good_structures_ids(self):
        """
        Get the IDs of structures listed in the goodStructure file

        This file in principle the non-redundant list of best 10 structures

        Returns:
            A list of id numbers.
        """
        array = np.genfromtxt(os.path.join(self.resultsDirectoryName, 'goodStructures'),
                              skip_header=2)
        ids = [int(row[0]) for row in array]
        return ids

    def get_good_structures_average_density(self, return_std:bool=False):
        """
        Get the average density (g.cm-3) of the best structures listed in the goodStructures file

        Args:
            return_std: bool (default is False)
                Wheter the deviation of densities over structures in goodStructures file
                shall be returned along with the average density.
        Returns:
            density of return_std is False, a tuple of density and density_std otherwise.
        """
        structures = self.get_structures_from_IDs(self.get_good_structures_ids())
        density = np.mean([s.density for s in structures])
        std_string = None
        if return_std:
            density_std = np.std([s.density for s in structures])
            std_string = ' +/- {:.3f}'.format(density_std)
        self.print('Average density of structures in goodStructures file is {:.3f}{} g.cm-3'.format(
            density, std_string), verb_th=1)
        if return_std:
            return density, density_std
        else:
            return density

    def get_good_structures_average_volume(self, return_std:bool=False):
        """
        Get the average volume (A^3) of the best structures listed in the goodStructures file

        Args:
            return_std: bool (default is False)
                Wheter the deviation of volumes over structures in goodStructures file
                shall be returned along with the average volume.
        Returns:
            volume of return_std is False, a tuple of volume and volume_std otherwise.
        """
        structures = self.get_structures_from_IDs(self.get_good_structures_ids())
        volume = np.mean([s.volume for s in structures])
        std_string = None
        if return_std:
            volume_std = np.std([s.volume for s in structures])
            std_string = ' +/- {:.2f}'.format(volume_std)
        self.print('Average volume of structures in goodStructures file is {:.2f}{} A^3'.format(
            volume, std_string), verb_th=1)
        if return_std:
            return volume, volume_std
        else:
            return volume

    def get_structure_IDs_from_property(self, filter_dict):
        """
        filter_dict example:
        {'Thickness/Volume/Enthalpy' : {'lt/gt/le/ge/in': value/list}}

        WARNING: CASE SENSITIVE
        """
        def test(value, operator, filter_value):

            if operator.lower() in ['<', 'lt']:
                test = True if value < filter_value else False
            elif operator.lower() in ['<=', 'le']:
                test = True if value <= filter_value else False
            elif operator.lower() in ['>', 'gt']:
                test = True if value > filter_value else False
            elif operator.lower() in ['<=', 'le']:
                test = True if value >= filter_value else False
            elif operator.lower() in ['==', 'eq']:
                test = True if value == filter_value else False
            elif operator.lower() == 'in':
                test = True if value in filter_value else False
            else:
                raise ValueError('Unkown selection operator')
            self.print('Testing {} {} {}: {}'.format(value, operator, filter_value, test), verb_th=3)
            return test

        available_properties = list(self.individuals_dict.keys())
        # TODO: add volume or density in case they are not in individuals_dict
        filter_properties = list(filter_dict.keys())
        filtered_IDs = set(self.IDs)
        for filter_property in filter_properties:
            if filter_property in available_properties:
                for operator, filter_value in filter_dict[filter_property].items():
                    matching_IDs = [self.IDs[i] for i, v in
                                    enumerate(self.individuals_dict[filter_property])
                                    if test(v, operator, filter_value)]
                    self.print('matching_IDs = {}'.format(matching_IDs), verb_th=3)
                    filtered_IDs = filtered_IDs.intersection(matching_IDs)

        self.print('{} structure IDs have been selected.'.format(len(filtered_IDs)), verb_th=1)

        filtered_IDs = list(filtered_IDs)
        filtered_IDs.sort()
        self.print(filtered_IDs, verb_th=2)

        return filtered_IDs

    def get_input_folder(self):
        return os.path.split(os.path.abspath(self.resultsDirectoryName))[0]

    def export_as_seed_or_anti_seed(self, structureIDs, destination='Seeds',
                                    output_basename='POSCARS', write_mode='a',
                                    use_initial=False):
        """
        Copy POSCAR files selected by IDs to (Anti)Seeds/POSCARS

        Args:
            structureIDs: int, list, tuple
                structure ID(s)
            destination: str (default is Seeds)
                Use selected structures as Seeds or AntiSeeds (non-case-sensitive)
            output_basename: str (default: 'POSCARS')
                Relative name of the output file
            write_mode: str (default is 'a')
                Write mode of the output file ('a' to append, 'w' to overwrite)
            use_initial: bool (default is False)
                Whether initial rather than relaxed structures should be used.

        Returns:
            destination_file: str
                absolute output file path
        """
        if 'antiseed' in destination.lower():
            destination_folder_basename = 'AntiSeeds'
        elif 'seed' in destination.lower():
            destination_folder_basename = 'Seeds'

        poscars = self.get_poscars_from_IDs(structureIDs, use_initial=use_initial)

        destination_file = os.path.join(self.get_input_folder(), destination_folder_basename, output_basename)
        with open(destination_file, write_mode) as f:
            for poscar in poscars:
                # Need to remove symbols at the end of coordinate lines
                poscar_lines = poscar.get_str(significant_figures=8).splitlines()
                for index, line in enumerate(poscar_lines):
                    if index >= 8 and len(line.split()) > 3:
                        poscar_lines[index] = ''.join(['{:>15}'.format(s) for s in line.split()[:3] ])
                f.writelines([line + '\n' for line in poscar_lines])

        self.print('{} POSCAR files have been written to file {}.'.format(
            len(poscars), destination_file), verb_th=1)

        return destination_file

    def get_energy_difference_matrix(self, selected_indexes: list=None, 
                                     selected_ids:list =None, 
                                     use_absolute_differences=False):
        """
        Docstring for plot_energy_differences_vs_distances
        
        Args:
            selected_indexes: list (default is None)
                Selected structure indexes
            selected_ids: list (default is None)
                Selected structure IDs
            use_absolute_differences: bool (default is False)
                Whether absolute energy differences  should be computed
                
        Returns:
            energy_difference_matrix: numpy 
                
        """
        if selected_indexes is None:
            if selected_ids is None:
                selected_indexes = np.arange(len(self.IDs))
            else:
                selected_indexes = self.get_indexes_from_IDs(selected_ids)   

        n_indexes = len(selected_indexes)
        energy_difference_matrix = np.zeros((n_indexes, n_indexes))
        for i in  selected_indexes:
            for j in selected_indexes:
                if j > i:
                    break
                elif i == j:
                    energy_difference_matrix[i, j] = 0
                else:
                    energy_difference_matrix[i, j] = self.enthalpies[j] / self.numbersOfAtoms[j] \
                                                     - self.enthalpies[i] / self.numbersOfAtoms[j]
        
                    energy_difference_matrix[j, i] = - energy_difference_matrix[i, j]

        if use_absolute_differences:
            energy_difference_matrix = np.abs(energy_difference_matrix)

        return energy_difference_matrix


    def plot_energy_differences_vs_distances(self, selected_indexes: list=None, 
                                             selected_ids:list =None):
        """
        Docstring for plot_energy_differences_vs_distances
        
        Args:
            selected_indexes: list (default is None)
                Selected structure indexes
            selected_ids: list (default is None)
                Selected structure IDs
        Returns:
            fig: plotly figure
            df: pandas DataFrame
                Dataframe containing pairwise structure indexes and ids, 
                energy differences, etc. 
        """
        energy_difference_matrix = self.get_energy_difference_matrix()
        
        if not self.is_full_distance_matrix():
            self.calculate_cosine_distance_matrix()

        if selected_indexes is None:
            if selected_ids is None:
                selected_indexes = np.arange(len(self.IDs))
            else:
                selected_indexes = self.get_indexes_from_IDs(selected_ids)        
        
        n_indexes = len(selected_indexes)
        
        ranks_by_fitness = self.get_ranks_from_IDs(self.IDs[selected_indexes])

        flat_data = {
            "absolute_energy_difference": [], 
            "distance": [], 
            "pair_indexes":[], 
            "pair_ids": [], 
            "pair_ranks_by_fitness": [], 
        }
        count = 0
        for i in range(n_indexes):
            for j in range(i+1):
                flat_data["absolute_energy_difference"].append(np.abs(energy_difference_matrix[i, j]))
                flat_data["distance"].append(self.distance_matrix[i, j])
                flat_data["pair_indexes"].append((i, j))
                flat_data["pair_ids"].append((self.IDs[i], self.IDs[j]))
                flat_data["pair_ranks_by_fitness"].append((ranks_by_fitness[i], ranks_by_fitness[j]))

        df = DataFrame(flat_data)
        fig = px.scatter(df, x="absolute_energy_difference", y="distance", opacity=0.5, 
                         hover_data=["pair_indexes", "pair_ids", "pair_ranks_by_fitness"], 
                         template="simple_white", 
                         title=f"{self.uspex_run_name} - pairwise energy differences vs distances.")
        fig.update_traces(marker={'sizemode': 'diameter', 'size': 4})

        fig.show()

        return fig, df

# end of class uspexStructuresData
