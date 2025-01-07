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
import matplotlib.pyplot as plt
import pickle
import structureComparisonsPkg.distanceTools as dt
import json


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
    def __init__(self, fileOrDirectoryName = '',selectedStructures = 'all',
                 extractedPOSCARBaseName='ID-', extractedPOSCARExtention='.vasp',
                 r=None, sigma=0.02, r_max=8.0, r_steps=512, verbosity=1):

        self.verbosity = verbosity

        # Initialization of distance matrix data
        self.distance_data  = dt.distanceMatrixData(R=r, sigma=sigma,
            Rmax=r_max, Rsteps=r_steps)

        self.extractedPOSCARBaseName =  extractedPOSCARBaseName
        self.extractedPOSCARExtention = extractedPOSCARExtention

        if len(fileOrDirectoryName) > 0 :
            #TODO add a try on selectedStructures
            if selectedStructures == 'all' :
                self.load_uspex_structure_data(fileOrDirectoryName)
            elif selectedStructures == 'best' :
                print('selectedStructures = ',selectedStructures)
                #TODO : read only best structures from file
            else :
                errorMsg = 'Invalid input parameter selectedStructures = ' + selectedStructures + ' for class uspexStructuresData.'
                raise ValueError(errorMsg)
        else :
            print('Empty uspexDataManipulation.uspexStructuresData object created. Load data with class method load_uspex_structure_data.')

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
        structures = get_structures_from_IDs(self.IDs)
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

    # Step 1 : read energies and structure IDs from USPEX output files ()
    # characteristics of the different structures (structures IDs, enthalpies and possibly other parameters) should read from file Individuals
    def load_uspex_structure_data_old(self,fileOrDirectoryName,**kwargs) :
        """
        Reading structure IDs, enthalpies, generation number, volumes, etc. from Individuals file

        OBSOLETE. TO BE REMOVED. USE load_uspex_structure_data instead.

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

        print('Reading uspex-structure data from file : ',fileName)
        # tableData = np.genfromtxt(fileName, skip_header=2, dtype=None, encoding=None)
        # Does not work because '[' characters may or may not be attached other characters.
        # Proceed line by line and split lines between parts outside and inside the brackets
        # print('tableData = ',tableData)

        file1 = open(fileName, 'r')
        count = 0
        lines = []
        while True:
            count += 1
            # Get next line from file
            line = file1.readline()
            # if line is empty end of file is reached
            if not line:
                break
            lines.append(line)
        file1.close()

        #TODO : check and skip header (2 lines). Remove empty lines ?

        # read Number of atom types and species of each type from first table row
        # rowAsList = list(tableData[0])
        # self.nbOfAtomTypes = rowAsList.index(']') - rowAsList.index('[') - 1
        # print('self.nbOfAtomTypes = ',self.nbOfAtomTypes)

        nbOfHeaderLines = 2
        self.nbOfStructures = len(lines)-nbOfHeaderLines
        print('self.nbOfStructures = ',self.nbOfStructures)

        self.generationNumbers = np.empty(self.nbOfStructures,dtype=int)
        self.IDs = np.empty(self.nbOfStructures,dtype=int)
        self.creationMethods = []
        self.numbersOfAtomTypes = np.empty(self.nbOfStructures,dtype=int)
        self.numbersOfAtomsOfEachType = []
        self.numbersOfAtoms = np.empty(self.nbOfStructures,dtype=int)
        self.enthalpies = np.empty(self.nbOfStructures)
        self.volumes = np.empty(self.nbOfStructures)
        self.densities = np.empty(self.nbOfStructures)
        self.fitnesses = np.empty(self.nbOfStructures)
        self.kpoints = np.empty((self.nbOfStructures,3),dtype=int)
        self.symmGroupNb = np.empty(self.nbOfStructures,dtype=int)
        self.Q_entr = np.empty(self.nbOfStructures)
        self.A_order = np.empty(self.nbOfStructures)
        self.S_order = np.empty(self.nbOfStructures)

        for structureIndex, line in enumerate(lines[(nbOfHeaderLines):]) :
            # There is probably a much more elegant way to to this, but it works...
            # TODO : improve using str.split('[') and str.split(']')
            openBracket1Index = line.find('[')
            closeBracket1Index = line.find(']')
            openBracket2Index = line.find('[',closeBracket1Index+1)
            closeBracket2Index = line.find(']',closeBracket1Index+1)
            lineSplit1 = line[:openBracket1Index].split()
            lineSplit2 = line[(openBracket1Index+1):closeBracket1Index].split()
            lineSplit3 = line[(closeBracket1Index+1):openBracket2Index].split()
            lineSplit4 = line[(openBracket2Index+1):closeBracket2Index].split()
            lineSplit5 = line[(closeBracket2Index+1):(len(line)-1)].split()
            self.generationNumbers[structureIndex] = int(lineSplit1[0])
            self.IDs[structureIndex] = int(lineSplit1[1])
            self.creationMethods.append(lineSplit1[2])
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
            self.S_order[structureIndex] = float(lineSplit5[3])
        np.asarray(self.kpoints)
        np.asarray(self.numbersOfAtomsOfEachType)

    # end of method get_uspex_structures_data

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


    def get_structures_from_IDs(self, structureIDs, extract_poscar_files=False,
                                use_initial=False) :
        """
        Create a (list of) pymatgen Structure object(s) obtained from a crystal structure prediction run with USPEX.

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

        if not extract_poscar_files:
            return [poscar.structure for poscar in self.get_poscars_from_IDs(structureIDs)]

        missingIDs = [ID for ID in structureIDs if not os.path.exists(
                      self.get_extracted_POSCAR_file_name(ID, use_initial=use_initial))]
        if len(missingIDs) > 0 :
            self.extract_POSCARS_from_IDs(missingIDs, use_initial=use_initial)

        selectedStructures = [Structure.from_file(fileName) for fileName in
                              self.get_extracted_POSCAR_file_names(structureIDs, use_initial=use_initial)]

        return selectedStructures


    def get_structure_from_ID(self, structureID, extract_poscar_file=False,
                              use_initial=False) :
        """
        Create a pymatgen Structure object(s) obtained from a crystal structure prediction run with USPEX.

        The structure with the requested ID will be read from the gatheredPOSCARS file (the default) or alternatively from gatheredPOSCARS_order or goodStructures_POSCARS files.

        Args :
            structureID : index of considered structure. Accepts an array or list containing a single ID. A warning will be printed if an array or list of multiple elements. The function will in this case call get_structures_from_IDs and return a list of structures.

        Returns :
            selectedStructures = pymatgen.core.structure Structure class object(s) (or a list thereof, see above.)
        """

        # Convert structureID to a single element to use in class method get_structures_from_IDs and returns the first (and in principle only structure in list
        structureID = self.make_single_value_if_list_of_single_element(structureID)
        if not extract_poscar_file:
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
                    structure = self.get_structures_from_IDs(structureID, use_initial=use_initial)
            except TypeError :
                # structureID is neither an array nor a list
                structure = self.get_structures_from_IDs([structureID], use_initial=use_initial)[0]
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

    # end of method get_structures_from_IDs


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


    def get_ranks_from_IDs(self,structureIDs):
        """
        Get rank (from 1 to nbOfStrctures+1) based on fitness for selected IDs
        """
        structureIDs = self.make_list_if_single_element(structureIDs)
        # rank all structures based on fitnesses
        ranks = rankdata(self.fitnesses,method='min')
        # select structures associated with the selected IDs
        selectedRanks = ranks[self.get_indexes_from_IDs(structureIDs)]

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
            sorted_indexes = np.argsort(self.fitness)
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

    def get_structure_index_from_ID(self, ID):
        [index],  = np.where(self.IDs == ID)
        return index

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

    def select_distant_structures(self, nb_of_structures,
                                  max_calc_per_structure=10,
                                  pick_lowest_energy=True,
                                  pick_highest_energy=True,
                                  sigma=None, r_max=None, seed=None,
                                  show_full_distance_matrix=True):
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

        # Sort structures by enthalpy
        sorted_indexes = np.argsort(self.enthalpies)
        sorted_E_rel = (self.enthalpies[sorted_indexes]-np.min(self.enthalpies)
                        ) /  self.numbersOfAtoms[sorted_indexes]
        sorted_IDs = self.IDs[sorted_indexes]

        global_distance_matrix = np.zeros((len(sorted_IDs), len(sorted_IDs)))

        # split into energy segments
        hist, bin_edges = np.histogram(sorted_E_rel, bins=nb_of_structures)
        _ = np.digitize(sorted_E_rel, bin_edges, right=False)
        bin_indexes = np.where(_ == nb_of_structures + 1, nb_of_structures, _) - 1

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
                if i < nb_of_structures - 1:
                    # append lowest-energy structure from the first non-empty bin above
                    incr = 1
                    while i + incr < len(hist):
                        IDs_in_next_bin = [ID for ID in sorted_IDs[bin_indexes == i + incr]
                                           if ID not in selected_ids]
                        if len(IDs_in_next_bin):
                            selected_ids.append(IDs_in_next_bin[0])
                            break
                        incr += 1
                    print(('WARNING: No non-empty bins have been found the current one. '
                           'The final number of tructures could be lower than requested.'))
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

        self.print('Calculating distances between selected structures:\n {}'.format(
            selected_ids), verb_th=1)
        selected_dist_matrix = self.get_cosine_distance_matrix(selected_ids)
        self.print(selected_dist_matrix, verb_th=2)

        result = (selected_ids, seed)

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

# end of class uspexStructuresData

