# -*- coding: utf-8 -*-
"""
A module dedicated to the manipulation of data generated with supercell program

Supercell program : a combinatorial structure-generation approach for the 
    local-level modeling of atomic substitutions and partial occupancies in
    crystals.
    https://orex.github.io/supercell/
    Article (open access) : 
    Okhotnikov, K., Charpentier, T., & Cadars, S. (2016), Journal of
    Cheminformatics, 8(1), 17.
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0129-3

@author: Sylvian Cadars, Institut de Rechrche sur les Céramiques, 
         Université de Limoges, CNRS, France
"""
import os.path
import numpy as np
import glob

class supercellData():
    """
    A class to manipulate data from the supercell program
    
    Supercell program : a combinatorial structure-generation approach for the 
    local-level modeling of atomic substitutions and partial occupancies in
    crystals.
    https://orex.github.io/supercell/
    Article (open access) : 
    Okhotnikov, K., Charpentier, T., & Cadars, S. (2016), Journal of
    Cheminformatics, 8(1), 17.
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0129-3
    
    
    """
    
    availableSelectionMethods = ['r','f','a','l','h']
    
    def __init__(self,structuresDirectory=''):
        print('Initializing supercellData instance')
        self.structuresDirectory = structuresDirectory
        self.coulombEnergyFiles = []
        self.IDs = np.asarray([],dtype=int)
        self.IDstrings = []
        self.weights = np.asarray([],dtype=int)
        self.E_C_eV = np.asarray([],dtype=float)
        self.selectionMethods = [] # 'l','r','h', etc.
        self.cifFileNames = []
        
    def set_from_all_cif_files(self,structuresDirectory=''):
        """
        Set parameters in structureData instance from all cif files contained in a directory.

        Parameters :
            structuresDirectory (OPTIONAL) : default is ''

        Returns :
            None.
        """
        if os.path.isdir(structuresDirectory) :
            self.structuresDirectory = structuresDirectory
        elif structuresDirectory != '' :
            raise ValueError(f'{structuresDirectory} is not a directory.')
        self.cifFileNames = glob.glob(os.path.join(self.structuresDirectory,'*.cif')) 
        self.set_IDs_and_weights_from_cif_file_names(setSelectionMethod=True)

    def find_coulomb_energy_files(self):
        """
        Find supercell Coulomb-energy output files <prefix>_coulomb_energy_<selectionMethod>.txt

        Returns
        -------
        None.

        """
        self.coulombEnergyFiles = glob.glob(os.path.join(
            self.structuresDirectory,'*_coulomb_energy_*.txt'))
    
    
    def read_coulomb_energy_file(self,fileName:str,reset=False):
        """
        Read and assign supercellData from coulomb energy file

        All structures in list and corresponding information will be appended
        to existing
        
        Parameters
        ----------
        fileName : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if reset == True:
            # Re-initialize data
            self.IDs = np.asarray([],dtype=int)
            self.weights = np.asarray([],dtype=int)
            self.E_C_eV = np.asarray([],dtype=float)
            self.selectionMethods = [] # see self.availableSelectionMethods
            self.cifFileNames = [] 
        
        selectionMethod=self.get_selection_method_from_coulomb_file_name(fileName)
        tableData = np.genfromtxt(fileName, skip_header=0, dtype=None, \
                                  names=['cif','E_C','E_C_unit'],encoding=None)
        try :
            for row in tableData :
                # Read and assign cif file name
                self.selectionMethods.append(selectionMethod)
                self.cifFileNames.append(os.path.basename(row[0]))
                # TODO : extract structure ID from cif file name
                self.E_C_eV = np.append(self.E_C_eV,row[1])
                _,_,IDstring,weight = self.get_info_from_cif_file_name(
                    self.cifFileNames[-1])
                self.IDs = np.append(self.IDs,int(IDstring))
                self.IDstrings.append(IDstring)
                self.weights = np.append(self.weights,weight)
        except TypeError :  # Possibly only one line in file 
            if len(tableData.shape) == 0 :
                self.selectionMethods.append(selectionMethod)
                self.cifFileNames.append(os.path.basename(
                    tableData['cif'].item()))
                self.E_C_eV = np.append(self.E_C_eV,tableData['E_C'].item())
                _,_,IDstring,weight = self.get_info_from_cif_file_name(
                    self.cifFileNames[-1])
                self.IDs = np.append(self.IDs,int(IDstring))
                self.IDstrings.append(IDstring)
                self.weights = np.append(self.weights,weight)
        
        # TODO : deal with other possible exceptions             
    
    # end of method read_coulomb_energy_file

    def get_selection_method_from_coulomb_file_name(self,coulombFileName : str):
        """
        Get the selection method from <prefx>_coulomb_energy_<method>.txt supercell files

        Parameters
        ----------
        coulombFileName : str
            DESCRIPTION.

        Returns
        -------
        selectionMethod : str
            selection method character ('r','l','a','h,', etc.) : see
            self.availableSelectionMethods

        """
        if 'coulomb_energy_l.txt' in coulombFileName :
            selectionMethod = 'l'
        elif 'coulomb_energy_h.txt' in coulombFileName :
            selectionMethod = 'h'
        elif 'coulomb_energy_r.txt' in coulombFileName :
            selectionMethod = 'r'
        # TODO : add other selection methods here
        else :
            print('Supercell structure selction method not detexted from file : ',
                  coulombFileName)
            selectionMethod = 'unknown'
        return selectionMethod
    
    def get_common_cif_files_prefix(self):
        """
        A method to get the prefix from the list of cif file names

        Returns
        -------
        prefix : str
            Common prefix f cif files without the '_' character before the
            selection method character ('r','h,', etc. if any) and structure
            ID.

        """
        prefix = os.path.commonprefix(self.cifFileNames)
        if prefix[-2:] == '_i':
            prefix = prefix[:-2]
        return prefix

    def set_IDs_and_weights_from_cif_file_names(self,setSelectionMethod=False):
        """
        Set IDs, weights and optionaly selectionMethods properties from cifFileNames

        Parameters
        ----------
        setSelectionMethod : bool, optional
            decides whether the selection methods should be set.
            The default is False.

        Returns
        -------
        None. supercellData object modified in place.

        """
        
        self.weights = np.ones(len(self.cifFileNames))
        self.IDs = np.zeros(len(self.cifFileNames))
        self.IDstrings = []
        if setSelectionMethod == True:
            self.selectionMethods = []
        for index,cif in enumerate(self.cifFileNames):
            info_dict = self.get_info_from_cif_file_name(cif)
            self.IDs[index] = int(info_dict['IDstring'])
            self.IDstrings.append(info_dict['IDstring'])
            if setSelectionMethod == True:
                self.selectionMethods.append(info_dict['selectionMethod'])
        
    def get_info_from_cif_file_name(self,cifFileName):
        """
        A method to extract information in supercell program cif output files
        
        cifFileName should be structured as 
        <prefix>_i<selmethod><ID>_w<weight>.cif
        
        returns :
            dict containing the following items :
                'prefix' : str
                    ADD DESCRIPTION
                'selectionMethod' : str
                    'r','f','a','l' or 'h', see self.availableSelectionMethods
                'IDstring' : str
                    structure ID as a string (including zeros). Use int(IDstring)
                    to get ID.
                'weight' : int
                    structure weight calculated based on structure symmetry
                    (defaults to one if 'merge' supercell option was not used)
        """
        cifFileName = os.path.basename(cifFileName)
        # Initialize output_dict with default values
        output_dict = {
            'selectionMethod' : 'a', 
            'weight' : 1
        }

        # Check and remove .cif extension
        list1 = os.path.splitext(cifFileName)
        if list1[1] != '.cif' :
            print('WARNING : in function get_ID_and_weight_from_cif_file_name.',
                  'Extension of file : ',cifFileName,' is not cif.')
        str1 = list1[0]
        # Read and discard discard '_w<weight>' (if present)
        list1 = str1.split('_w')
        if len(list1) >= 2:
            output_dict['weight'] = int(list1[-1])
        str1 = '_w'.join(list1[:-1])
        # Read and discard '_i<selectionMethod><prefix>'
        list1 = str1.split('_i')
        str2 = list1[-1]
        if str2[0] in self.availableSelectionMethods :
            output_dict['selectionMethod'] = str2[0]
            str2 = str2[1:]
        if str2.isnumeric() == True :
            output_dict['IDstring'] = str2
        else :
            print('WARNING : ID could not be read from cif file :',cifFileName)
            output_dict['IDstring'] = ''
        output_dict['prefix'] = '_i'.join(list1[:-1])
        
        return output_dict
    
    
    def get_indexes_from_selection_method(self,selectionMethod:str) :
        
        if selectionMethod not in self.availableSelectionMethods :
            print('WARNING : unknown supercell-program selection method.')
            indexes = []
        indexes = [i for i,sm in enumerate (self.selectionMethods) \
                   if sm == selectionMethod]
        
        return indexes

    def get_indexes_from_IDs(self,IDs):
        indexes = [i for i,ID in enumerate (self.IDs) if ID in IDs]
        return indexes

    def get_selection_method_long_name(self,indexOrMethodShortName) :
        
        if isinstance(indexOrMethodShortName,int) :
            [index] = self.get_indexes_from_IDs([indexOrMethodShortName])
            selMethod = self.selectionMethods[index]
        elif isinstance(indexOrMethodShortName,str) :
            if indexOrMethodShortName in self.availableSelectionMethods :
                selMethod = indexOrMethodShortName
            else :
                print('WARNING : unknown semection method : ',indexOrMethodShortName)
        else :
            print('WARNING : indexOrMethodShortName argument to method ',
                  'get_selection_method_full_name should be an index or a string')
        if selMethod == 'r':
            methodLongName = "random"
        elif selMethod == 'l':
            methodLongName = "lowest-E_C"
        elif selMethod == 'h':
            methodLongName = "highest-E_C"
        elif selMethod == 'f':
            methodLongName = "first structures"
        elif selMethod == 'a':
            methodLongName = "last structures"
        
        return methodLongName
            
# End of class supercellData
