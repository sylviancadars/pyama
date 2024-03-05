# -*- coding: utf-8 -*-
"""
Module for the calculation of atomic form factors

Created on Mon Nov 16 2020

@author: Sylvian Cadars, Institut de Recherche sur les Ceramiques, CNRS,
         Université de Limoges
"""

# Total X-ray diffusion function
# Masson O? and Thomas P., J. Appl. Cryst. (2013). 46, 461–465

# Let us consider a pair of atomic types i and j (alpha and beta in paper)

# STEP 1 : calculate f_j(Q) (complex number) for any element
# Can be obtained for tables : https://henke.lbl.gov/optical_constants/sf/h.nff 
# 
# OR (prefered) calculated from scratch for any Q table using the coefficients listed in file : 
# D:\cadarp02\Documents\Programming\Utils\Diffraction\CoefficientsForFormFactorCalculations_2020-10-05.txt
# Taken from http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
# taken in turn from : International Tables for Crystallography: http://it.iucr.org/Cb/ch6o1v0001/.
# and the expression : 
# Read a1...4, b1...4, c from file with numpy.loadtxt

# STEP 2 :
# Correct f to account for incident-energy-dependent (but Q-indendent) diffusion effects f' and f'' (or f1 and f2):
# f(Q,E) = f0(Q) + f1(E) +if2(E)
# f' (f1) and f''(f2) are independent
# These should be obtained from the following files in the Dabax files of the ESRF : 
# http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/f0_WaasKirf.dat
# http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/f1f2_Henke.dat

import numpy as np
import matplotlib.pyplot as plt
import re
import diffractionPkg.utils as dutils

class formFactorCoefficients() :
    """
    A class to describe coefficients for the calculation of atomic form factors  :
    
    properties :
        specieName :
        nbOfcoeffs : e.g. 11 in case where X = 5 (a1...a5,b1...b5,c)
        coeffNames : aX, bX, c cofficients with X depending on file 
        coeffValues : 
    """

    def __init__(self):
        self.specieName = ''
        self.coeffNames = []
        self.coeffValues = []

    def get_coefficient(self,coeffIndexOrName) :
        """    
        A method to get a coefficient value from its name or index
        
        Parameters :
            coeffIndexOrName : coeff name aX, bX with X = 1,2,...etc, or c
        
        Returns :
            coefficient value
        """
        try :
            coeff = self.coeffValues[coeffIndexOrName]
        except IndexError :
            if isinstance(coeffIndexOrName,str) :
                coeff = self.coeffValues[self.coeffNames.index(coeffIndexOrName)]
            else :
                errorMsg = 'Requested coefficient index for method ' + \
                    'get_coefficient is out of range (max = ' +  \
                        str(self.nbOfCoeffs-1) + ')'
                raise IndexError(errorMsg)
        except TypeError :
            # TODO : manage error obtained with self.get_coefficient('toto')
            try :
                coeff = self.coeffValues[self.coeffNames.index(coeffIndexOrName)]
            except ValueError :
                print('here')
                if isinstance(coeffIndexOrName,str) : 
                    errorMsg = coeffIndexOrName + ' is not in the list of ' + \
                        'available coefficients for ' + self.specieName + \
                        ' : ' + str(self.coeffNames)
                    raise ValueError(errorMsg)
                elif isinstance(coeffIndexOrName,list) or \
                   isinstance(coeffIndexOrName,tuple) :
                    coeff = self.get_coefficients_from_list(coeffIndexOrName)
                
        return coeff
    
    def get_coefficients_from_list(self,listOrTupleOfIndexesOrNames) :
        """
        A method to get values of aX, bX for a list of coeff names or indexes

        Parameters
        ----------
        listOrTupleOfIndexesOrNames : list or tuple
            List or tuple of coefficient indexes or names (aX, bX or c).

        Returns
        -------
        List of coefficient values

        """
        listOfCoeffs = []
        try : 
            [listOfCoeffs.append(self.get_coefficient(indexOrName)) \
                for indexOrName in listOrTupleOfIndexesOrNames]
        except TypeError :
            listOfCoeffs.append(self.get_coefficient(listOrTupleOfIndexesOrNames))
            
        return listOfCoeffs

class formFactor() :
    """
    A class to calculate and manipulate f0 atomic form factors
    
    Properties :
        specieName : str
           name of element or specie as designated in the original file.
        f0 : numpy.darray
            atomic form factor f0(Q)
        Q : numpy.darray
            Vector of Q values where Q = 4*pi*sin(theta)/lambda withe theta
            half the scattering angle and lambda the incident beam wavelength.
        f1 : numpy.darray
            f1(E) correction to the atomic form factor at energy(ies) self.E_eV
        f2 : numpy.darray
            f2(E) correction to the atomic form factor at energy(ies) self.E_eV
        E_eV : numpy.darray
            Array of (or single) E value(s) in eV
    """
    def __init__(self):
        self.specieName = ''
        self.Z = 0
        self.f0 = np.asarray([],dtype=float)
        self.Q = np.asarray([],dtype=float)
        self.f1 = np.asarray([],dtype=float)
        self.f2 = np.asarray([],dtype=float)
        self.E_eV = np.asarray([],dtype=float)

    def set_f0(self,**kwargs):
        if 'specieName' in kwargs :
            self.specieName = kwargs['specieName']
        [ffc] = read_coeffs_for_f0_form_factor([self.specieName])
        [self.f0],_,self.Q = calculate_form_factors_f0_from_coeffs(
            [ffc],**kwargs)
     
    def set_f1_f2(self,**kwargs):
        if 'specieName' in kwargs :
            self.specieName = kwargs['specieName']
        [dc] = read_dispersion_corrections_f1_f2_from_file(
            [self.specieName],**kwargs)
    
    def get_corrected_form_factor(self,resizeTo0D=True,**kwargs):
        """
        Calculate total atomic form factor f(Q,E) = f0(Q) + f1(E) + j*f2(E)
        
        Parameters
            resizeTo0D : boolean, OPTIONAL : default = True
                If True the f(Q,E) array will be converted to a simple f(Q)
                vector if len(self.E_eV) = 1
        
        Optional keyword parameters
            E_eV : float or list or array
                (list or array of) enery value in eV
        
        Returns :
            f : numpy array with dtype=complex
                array of dimension (len(E_eV),len(Q))
        """
        
        # change self E_eV property if value set by user
        if 'E_eV' in kwargs :
            self.E_eV = kwargs['E_eV']
        
        # Test whether self_eV is made f a sinle value, 
        try :
            len(self.E_eV)
        except TypeError :
            self.E_eV = [self.E_eV]
        
        print('Calculating corrected form factor for ',self.specieName,
              ' (Z = ',self.Z,') at energy(ies) (in eV) : ',self.E_eV)
        
        #TODO : check this expression
        # f = np.resize(self.f0,(len(self.E_eV),len(self.Q))) + \
        #     (1.0/self.Z)*np.transpose(np.resize( self.f1 + self.f2*1j , 
        #                            (len(self.Q),len(self.E_eV)) ))
        
        print('WARNING : TEST WITH f1 and f2 SET TO 0')
        f = np.resize(self.f0,(len(self.E_eV),len(self.Q)))
        
        if resizeTo0D==True:
            try :
                if len(self.E_eV) == 1 :
                    [f] = f
            except TypeError :
                if len([self.E_eV]) == 1 :
                    [f] = f
        return f
    
def get_list_of_formFactor_objects_from_list_of_species(listOfSpecies,
    E_eV = 8047.79,plotDispersionCorrections=False,plotf0=False,
    f0CoeffsFileName='f0_WaasKirf.dat',
    dispersionCorrectionsFileName='f1f2_Henke.dat',**kwargs):
    """
    A function to create a list of formFactor objects for a list of species
    
    Files containing coefficients for f0 calculation and f1, f2 dispersion
    corrections are each opened only once to get information for all elements

    Parameters
    ----------
    listOfSpecies : list
        List of species names.
    
    E_eV : float, OPTIONAL (default=8047.78 : energy of Cu K_alpha in eV)
        Energy at which f1(E) and f2(E) dispersion corrections will be
        interpolated. Will be ignored if keyword argument waveLenth is used.
    
    f0CoeffsFileName : str, OPTIONAL (efault='f0_WaasKirf.dat')
        Name of file from which coefficients for the calculation of f0(Q) will
        be read.
    
    dispersionCorrectionsFileName : str, OPTIONAL (default='f1f2_Henke.dat')
        Name of file from which dispersion corrections f1(E) and f2(E) will
        be read.
    
    plotDispersionCorrections : bool (default=False)
        Show plot of dispersion corrections f1(E), f2(E) for identified species
        
    plotf0 : bool
        
    
    kwargs : optional keyword parameters, including :
        waveLength : float
            set energy for f1(E) and f2(E) as wavelength 
        waveLengthUnit : str
            'A' for Anstroms (the default), 'nm' or 'm'
        Q, Qmax, Qmin, Qstep, Qsteps    

    TODO : Add other optional arguments :
        energyUnit (float) : 'eV' (default) TO BE IMPLEMENTED

    Returns
    -------
    listOfFormFactors : list
        List of formFactor objects
    listOfSpecies : list
        List of species for which f0, f1 and f2 could be obtained
    """
    
    listOfSpecies = dutils.check_list_of_species(listOfSpecies)
    Q,Qmin,Qmax,Qstep,Qsteps = dutils.get_Q_vector(**kwargs)
    
    waveLengthUnit = 'A'
    # If keyword arument waveLenth is define it takes priority over E_eV
    if 'waveLength' in kwargs :
        waveLength = kwargs['waveLength']
        if 'waveLengthUnit' in kwargs :
            if kwargs['waveLengthUnit'] in ['A','nm','m'] :
                waveLengthUnit = kwargs['waveLengthUnit']
            else :
                print('Unknown waveLength unit. usin the default : A (Angstroms)')
        (E_eV,energyUnit) = dutils.convert_wavelength_to_energy(
            waveLength,waveLengthUnit=waveLengthUnit,energyUnit='eV')
    
    listOfFormFactorCoeffs = read_coeffs_for_f0_form_factor(listOfSpecies)
    
    listOfDispersionCorrections = read_dispersion_corrections_f1_f2_from_file(
        listOfSpecies,showPlots=plotDispersionCorrections)
    
    f0,listOfSpecies,Q = calculate_form_factors_f0_from_coeffs(
        listOfFormFactorCoeffs,Q=Q,showPlots=plotf0)
    
    listOfFormFactors = []
    for index,specie in enumerate(listOfSpecies) :
        dc = listOfDispersionCorrections[index]
        ff = formFactor()
        ff.specieName
        ff.Z = dc.Z
        ff.Q = Q
        ff.f0 = f0[index][:]
        ff.E_eV = E_eV
        ff.f1,ff.f2 = dc.calculate_f1_f2_from_E_eV(E_eV)
        listOfFormFactors.append(ff)
    
    return listOfFormFactors,listOfSpecies
    
    
class dispersionCorrections() :
    """
    A class to manipulate f1 and f2 dispersion corrections to atomic form factors
    
    Properties : 
        specieName : str
            name of element or specie as designated in the original file.
        f1 : numpy.ndarray
            real part of the dipsersion correction (also called f'(E)) 
        f2 : numpy.ndarray
            imaginary part of the correction (also called f"(E))
        E : numpy.ndarray
    """
    def __init__(self):
        self.specieName = ''
        self.Z = 0
        self.f1 = np.asarray((),float)
        self.f2 = np.asarray((),float)
        self.E = np.asarray((),float)
    
    def calculate_f1_f2_from_E_eV(self,E_eV):
        """
        Calculate f1(E) and f2(E) for a given value, using interpolation if necessary
        
        Parameters :
            E_eV : np.darray of floats
        """
        
        f1_E = np.interp(E_eV,self.E,self.f1)
        f2_E = np.interp(E_eV,self.E,self.f2)
        return f1_E,f2_E
    
                
def read_coeffs_for_f0_form_factor(listOfSpecies,fileName='f0_WaasKirf.dat',
                                   fileFormat='DABAX',**kwargs) :
    """
    Reading coefficients needed to calculate f0 form factor from DABAX files 
    
    Helper function designed to extract coefficiants from ESRF DABAX library
    files (http://www.esrf.fr/computing/scientific/dabax).
    Coefficient may then be used to calculate the atomic form factor from the
    following expression  :
        
    f(Q) = a1*exp(-b1*(Q/4*pi)^2) + a2*exp(-b2*(Q/4*pi)^2) + a3*exp(-b3*(Q/4*pi)^2) + a4*exp(-b4*(Q/4*pi)^2) + c
    
    Args : 
        - listOfSpecies : list of neutral ('B', 'La') or ionic species ('Al3+',
          etc.)
        - fileName : name from of file from which coefficients should be 
          obtained. By default the function uses files in the DABAX format.
        - fileFormat : 'DABAX' (the default)
        
    Returns :
        - listOfFormFactorCoeffs : list
            List of formFactorCoefficients objects
    """
    listOfSpecies = dutils.check_list_of_species(listOfSpecies)    

    try:
        with open(fileName,'r') as f:
            listOfFormFactorCoeffs = []
            specieIdentified = False
            identifiedSpecieIndex = 0
            for line in f :
                if not line :   # End of file
                    break 
                elif identifiedSpecieIndex >= len(listOfSpecies) :
                    print('Requested number of species (',len(listOfSpecies),
                          ') reached.')
                    break
                line.strip()    # Removing EOL character 
                if line[0] == '#':
                    if line[1:3] == 'S ':
                        ffc = formFactorCoefficients()
                        # read Z and specie
                        [Z,specieName] = line[2:-1].split()
                        if specieName in listOfSpecies :
                            ffc.specieName = specieName
                            ffc.Z = int(Z)
                            # print('Z = ',ffc.Z,' ; specieName = ',ffc.specieName)
                            specieIdentified = True
                    elif specieIdentified == True and line[1:3] == 'N ':
                        [nbOfCoeffs] = line[2:-1].split()
                        ffc.nbOfCoeffs = int(nbOfCoeffs)
                    elif specieIdentified == True and line[1:3] == 'L ':
                        ffc.coeffNames = line.split()[1:]
                # Line does not start with # -> line containing coefficients ?
                elif specieIdentified == True :
                    # print('Reading line containing the ',ffc.nbOfCoeffs,
                    #       ' coefficients ',ffc.coeffNames,' for ',
                    #       ffc.specieName,' :\n',line)
                    ffc.coeffValues = np.asarray(line.split(),float)
                    listOfFormFactorCoeffs.append(ffc)
                    identifiedSpecieIndex += 1
                    specieIdentified = False

                # else : line does not start with # : do nothing    
            
    except FileNotFoundError:
        print('File ',fileName,' not found or not accessible.')

    return listOfFormFactorCoeffs
    # End of function read_coeffs_for_f0_form_factor
    
def read_dispersion_corrections_f1_f2_from_file(listOfSpecies,showPlots=False,
    fileName='f1f2_Henke.dat',fileFormat='DABAX',**kwargs) :
    """
    Reading dispersion corrections f1(E) and f2(E) from ESRF-DABAX file 
    
    Helper function designed to extract f1(E) and f2(E) from ESRF DABAX librarya1*exp(-b1*(Q/4*pi)^2) + a2*exp(-b2*(Q/4*pi)^2) + a3*exp(-b3*(Q/4*pi)^2) + a4*exp(-b4*(Q/4*pi)^2) + c
    files (http://www.esrf.fr/computing/scientific/dabax).
    Coefficient may then be used to calculate the atomic form factor from the
    following expression  :
        
    f(Q,E) = f0(Q) + f1(E) + i*f2(E)
    
    Args : 
        listOfSpecies : list
            list of neutral ('B', 'La') or ionic species ('Al3+',etc.)
        fileName : str
            name from of file from which coefficients should be obtained.
            By default the function uses files in the DABAX format.
        fileFormat : str (default = 'DABAX')
            Only the ESRF DABAX database format is currently implemented.
        - showPlots : bool (default = False)
            If true the function will plot f1(E) anf f2(E) for all atoms
    Optional keyword parameters :
        - Q, Qmin, Qmax, Qstep, Qsteps
    
    No other formats besides ESRF-DABAX are currently implemented.    
    
    Returns :
        - listOfDispersionCorrections : list of objects for class
          dispersionCorrections
    
    """
    listOfSpecies = dutils.check_list_of_species(listOfSpecies)    

    try:
        with open(fileName,'r') as f:
            listOfDispersionCorrections = []
            specieIdentified = False
            identifiedSpecieIndex = 0
            for line in f :
                if not line :   # End of file
                    break 
                elif identifiedSpecieIndex >= len(listOfSpecies) :
                    print('Requested number of species (',len(listOfSpecies),
                          ') reached.')
                    break
                line.strip()    # Removing EOL character 
                if line[0:3] == '#S ' and specieIdentified == False:            
                    if line[1:3] == 'S ':
                        dc = dispersionCorrections()
                        [Z,specieName] = line[2:-1].split()
                        if specieName in listOfSpecies :
                            dc.specieName = specieName
                            dc.Z = int(Z)
                            print('Z = ',dc.Z,' ; specieName = ',dc.specieName)
                            specieIdentified = True
                # elif specieIdentified == True and line[0:3] == '#N ':
                #     [nbOfColumns] = line[2:-1].split()
                #     nbOfColumbns = int(nbOfColumns)
                elif specieIdentified == True and line[0:4] == '#UO ':
                    columnNames = line.split()[1:]
                # Line does not start with # -> line containing coefficients ?
                elif line[0] != '#' and specieIdentified == True :
                    colValues = line.split()
                    for colNb,colName in enumerate(columnNames) :
                        if colName == 'E(eV)' :
                            dc.E = np.append(dc.E,float(colValues[colNb]))
                        elif colName == 'f1' :
                            dc.f1 = np.append(dc.f1,float(colValues[colNb]))
                        elif colName == 'f2' :
                            dc.f2 = np.append(dc.f2,float(colValues[colNb]))
                elif line[0:3] == '#S ' and specieIdentified == True:
                    listOfDispersionCorrections.append(dc)
                    identifiedSpecieIndex += 1
                    specieIdentified = False

                # else : line does not start with # : do nothing    
            
    except FileNotFoundError:
        errorMsg = 'File '+fileName+' not found or not accessible.'
        raise FileNotFoundError(errorMsg)
    
    if showPlots :
        fig = plt.figure('Dispersion corrections to atomic form factors')
        ax = fig.add_subplot(1,1,1)
        legend = []
        for dc in listOfDispersionCorrections :
            ax.plot(dc.E,dc.f1)
            legend.append(dc.specieName+' : f1')
            ax.plot(dc.E,dc.f2)
            legend.append(dc.specieName+' : f2')
        ax.legend(legend)
        ax.set(xlabel='E (eV)',ylabel='Dispersion correction f1(E) or f2(E)',
               title='Dispersion corrections f1, f2 to atomic form factors')
        plt.show()

    return listOfDispersionCorrections
    # End of function read_coeffs_for_f0_form_factor    
    



def calculate_form_factors_f0_from_coeffs(listOfFormFactorCoeffs : list,**kwargs) :
    """
    A function to calculate atomic form factor f0 from Gaussian cefficients
    
    Parameters :
        listofFormFactorCoeffs : list
            List of objects of formFactorCoefficients class
    
    Optional parameters : 
    * Q=<numpy.array> defaults to 0-25 Angtroms^-1 by steps of 0.1.
    * Qmax=<float>
    * Qmin=<float> (defaults to 0)
    * Qsteps=<int>
    * showPlots=False
    
    Returns :
        f0 : numpy array
            array of dimension (nbOfSpecies,len(Q)) giving the corresponding
            atomic form factors
        listOfSpecies : list of species names
        Q : numpy array
            array of Q values used for the calculation of f
    
    """
    Q,Qmin,Qmax,Qstep,Qsteps = dutils.get_Q_vector(**kwargs)

    if 'showPlots' in kwargs :
        showPlots = bool(kwargs['showPlots'])
    
    listOfSpecies = []
    f0 = np.zeros((len(listOfFormFactorCoeffs),len(Q)))
    for speciesIndex,ffc in enumerate(listOfFormFactorCoeffs) :
        # TODO :
        # - read species
        # - loop over X in aX and bX to calculate gaussian component
        # - read and add c coefficient
        listOfSpecies.append(ffc.specieName)
        for x in range(1,ffc.nbOfCoeffs//2 + 1) :
            a_x = ffc.get_coefficient('a'+str(x))
            b_x = ffc.get_coefficient('b'+str(x))
            f0[speciesIndex][:] += a_x*np.exp(-b_x*np.square(Q/(4*np.pi)))
        f0[speciesIndex][:] += ffc.get_coefficient('c')
    
    if showPlots :
        fig = plt.figure('Atomic form factors')
        ax = fig.add_subplot(1,1,1)
        for speciesIndex,specieName in enumerate(listOfSpecies) :
            ax.plot(Q,f0[speciesIndex][:],label=specieName)
        ax.legend(listOfSpecies)
        ax.set(xlabel='Q (1/Angstrom)',ylabel='Atomic form factor f0(Q)',
               title='Atomic form factors f0')
        plt.show()

    return f0, listOfSpecies, Q

def get_atom_form_factors_from_coeff_file(listOfSpecies,**kwargs) :
    """
    Get atomic form factor function f(Q) for a list of selected elements (or ions) in the form H, H1-, Li1+, Al, Fe2+. The f functions are calculated based on a list of elements provided by :
    http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    which were taken in turn from : International Tables for Crystallography: http://it.iucr.org/Cb/ch6o1v0001/.
    with the following expression :
    f(Q) = a1*exp(-b1*(Q/4*pi)^2) + a2*exp(-b2*(Q/4*pi)^2) + a3*exp(-b3*(Q/4*pi)^2) + a4*exp(-b4*(Q/4*pi)^2) + c
    
    Inputs : 
    * listOfSpecies : a single string or a list of strings designating one element or ion in the form 'H', 'H1-', 'He', 'Li', 'Li1+'. The syntax must match the syntax used

    Optional inputs : 
    * Q=<numpy.array> defaults to 0-25 Angtroms^-1 by steps of 0.1.
    * Qmax=<float>
    * Qmin=<float> (defaults to 0)
    * Qsteps=<int>
    * showPlots=False
    TO BE IMPLEMENTED : * coeffFileName=???
    
    Returns :
    numpy.array of dimension (nbOfInputElements,size(Q))

    OBSOLETE : USE FUNCTIONS READING COEFFICIENTS FROM ESRF DABAX Files
    
    """

    listOfSpecies = dutils.check_list_of_species(listOfSpecies)    
    Q,Qmin,Qmax,Qstep,Qsteps = dutils.get_Q_vector(**kwargs)

    if 'showPlots' in kwargs :
        showPlots = bool(kwargs['showPlots'])
    
    coeffFileName = r'D:\cadarp02\Documents\Programming\Utils\Diffraction\CoefficientsForFormFactorCalculations_2020-10-05.txt'
    tableData = np.genfromtxt(coeffFileName, skip_header=0, names=True, dtype=None, encoding=None, )

    listOfTabulatedSpecies = []
    [listOfTabulatedSpecies.append(tableDataRow[0]) for tableDataRow in tableData]
    
    # Identify row indexes corresponding to requested species in tableData
    listRowIndexes = []
    for speciesIndex,specieName in enumerate(listOfSpecies) :
        # check whether speciesName exists in file
        try :
            index = listOfTabulatedSpecies.index(specieName)
            listRowIndexes.append(index)
        except ValueError :
            # If an charged specie is required, try with corresponding neutral element instead
            matchObj = re.search(r'\d+', specieName)
            if matchObj == None :
                errorMsg = 'Requested specie : ' + specieName + ' not found in table of atomic-form-factor coefficients.'
                raise ValueError (errorMsg)
            else :
                fisrtNumCharIndex = matchObj.start()
                try :
                    index = listOfTabulatedSpecies.index(specieName[:fisrtNumCharIndex])
                    print('WARNING : specie not found. Using neutral element : ',specieName[:fisrtNumCharIndex],' instead.')
                    listRowIndexes.append(index)
                    listOfSpecies[speciesIndex] = specieName[:fisrtNumCharIndex]
                except ValueError :
                    errorMsg = 'Requested specie : ' + specieName+' or ' + specieName[:fisrtNumCharIndex] + ' not found in table of atomic-form-factor coefficients.'
                    raise ValueError(errorMsg)
    print('Coefficients for the calculation of atomic form factors available for the following species : ',listOfSpecies)                
    #TODO : deal with case where species are not found in list
    # Several possibilities :
    #  - unknown oxidation state -> use neutral element and print warning
    #  - invalid element name -> raise error
    
    f = np.array([])
    for elemtIndex,rowIndex in enumerate(listRowIndexes):
        # For some reason tableData[rowIndex] is a numpy.void of dimension 0 but size 10 (?!) whose elements can only be retrieved one by one, i.e tableData[rowIndex][1:9] won't work.
        _ = tableData[rowIndex]
        a1,b1,a2,b2,a3,b3,a4,b4,c = _[1],_[2],_[3],_[4],_[5],_[6],_[7],_[8],_[9]
        #Calculating atomic form factor for the element current table row
        f_tmp = np.resize(np.sum([a1*np.exp(-b1*np.square(Q/(4*np.pi))),a2*np.exp(-b2*np.square(Q/(4*np.pi))),a3*np.exp(-b3*np.square(Q/(4*np.pi))),a4*np.exp(-b4*np.square(Q/(4*np.pi)))+c],axis=0),(1,Q.size)) ;
        if elemtIndex == 0: 
            f = f_tmp
        else :
            f = np.append(f,f_tmp,axis=0)

    if showPlots :
        #TODO : plot for every specie in listOfSpecies
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for speciesIndex,specieName in enumerate(listOfSpecies) :
            ax.plot(Q,f[speciesIndex],label=specieName)
        ax.legend(listOfSpecies)
        plt.xlabel = 'Q (1/Angstrom)'
        plt.ylabel = 'Atomic form factor f(Q)'
        plt.show()
        plt.close()

    # TODO (?) convert f, Q into numpy arrays of dimensions (1,Q.size) instead of (Q.size,)
    return f, Q
#****** end of function get_atom_form_factors_from_coeff_file ****




