# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:09:25 2020

@author: cadarp02
"""
import numpy as np
import scipy.constants

def get_Q_vector(**kwargs) :
    """
    A helper function to calculate Q vector in different ways
    
    Without arguments the function will use default values to define Q
    With keyword Q argument defined the function ignore other arguments and 
    return the specified Q array as numpy array of float numbers, along with
    the corresponding Qmin, Qmax, Qstep and Qsteps.
    
    Optional arguments : 
        - Q=<numpy.array> defaults to 0-25 Angtroms^-1 by steps of 0.1.
        - Qmin=<float> (defaults to 0 Angtroms^-1)
        - Qmax=<float> (defaults to 25 Angtroms^-1)
        - Qstep=<float> (defaults to 0.1 Angtroms^-1)
        - Qsteps=<int> : number of steps in Q
    
    Returns :
        - Q,Qmin,Qmax,Qstep,Qsteps
    
    """
    #**** SETTING DEFAULT VALUES FOR OPTIONAL FUNCTION INPUT ARGUMENTS *****
    Qmin = 0
    Qmax = 25
    Qstep = 0.1
    Qsteps = (Qmax-Qmin)/(Qstep)+1
    # Or alternatively :
    #Qsteps = 1000
    #Qstep = (Qmax-Qmin)/(Qsteps-1)
    showPlots = False
    Q = np.arange(Qmin,Qmax+Qstep,Qstep)  #includes Qmax in the table
    
    #**** READING AND PROCESSING OPTIONAL FUNCTION INPUT ARGUMENTS *******
    if 'Q' in kwargs :
        Q = np.asarray(kwargs['Q'])
    else :
        print('Q not specified. Looking for Qmax, Qmin and Qstep or Qsteps.')
        if 'Qmax' in kwargs :
            try :
                float(kwargs['Qmax'])
                if kwargs['Qmax'] > 0 :
                    Qmax = kwargs['Qmax']
                else :
                    print(f'WARNING : Qmax should be positive. Using the default : Qmax = ',Qmax)
            except ValueError :
                print(f'WARNING : Qmax should be a number. Using the default : Qmax = ',Qmax)  
        if 'Qmin' in kwargs :
            try :
                float(kwargs['Qmin'])
                if kwargs['Qmin'] >= 0 and  kwargs['Qmin'] < Qmax :
                    Qmin = kwargs['Qmin']
                else :
                    print('WARNING : Qmin should be between 0 and Qmax (',Qmax,'). Using the default : Qmin = ',Qmin)
            except ValueError :
                print(f'WARNING : Qmin should be a number. Using the default : Qmin = ',Qmin)
        if 'Qstep' in kwargs :
            try :
                float(kwargs['Qstep'])
                if kwargs['Qstep'] > 0 and  kwargs['Qstep'] < Qmax-Qmin :
                    Qstep = kwargs['Qstep']
                    Qsteps = (Qmax-Qmin)/(Qstep)+1
                else :
                    print('WARNING : Qstep should positive and smaller than Qmax-Qmin (',Qmax-Qmin,'). Using the default : Qstep = ',Qstep)
            except ValueError :
                print(f'WARNING : Qstep should be a number. Using the default : Qstep = ',Qstep)
        elif 'Qsteps' in kwargs : # Qsteps will be ignored if Qstep is set
            try :
                int(kwargs['Qsteps'])
                if kwargs['Qsteps'] > 0 :
                    Qsteps = kwargs['Qsteps']
                    Qstep = (Qmax-Qmin)/(Qsteps-1)
                else :
                    print('WARNING : Qsteps should positive. Using the default : Qsteps = ',Qsteps)
            except ValueError :
                print(f'WARNING : Qsteps should be an integer number. Using the default : Qsteps = ',Qsteps)
        Q = np.arange(Qmin,Qmax+Qstep,Qstep)

        print('Q vector of length ',Q.size,' generated with (Qmin,Qmax,Qstep,Qsteps) = ',(Qmin,Qmax,Qstep,Qsteps))
    
    return Q,Qmin,Qmax,Qstep,Qsteps

def get_r_vector(**kwargs) :
    """
    A helper function to create a r vector in different ways
    """
    # Setting default values
    rMin = 0.1 # rMin=0 may cause division by 0 issues
    rMax = 8
    rSteps = 512
    
    if 'r' in kwargs :
        r = np.asarray(kwargs['r'])
    else :
        if 'rMax' in kwargs :
            rMax = kwargs['rMax']
        if 'rMin' in kwargs :
            rMin = kwargs['rMin']
        # rSteps takes priority over rStep
        if 'rSteps' in kwargs :
            rSteps = kwargs['rSteps']
        elif 'rStep' in kwargs :
            rStep = kwargs['rStep']
            rSteps = (rMax-rMin)/(rStep)+1
            
        r,rStep = np.linspace(rMin,rMax,rSteps,endpoint=True,retstep=True)
    
    return r,rMin,rMax,rStep,rSteps

def check_list_of_species(listOfSpecies) :
    """
    A helper function to check the format of a list of species
    
    The function will convert a single string into a list containing one string.
    Neutral species names containing '0+' or '0-' such as 'Fe0+' will be
    converted to simple element names (e.g. 'Fe')
    
    Args :
        listOfSpecies : a string or list of species.
        
    Returns :
        formatedListOfSpecies : a formated list of species names
    """
    
    if isinstance(listOfSpecies,list) :
        formatedListOfSpecies = listOfSpecies
    else :
        if isinstance(listOfSpecies,str) :
            formatedListOfSpecies = [listOfSpecies]
        else :
            raise TypeError ('listOfSpecies input should be a list of strings or a string.')
    
    for specieIndex,specieName in enumerate(formatedListOfSpecies) :
        if not isinstance(specieName,str) :
            raise TypeError ('listOfSpecies input should be a list of strings or a string.')
        # print(specieIndex,' : ',specieName)
        
        # Check if specie name contains '0+' (as pymatgen.core.periodic_table Species class may return for neutral elements).
        try:
            index = specieName.index('0+')
            formatedListOfSpecies[specieIndex] = specieName[:index]
            print('WARNING : requested neutral specie ',specieName,' modified to ',specieName[:index])
        except ValueError :
            pass # SpecieName does not contain \'0+\' (simple element name). Do nothing.

    return formatedListOfSpecies

def convert_wavelength_to_energy(waveLength,waveLengthUnit='A',energyUnit='eV'):
    """
    A helper function to calculate the energy associated with a given wavelength

    Parameters
    ----------
    waveLength : float
        wavelength in units given in waveLengthUnit (default is 'A' for Angtroms)
    waveLengthUnit : str, optional
        wavelength unit. Possible values are 'A' (the default), 'nm', 'm'.
    energyUnit : TYPE, optional
        DESCRIPTION. The default is 'eV'.

    Returns
    -------
    E : float
        Energy associated with specified wavelength ().
    energyUnit : str
        Unit of energy associated with result. Default is 'eV'.
    """
        
    if waveLengthUnit=='A' :
        waveLength_m = waveLength*1e-10
    elif waveLengthUnit=='nm' :
        waveLength_m = waveLength*1e-9
    elif waveLengthUnit=='m' :
        waveLength_m = waveLength
        
    E = scipy.constants.physical_constants["joule-electron volt relationship"][0] \
        * scipy.constants.h*scipy.constants.c / waveLength_m

    # TODO : modify energy according to parameter energyUnit
    
    return E,energyUnit

def get_pair_index_from_indexes(i,j,N) :
    """
    Function to obtain a pair index [0...(N^2)[ for indexes between 0 and N-1
    
    Parameters : 
        i : int
            first index
        j : int
            second index
        N : int
            Number of elements (0 <= i,j <= N)
        
    Returns :
        pairIndex : int
            Pair index calculated with (i*N)+j
    """
    #TODO : test input parameters : i, j, N as integers, N > 0 and 0 <= i,j <= N
    
    pairIndex = pairIndex = (i*N)+j
    return pairIndex
#****** end of function get_pair_index_from_indexes ****

def get_indexes_from_pair_index(pairIndex,N) :
    """
    Function to retrieve (i,j) indexes (between 0 and N-1) from a pair index between 0 and (N^2-1)
    
    Parameters : 
        pairIndex : pair index defined as (i*N)+j
        N : Number of elements (0 <= i,j <= N)
        
    Returns :
        (i, j) : tuple of integers corresponding to (i, j) indexes
    """
    #TODO : test input parameters : pairIndex, N as integers, N > 0 and 0 <= pairIndex <= (N^2-1)
    
    if pairIndex == 0 :
        i, j = 0, 0
    else :
        j = pairIndex % N
        i = pairIndex // N
    return i, j
#****** end of function get_pair_index_from_indexes ****