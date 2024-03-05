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
import math
import re
import copy


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
        if isinstance(coeffIndexOrName,int) :
            try :
                coeff = self.coeffValues[coeffIndexOrName]
                print('text')
            except IndexError :
                print('Invalid atomic form factor coefficient index.')
        elif isinstance(coeffIndexOrName,str) :
            try :
                coeff = self.coeffValues[self.coeffNames.index(coeffIndexOrName)]
                print('text')
            except IndexError :
                print('Invalid atomic form factor coefficient name.')
        
        return coeff

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
        - listOfIdentifiedSpecies
        - numpy array of dimension (nbOfSpecies,11) where nbOfSpecies corresponds
          to the species identified in the file. length 9 in second dimension
          corresponds to coefficients : a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,,c in
          the above
    
    """
    listOfSpecies = check_list_of_species(listOfSpecies)    
    Q,Qmin,Qmax,Qstep,Qsteps = get_Q_vector(**kwargs)

    if 'showPlots' in kwargs :
        showPlots = bool(kwargs['showPlots'])
    
    # parse file (see sample of relevant lines below)
#S  1  H
#N 11
#L a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5
#  0.413048  0.294953  0.187491  0.080701  0.023736  0.000049  15.569946 32.398468  5.711404 61.889874  1.334118  
    
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
                            print('Z = ',ffc.Z,' ; specieName = ',ffc.specieName)
                            specieIdentified = True
                    elif specieIdentified == True and line[1:3] == 'N ':
                        [ffc.nbOfCoeffs] = line[2:-1].split()
                    elif specieIdentified == True and line[1:3] == 'L ':
                        ffc.coeffNames = line.split()[1:]
                # Line does not start with # -> line containing coefficients ?
                elif specieIdentified == True :
                    print('Reading line containing the ',ffc.nbOfCoeffs,
                          ' coefficients ',ffc.coeffNames,' for ',
                          ffc.specieName,' :\n',line)
                    ffc.coeffValues = line.split()
                    listOfFormFactorCoeffs.append(ffc)
                    identifiedSpecieIndex += 1
                    specieIdentified = False

                # else : line does not start with # : do nothing    
            
    except FileNotFoundError:
        print('File ',fileName,' not found or not accessible.')

    return listOfFormFactorCoeffs
    # End of function read_coeffs_for_f0_form_factor
    

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


# main
ffc = formFactorCoefficients()
print(ffc.get_coefficient('a1'))

listOfFormFactorCoeffs = listOfFormFactorCoeffs = read_coeffs_for_f0_form_factor(['La','B'])
print(listOfFormFactorCoeffs[0].get_coefficient('a1'))