# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:10:13 2020

@author: cadarp02
"""

# Clearing variables before script execution
from IPython import get_ipython
get_ipython().magic('reset -sf')

import structureComparisonsPkg.distanceTools as dt
import uspexAnalysesPkg.uspexDataManipulation as udm
# import pymatgen.core.structure
import pickle

#*************** MAIN PROGRAM ******************
#fileOrDirectoryName = r'D:\cadarp02\Documents\Programming\examples\uspex\uspex9.4.4_EX01-3D_Si_vasp\results1'

# fileOrDirectoryName = r'D:\cadarp02\Documents\Modeling\USPEX\AsTe3\AsTe3_Z-4_vaspPBE_201014\results1'

fileOrDirectoryName = r'D:\cadarp02\Documents\Modeling\USPEX\AsTe3\AsTe3_Z-6_vaspPBE_201019\results1'

usd = udm.uspexStructuresData(fileOrDirectoryName,'all')
# dmd = dt.distanceMatrixData(sigma=0.02,Rmax=8.0,Rsteps=512)

# Vizualize two of the best 20 structures with VESTA
# IDs = usd.get_structure_IDs(bestStructures=20)
# usd.visualize_structure(IDs[[1,10]],visualizationProgram='VESTA')
# usd.extract_POSCARS_from_IDs(usd.get_structure_IDs(mostStableStructures=10))

# Plot enthalpies for the best 50 structures
fig = usd.plot_enthalpies(usd.get_structure_IDs(bestStructures=50),
                          relativeEnthalpies=True,enthalpyUnit='eVByAtom')


# Load the best and worst structures of the uspex run 
# [structure1] = usd.get_structures_from_IDs(usd.get_structure_IDs(bestStructures=1))
# [structure2] = usd.get_structures_from_IDs(usd.get_structure_IDs(worstStructures=1))

# plot one fingerprint, calculate cosine distance
# dmd.calculate_fingerprint_AB_component(structure1,'Te','Te',showPlot=True)

# Load two structures among the best
# bestStructures = usd.get_structures_from_IDs(usd.get_structure_IDs(bestStructures=20))
# structure1 = bestStructures[1]
# structure2 = bestStructures[10]
# print('Cosine distance = ',dmd.calculate_cosine_distance(structure1,structure2,showPlot=True))

# # Calculate cosine distance matrix from the best X structures
# IDs = usd.get_structure_IDs(bestStructures=40)[[0,9,19,29,39]]
# usd.extract_POSCARS_from_IDs(IDs)
# usd.calculate_cosine_distance_matrix(IDs,sigma=0.2,Rmax=6.0,Rsteps=256)
# # usd.distanceMatrixData.plot_distance_matrix()

# # Save uspexStructuresData object as pickle file to then re-open it
# usd.save_data_to_file()
# fileName = usd.saveFileName

# Opening previously-stored uspexStructuresData object from pickle file
# fileName = r'D:\cadarp02\Documents\Modeling\USPEX\AsTe3\AsTe3_Z-6_vaspPBE_201019\results1\uspexStructuresDataDir\savedData.pkl'
# usd2 = udm.uspexStructuresData()

# with open(fileName,'rb') as f:
#     usd2 = pickle.load(f)
# print("usd2.IDs = ",usd2.IDs)
# print("usd2.saveDistMatrixFileName = ",usd2.saveDistMatrixFileName)
# usd2.distMatrixData = []
# usd2.load_distance_matrix_from_file()
# usd2.distMatrixData.plot_distance_matrix()


# bestIDs =  usd.get_structure_IDs(bestStructures=20)
# struct1 = usd.get_structure_from_ID(bestIDs[0])
# struct2 = usd.get_structure_from_ID(bestIDs[19])
# usd.distanceMatrixData.calculate_cosine_distance(struct1,struct2,showPlot=True)

# fileName = r'D:\cadarp02\Documents\Modeling\USPEX\AsTe3\AsTe3_Z-6_vaspPBE_201019\results1\uspexStructuresDataDir\distMatData.pkl'
# with open(fileName,'rb') as f:
#     dmd = pickle.load(f)
# print(dmd.__dict__)
# print('dmd.Dmatrix = ',dmd.Dmatrix)