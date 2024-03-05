from pymatgen import Lattice, Structure
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
from collections import Counter

"""
Identifies local structural units up to the second bonding shell for every site in a crystal structure

TO DO: implement as a function with the following arguments : 
- pymatgen.core Structure object or structure file name (e.g., .cif file)
- nearest-neighbor method (default : pymatgen.analysis.local_env.crystalNN)
- nearest-neighbor weight cutoff (default could be 1) : weight above which nearest-neighbor sites will be
  considered bonded.
- nearest-neighbor weight warning threshold (default could be 0.5) : threshold above which a warning will
  be printed to indicate that nearest-neighbor sites may potentially be considered bonded.

"""

# Load structure from CIF file
InputFileName = "D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif"
structure = Structure.from_file(InputFileName)
# Setting oxidation states to default
structure.add_oxidation_state_by_guess()
# TO DO : Setting user-defined oxidation states (e.g. with add_oxidation_state_by_site method)

species = structure.composition.elements
print(species)

print(structure.ntypesp,' distinct types of species in structure : ',structure.types_of_specie)
# print('Structure of composition ',structure.composition,' loaded from file ',InputFileName,'.')
print(structure)

# Creating an array of PeriodicSite objects
sites = structure.sites

# Nearest-neighbor-weight cutoff. Should be set to 1 in most "simple" cases.
NNWeightCutoff = 0.7
NNWeithtWarningThreshold = 0.5

# Initilization of a list of local unit names to be defined for each site in the structure
LocalUnitsStrList = [] ;

for centerSite,centerSiteIndex in zip(sites,range(0,structure.num_sites-1)):
    if centerSite.is_ordered:
        print (centerSite.specie,' ',centerSite.a,' ',centerSite.b,' ',centerSite.c)
        # Initialization de localUnits (same size as number of sites in structure)
        centerSiteStrList = [centerSite.specie.symbol] # not entirely sure brackets are necessary
        # Exploring first coordination shell
        firstShellNNData = CrystalNN().get_nn_data(structure,centerSiteIndex)
        num_1stNbr = 0
        for k in range(0,len(firstShellNNData.all_nninfo)-1):
            if firstShellNNData.all_nninfo[k]["weight"] >= NNWeightCutoff:
                num_1stNbr += 1
                firstShellSite = firstShellNNData.all_nninfo[k]["site"]
                firstShellSiteIndex = firstShellNNData.all_nninfo[k]["site_index"]
                firstShellSiteImage = np.array(firstShellNNData.all_nninfo[k]["image"])
                secondShellStrList = [firstShellSite.specie.symbol]
                # Exploring second shell
                print('Exploring second shell of center-atom ',centerSite.specie.symbol,' (index ',centerSiteIndex,\
                ') associated with 1st neighbor ',firstShellSite.specie.symbol,' (index ',firstShellSiteIndex,').')
                secondShellNNData = CrystalNN().get_nn_data(structure,firstShellSiteIndex)
                num_2ndNbr = 0
                # TO DO : Create molecule from 2nd-neighbor sites to get reduced formula
                # TO DO : Add formula to 1st-neighbor element name
                secondShellSiteList = []
                # print('secondShellNNData = ',secondShellNNData)
                for l in range(0,len(secondShellNNData.all_nninfo)-1):
                    secondShellSiteImage = np.array(secondShellNNData.all_nninfo[l]["image"])
                    if secondShellNNData.all_nninfo[l]["site_index"] == centerSiteIndex and \
                    np.array_equal(secondShellSiteImage,[0,0,0]-firstShellSiteImage):
                       # Skipping original center-site
                       """ print('Skipping original center-site with index ',secondShellNNData.all_nninfo[l]["site_index"],\
                       ' and image ',secondShellSiteImage,' to avoid double-count in 2nd shell.') """
                    elif secondShellNNData.all_nninfo[l]["weight"] >= NNWeightCutoff:
                        # Adding second-shell atom with crystalNN weight >= to NNWeightCutoff
                        num_2ndNbr += 1
                        secondShellSite = secondShellNNData.all_nninfo[l]["site"]
                        secondShellSiteList.append(secondShellSite)
                        # secondShellStrList.append(secondShellSite.specie.symbol)
                        """ print('Adding second-shell atom ',secondShellNNData.all_nninfo[l]["site"].specie.symbol,\
                        ' (index ',secondShellNNData.all_nninfo[l]["site_index"],', image ',secondShellNNData.all_nninfo[l]["image"],\
                        ') to list.') """
                    else:
                        currentSite = secondShellNNData.all_nninfo[l]["site"]
                        currentSiteIndex = secondShellNNData.all_nninfo[l]["site_index"]
                        if secondShellNNData.all_nninfo[l]["weight"] >= NNWeithtWarningThreshold and \
                        secondShellNNData.all_nninfo[l]["weight"] < NNWeightCutoff :
                            # Printing a warning to indicate potential nearest-neighbor site
                            warn('Skipping second-shell neighbor of center-atom ',centerSite.specie.symbol,\
                            '(index ',centerSiteIndex,') : site ',currentSite.specie.symbol,' with index ',\
                            secondShellNNData.all_nninfo[l]["site_index"],', image ',secondShellNNData.all_nninfo[l]["image"],\
                            ' and weight ',secondShellNNData.all_nninfo[l]["weight"],'. Distance to 1st-shell atom ',\
                            firstShellSite.specie.symbol,' (index ',firstShellSiteIndex,') = ',\
                            structure.get_distance(firstShellSiteIndex,currentSiteIndex,None),' A.')
                        else :
                            # Skipping second-shell neighbor with crystalNN weight < NNWeightCutoff
                            """ print('Skipping second-shell neighbor of center-atom ',centerSite.specie.symbol,\
                            '(index ',centerSiteIndex,') : site ',currentSite.specie.symbol,' with index ',\
                            secondShellNNData.all_nninfo[l]["site_index"],', image ',secondShellNNData.all_nninfo[l]["image"],\
                            ' and weight ',secondShellNNData.all_nninfo[l]["weight"],'. Distance to 1st-shell atom ',\
                            firstShellSite.specie.symbol,' (index ',firstShellSiteIndex,') = ',\
                            structure.get_distance(firstShellSiteIndex,currentSiteIndex,None),' A.') """
                            
                # formulaStr = Molecule().from_sites(secondShellSiteList).formula
                # print(secondShellSiteList)
                if len(secondShellSiteList) != 0:
                    structureTmp = Structure.from_sites(secondShellSiteList)
                    secondShellStrList.append(structureTmp.composition.reduced_formula)
                    """ print('Center atom ',centerSite.specie.symbol,' (index ',centerSiteIndex,\
                    '), 1st neighbor ',firstShellSite.specie.symbol,' (index ',firstShellSiteIndex,') : ',\
                    structureTmp.composition.reduced_formula) """
                else:
                    warn('No other 2nd-neighbor than center-atom found (e.g. terminal O atom).')
                centerSiteStrList.append('(' + ''.join(secondShellStrList) + ')')
        print(centerSiteStrList)
        c = Counter(centerSiteStrList[1:])
        localUnitStr = [centerSiteStrList[0]]
        for key,value in c.items() :
            localUnitStr.append(key + str(value))
        LocalUnitsStrList.append(''.join(localUnitStr))
    else:
        error('Mixed occupancy site detected.')
        print (centerSite.species,' ',centerSite.a,' ',centerSite.b,' ',centerSite.c,' (mixed-occupancy site)')  

for siteIndex in range(0,structure.num_sites-1)
    print(centerSite.species,'\t',centerSite.a,'\t',centerSite.b,'\t',centerSite.c,'\t',LocalUnitsStrList[siteIndex]
print('Done.')
  
