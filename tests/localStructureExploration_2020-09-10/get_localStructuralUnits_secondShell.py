def get_localStructuralUnits_secondShell(structureNameOrFile,**kwargs) :

    """
    Identifies local structural units up to the second bonding shell for every site in a crystal structure

    - structureNameOrFile should be a pymatgen.core Structure object or structure file name (e.g., .cif file)
    TO DO: implement as a function with the following arguments :
    - NNmethod : 'crystalNN' or  (default : pymatgen.analysis.local_env.crystalNN)
    - NNWeightCutoff : nearest-neighbor weight cutoff (default could be 1) : weight above which nearest-neighbor
      sites will be
      considered bonded.
    - NNWeithtWarningThreshold : nearest-neighbor weight warning threshold (default could be 0.5) above which a
      warning will be printed to indicate that nearest-neighbor sites may potentially be considered bonded.
    """

    from pymatgen import Lattice, Structure
    from pymatgen.analysis.local_env import CrystalNN
    import numpy as np
    from collections import Counter
    import os
    import warnings
    import numbers

    # Reading the structure function argument
    if isinstance(structureNameOrFile,str):
        if os.path.isfile(structureNameOrFile):
            structure = Structure.from_file(structureNameOrFile)
    elif isinstance(structureNameOrFile,Structure):
        structure = structureNameOrFile

    # Setting optional arguments to default values :
    NNWeightCutoff = 1
    NNWeithtWarningThreshold = 0.5
    
    print('kwargs = ',kwargs,', len(kwargs) = ',len(kwargs))
    if len(kwargs) > 0 :
        for key, value in kwargs.items():
            if key.lower() == 'NNWeightCutoff'.lower() : # Apparently not the most robust case-insensitive string comparison
            # tried to use unicodedata.normalize('NFKD','Hello') == unicodedata.normalize('NFKD','HELLO'))
            # but I had a problem...
                errormsg = 'Value associated with argument NNWeightCutoff should be a numerical value between 0 and 1.'
                if isinstance(value, numbers.Number):
                    if value >=0 and value <= 1:
                        NNWeightCutoff = value
                    else:
                        print(errormsg) ; return 0
                else:
                    error(errormsg)
            if key.lower() == 'NNWeithtWarningThreshold'.lower() :
                errormsg = 'Value associated with argument NNWeithtWarningThreshold should be a numerical value between 0 and 1.'
                if isinstance(value, numbers.Number):
                    if value >=0 and value <= 1:
                        NNWeithtWarningThreshold = value
                    else:
                        print(errormsg) ; return 0
                else:
                    print(errormsg) ; return 0
    
    # Adjusting nearest-neighbor weight warning threshold in case it is smaller than the NN weight cutoff
    if NNWeithtWarningThreshold >= NNWeightCutoff :
        NNWeithtWarningThreshold = max(0.01,NNWeightCutoff-0.2)

    print('NNWeightCutoff = ',NNWeightCutoff,'\n','NNWeithtWarningThreshold = ',NNWeithtWarningThreshold)
    
    # Setting oxidation states to default
    # TO DO : Check whether oxidation states are defined 
    structure.add_oxidation_state_by_guess()
    # TO DO : Setting user-defined oxidation states (e.g. with add_oxidation_state_by_site method)

    # print(structure.ntypesp,' distinct types of species in structure : ',structure.types_of_specie)
    # print('Structure of composition ',structure.composition,' loaded from file ',InputFileName,'.')
    print(structure)

    # Creating an array of PeriodicSite objects
    sites = structure.sites

    # Initilization of a list of local unit names to be defined for each site in the structure
    LocalUnitsStrList = [] ;

    for centerSite,centerSiteIndex in zip(sites,range(0,structure.num_sites-1)):
        if centerSite.is_ordered:
            # print (centerSite.specie,' ',centerSite.a,' ',centerSite.b,' ',centerSite.c)
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
                    """ print('Exploring second shell of center-atom ',centerSite.specie.symbol,' (index ',centerSiteIndex,\
                    ') associated with 1st neighbor ',firstShellSite.specie.symbol,' (index ',firstShellSiteIndex,').') """
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
                                print('WARNING : Skipping second-shell neighbor of center-atom ',centerSite.specie.symbol,\
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
                        print('WARNING : No other 2nd-neighbor than center-atom found (e.g. terminal O atom).')
                    centerSiteStrList.append('(' + ''.join(secondShellStrList) + ')')
            # print(centerSiteStrList)
            c = Counter(centerSiteStrList[1:])
            localUnitStr = [centerSiteStrList[0]]
            for key,value in c.items() :
                localUnitStr.append(key + str(value))
            LocalUnitsStrList.append(''.join(localUnitStr))
        else:
            error('Mixed occupancy site detected.')
            print (centerSite.species,' ',centerSite.a,' ',centerSite.b,' ',centerSite.c,' (mixed-occupancy site)')  

    for siteIndex in range(0,structure.num_sites-1):
        print(structure.sites[siteIndex].species,'\t',structure.sites[siteIndex].a,'\t',structure.sites[siteIndex].b,\
        '\t',structure.sites[siteIndex].c,'\t',LocalUnitsStrList[siteIndex])
    print('End of function get_localStructuralUnits_secondShell.')

#********************* TeO2-alpha ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif'
# Requires NNWeightCutoff <= 0.78 to account for longer Te-O bonds (2.08 A)

#********************* TeO2-gamma ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2-gamma_P212121_2000_COD-1520934.cif'
# Requires NNWeightCutoff <= 0.65 to account for longer Te-O bonds (2.20 A)

#********************* TeO2 Pbca ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TeO2_Pbca_1967_COD-9008125.cif'
# Requires NNWeightCutoff <= 0.65 to account for longer Te-O bonds (2.20 A)
# Problem with O(TeO3)2 units interpreted as O(TeO2)2 because the crystalNN approach does not allow double-counting.

#********************* Te2O(PO4) ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\Te2O(PO4)2_C1c1_2010_COD-4316124.cif'
# Requires NNWeightCutoff <= 0.5
# Double-counting issue is again present.
# N.B. making a supercell does not change the result


#********************* beta-As2Te3 ***********************
inputFileName = r'D:\cadarp02\Documents\CristalStructures\beta-As2Te3_2015_InorgChem-ic5b01676.cif'
# 

#********************* TiO2_rutile ***********************
# inputFileName = r'D:\cadarp02\Documents\CristalStructures\TiO2_rutile_P4_2mnm_1956_AMS_0009161.cif'
# Case of rutile is problematic because of 3-coordinated Oxygen forming short Ti-O-Ti-O rings.
# CrystalNN analysis in this case gives in this case local units that avoid double-counting of atoms.
# This is not the most pertinent approach from an NMR point view for example.

#get_localStructuralUnits_secondShell(inputFileName,NNWeightCutoff=0.5,NNWeithtWarningThreshold=0.3)

from pymatgen import Structure
structure = Structure.from_file(inputFileName)
# structure.make_supercell([2,1,1],True)
get_localStructuralUnits_secondShell(structure,NNWeightCutoff=0.5,NNWeithtWarningThreshold=0.3)

#IS THERE SOMETHING MORE THAN THE DOUBLE-COUNTING ISSUE ??? No small 2-metal-atom rings in Te2O(PO4) (only 3-metal rings)