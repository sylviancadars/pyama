*******************************************************
PyAMA - Python Atomic Modeling Analyses
*******************************************************
Author : Sylvian Cadars
Affiliation : Institut de Recherche sur les Ceramiques, CNRS, Universite de Limoges, France
e-mail : sylvian.cadars@unilim.fr

Description : 
Set of Python tools dedicated to the analyses and interoperation of codes for the atomic modeling of solid-state materials.

The library is based on different scientific libraries including :
- numPy
- SciPy
- pymatgen (https://pymatgen.org/)
- dscribe

Modeling programs and codes interfaced with the library currently include :
- supercell program (https://orex.github.io/supercell/)
- VASP (https://www.vasp.at/)
- USPEX (https://uspex-team.org/en/uspex/overview)

Currently-implemented packages :
- structureComparisonsPkg : tools to calculate periodic structure fingerprints and use them to calculate distances between these.
- uspexAnalysesPkg : manipulate and extract structures generated with the supercell program for combinatorial anlyses of disorder in crystals
- diffractionPkg : tools to calculate atomic form factors, and (in progress) caculate exact X-ray total scattering data
- ceramicNetworkBuilder : build a ceramic network atom-by atom with coordinance and clustering probabilities.

Currently-implemented packages :
- plotUtils : convenient tools to interactively modify plots created with matplotlib

Other directories :
- vaspCalcPreparations : examples of script to prepare series of VASP calculations and corresponding submission scripts.



