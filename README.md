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

	The diffractionPkg/nanopdf.py module was originally developped by Olivier Masson, Institut de Recherche sur les Ceramiques, CNRS, Universite de Limoges, France, and adapted by Sylvian Cadars. If you use this specific module, please cite:
	Masson O., and Thomas P., Exact and Explicit Expression of the Atomic Pair Distribution Function as Obtained from X-Ray Total Scattering Experiments, Journal of Applied Crystallography 2013, 46 (2), 461‑65. https://doi.org/10.1107/S0021889812051357.

- ceramicNetworkBuilder : build a ceramic network atom-by atom with coordinance and clustering probabilities.

Currently-implemented packages :
- plotUtils : convenient tools to interactively modify plots created with matplotlib

Other directories :
- vaspCalcPreparations : examples of script to prepare series of VASP calculations and corresponding submission scripts.



