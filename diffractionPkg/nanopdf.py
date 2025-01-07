#!/usr/local/miniconda/bin/python3
"""
Calculation of exact X-ray total scattering

Olivier Masson 05/05/2016 olivier.masson@unilim.fr
Masson, Olivier, and Philippe Thomas.
«Exact and Explicit Expression of the Atomic Pair Distribution Function as
Obtained from X-Ray Total Scattering Experiments».
Journal of Applied Crystallography 46, nᵒ 2 (1 avril 2013): 461‑65.
https://doi.org/10.1107/S0021889812051357.

Adapted for pyama and pymatgen by Sylvian Cadars, sylvian.cadars@unilim.fr
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import click
from copy import deepcopy
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize_scalar

from pymatgen.core.structure import Structure, IStructure, Molecule, IMolecule
from pymatgen.core.periodic_table import get_el_sp
from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData

MAXTYPE = 10    # Max type of atoms (cf. nanopdf.h)
MAXORDER = 20 # Max number of values in each row of rk 2D array (cf. nanopdf.h)
ECONST = 12398.4244 # Convertion factor wavelength (A) to energy (eV)

class nanopdfData():
    """
    Class containing all methods and properties necessary to run nanopdf

    TODO:
    * set ak, partials as propperties

    properties:
        self.verbosity
    """
    def __init__(self, structure=None, reference_structure=None, verbosity=1,
                 print_to_console=True, print_to_file=False,
                 lambda1=0.559422, Qmax=17.0, nkorder=10, fNy=5,
                 R_max=20, sigma=0.01,
                 experimental_file=None, experimental_title="",
                 print_performance=False):
        """
        Initialization

        Args:
            structure: pymatgen Structure instance or str
                pymatgen structure object or file name

            reference_structure: pymatgen Structure instance or str
                structure of identical composition as the system of interest
                used to calculated typenames and typeindexes properties when
                the exact X-ray pdf is calculated from pre-existing partials.
                The orering of sites in reference_structure does not matter as
                they will automatically be soiirted (with pymatgen).

            Lambda1:
                Wavelength in Angtroms
                0.709317 for Mo K_alpha_1 (KL_3)
                0.713607 for Mo K_apoha_2 (KL_2)
                0.559422 for Ag K_alpha_1
                0.563813 for Ag K_alpha_2

            experimental_file: str or None (default is None)
                File should be tab or space separated 2-column datafile without
                header. More complex files may be processed as in numpy.loadtxt()
                using the set_experiment_from_data_file() method.

            experimental_title: str (default is "")


        """
        self.verbosity = verbosity
        self.print_performance = print_performance

        self.print_to_console = print_to_console
        self.print_to_file = print_to_file
        self._output_text = []   # list of strings to be ultimately written in file

        self.lambda1 = lambda1  # ADD DEFINITION
        self.nkorder=nkorder  # ADD DEFINITION
        self.set_R(R_max, Qmax, fNy)  # also sets self.Qmax, self.R_max and self.fNy
        self.sigma = sigma
        self.typenames=None
        self.typeindexes=None
        self.energy=ECONST/lambda1
        if structure is not None:
            self.structure = self.get_sorted_structure(structure)
        else:
            self.structure = None
            if reference_structure is not None:
                self.reference_structure = self.get_sorted_structure(
                    reference_structure)
            else:
                self.reference_structure = None

        self.set_pathname()
        # self._outputfile_ = open('output.txt', 'w')   # This file will need to be closed automatically
        self.partials = None
        self.ak = None
        self.exactPDF = None
        self.experimental_title = experimental_title
        if experimental_file is not None:
            self.set_experiment_from_data_file(experimental_file)
        self.difference = None
        self.difference_x = None
        self.difference_rmsd = None

    def trace(self, *msg, verb_th=1):
        if self.verbosity >= verb_th:
            print(*msg)
        else:
            pass

    def error(*msg):
        print(*msg)
        exit()

    def set_pathname(self):
        path = os.path.abspath(__file__)
        self.pathname = os.path.dirname(path)

    #############################################################################
    ######################   Functions definition   #############################
    #############################################################################
    @staticmethod
    def fourier(u,fu,v):
        """ calculate sin fourier transform
        u -- absci for fu
        fu -- function to transform
        v -- absci of result
        """
        du = u[1]-u[0] # assume constant step
        f = np.array(fu)
        i = 0
        res = np.zeros(len(v))
        while i < len(v):
            uv = np.array(u)*v[i]
            uv = np.sin(uv)
            res[i] = np.dot(f,uv)
            i += 1
        return res*du


    @staticmethod
    def invfourier(u,fu,v):
        """ calculate inverse sin fourier transform
        u -- absci for fu
        fu -- function to transform
        v -- absci of result
        """
        du = u[1]-u[0] # assume constant step
        f=np.array(fu)
        i=0
        res=np.zeros(len(v))
        while i < len(v):
            uv=np.array(u)*v[i]
            uv=np.sin(uv)
            res[i] = np.dot(f,uv)
            i+=1
        return res*du*2.0/np.pi


    def readf0(self, atom):
        """
        Read f0 form factor for atom from file
        """
        res=[]
        found=0
        fname=os.path.join(self.pathname,"f0_WaasKirf.dat")
        self.trace('Reading f0({}) atomic form factor from file {}.'.format(
            get_el_sp(atom).name, fname), verb_th=2)
        try:
            f = open(fname)
        except:
            self.error("cannot open file",fname)
        while f:
            line=f.readline()
            if len(line)==0:
                break
            if line[0:2]=="#S":
                data = line.split()
                if data[2]==atom:
                    found=1
                    break
        if found==0:
            self.error("could not find atom/ion %s" % atom)
        atnum = int(data[1])
        line=f.readline()
        data=line.split()
        if int(data[1]) != 11:
            print(line)
            self.error("pb with f0_WaasKirf.dat")
        line=f.readline() # skip
        line=f.readline()
        data=line.split()
        i=0
        while i<11:
            res.append(float(data[i]))
            i+=1
        f.close()

        return atnum, res


    def readf1f2(self, atnum):
        """
        Read f1 and f2 contributions to X-ray form factors from file
        """
        found=0
        fname=os.path.join(self.pathname,"f1f2_Henke.dat")
        self.trace('Reading f1/f2({}) atomic form factors from file {}.'.format(
            get_el_sp(atnum).name, fname), verb_th=2)
        try:
            f = open(fname)
        except:
            self.error("cannot open file",fname)
        while f:
            line=f.readline()
            if len(line)==0:
                break
            if line[0:2]=="#S":
                data = line.split()
                if int(data[1])==atnum:
                    found=1
                    break
        if found==0:
            self.error("could not find atom/ion with Z= %i" % atnum)
        line=f.readline() #skip
        line=f.readline() #skip
        line=f.readline()
        data=line.split()
        if int(data[1]) != 3:
            print(line)
            self.error("pb with f1f2_Henke.dat")
        line=f.readline() # skip
        line=f.readline() # skip
        found=0
        line=f.readline()
        data=line.split()
        en=float(data[0])
        f1=float(data[1])
        f2=float(data[2])
        while en<self.energy:
            enold=en
            f1old=f1
            f2old=f2
            line=f.readline()
            data=line.split()
            en=float(data[0])
            f1=float(data[1])
            f2=float(data[2])
        f.close()
        self.trace(f"f1 and f2 values interpolated between {enold} and {en} eV",
                   verb_th=2)
        res1=(f1-f1old)/(en-enold)*(self.energy-enold)+f1old
        res2=(f2-f2old)/(en-enold)*(self.energy-enold)+f2old
        return res1-atnum, res2


    def getSF(self):
        """ read scattering data from files... """

        self.trace(">>> Reading scattering data from %s..." % self.pathname, verb_th=1)

        ntyp=len(self.typenames)
        f0=[]
        f1=[]
        f2=[]
        atnum=[]
        fat=[]

        i=0
        while i<ntyp:
            res=self.readf0(self.typenames[i])
            atnum.append(res[0])
            fat.append(float(res[0]))
            f0.append(res[1])
            res=self.readf1f2(atnum[i])
            f1.append(res[0])
            f2.append(res[1])
            self.trace("data for {} :".format(self.typenames[i]), verb_th=2)
            self.trace("  Z = {}\n  f0 = {}\n  f1 = {}\n  f2 = {}".format(
                atnum[i], f0[i], f1[i], f2[i]), verb_th=2)
            i+=1

        return f0,f1,f2

    def getf0(self, f0, sintol):
        """ for the non-dispersive part of the atomic scattering factor is a
        // function of the selected element and of sin(theta)/lambda, where
        // lambda is the photon wavelengh and theta is incident angle.
        // This function can be approximated by a function:
        //
        //   f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
        //               i=1,5
        //
        // where k = sin(theta) / lambda and c, a_i and b_i
        // are the coefficients tabulated in this file (in columns:
        // a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5
        """
        f=np.array([f0[5]]*len(sintol))
        i=0
        while i<5:
            f += f0[i]*np.exp(-f0[i+6]*np.square(sintol))
            i+=1
        return f


    def getfmbfn(self, m,n,f0,f1,f2,sintol):
        """ given f = x +iy with x=f0+f1 and y=f2,
        compute 0.5*[fm*conj(fn)+conj(fm)*fn],
        i.e. the real part of fm*conj(fn) (or conj(fm)*fn)
        """

        xm = self.getf0(f0[m],sintol) + f1[m]
        ym = f2[m]
        xn = self.getf0(f0[n],sintol) + f1[n]
        yn = f2[n]

        return xm*xn + ym*yn # Real part of fm * conj(fn)


    def getavfsq(self, compo,f0,f1,f2,sintol):
        # f = x +iy x=f0+f1, y=f2;
        x=[]
        y=[]
        i=0
        while i<len(compo):
            x.append(self.getf0(f0[i],sintol)+f1[i])
            y.append(np.array([f2[i]]*len(sintol)))
            i+=1

        sum=np.sum(compo)
        res=np.zeros(len(sintol))

        i=0
        while i<len(compo):
            res+=(np.square(x[i])+np.square(y[i]))*(compo[i]/sum)
            i+=1

        return res


    def getsqavf(self, compo,f0,f1,f2,sintol):
        #// f = x +iy x=f0+f1, y=f2
        # TODO: UNDERSTAND WHETHER THERE IS A DIFFERENCE BETWEEEN
        # getavfsq() and getsqavf() AND CHOSE ONE OTHERWISE.
        x=np.zeros(len(sintol))
        y=np.zeros(len(sintol))
        sum=np.sum(compo)
        i=0
        while i<len(compo):
            x+=(self.getf0(f0[i],sintol)+f1[i])*(compo[i]/sum)
            y+=np.array([f2[i]*(compo[i]/sum)]*len(sintol))
            i+=1

        return np.square(x)+np.square(y)


    def calculate_ak(self):
        """ calculate ak coefficients J.Appl.Cryst (2013) 46, 461-465 """

        ntyp=len(self.typenames)
        nbatoms = self.typeindexes[ntyp] # global var nbatoms is now initialized...

        self.trace(">>> Number of atoms : {} {}".format(nbatoms,
            [self.typeindexes[i+1]-self.typeindexes[i] for i in range(ntyp)]),
            verb_th=2)
        self.trace(">>> Atom types : {}".format(self.typenames), verb_th=2)

        cat=[0.0]*ntyp
        i=0
        while i < ntyp:
            cat[i]=float(self.typeindexes[i+1]-self.typeindexes[i])/float(nbatoms)
            i+=1

        self.trace(">>> Atom concentrations : {}".format(
            [cat[i] for i in range(ntyp)]), verb_th=2)

        nbpairs=ntyp*(ntyp+1)//2

        self.trace((">>> atomic scattering factors will be calculated at {} eV"
                    ).format(self.energy), verb_th=2)

        f0,f1,f2 = self.getSF()

        for i in range(ntyp):
            pass
            self.trace((">>> atomic scattering factors for atom {} at Q=0 is {}"
                        ).format(i, self.getf0(f0[i],[0.0])), verb_th=2)

        Qstar = 0.02 # target Qstep, should be small enough...
        nbQ = (int)(self.Qmax/Qstar)
        Qstep = self.Qmax / nbQ
        Q1=np.arange(0,self.Qmax+0.00001,Qstep)
        sintol=Q1/(4.0*np.pi)

        self.trace("number of Q points = {}".format(len(Q1)), verb_th=2)

        favsq=self.getavfsq(cat,f0,f1,f2,sintol)
        fsqav=self.getsqavf(cat,f0,f1,f2,sintol)
        gamQ=[]

        cicj=np.zeros(nbpairs)
        npair=0
        for i in range(ntyp):
            for j in range(i,ntyp):
                cicj[npair]=cat[i]*cat[j]
                if i==j:
                    gamQ.append(self.getfmbfn(i,j,f0,f1,f2,sintol)/fsqav)
                else:
                    gamQ.append(2.0*self.getfmbfn(i,j,f0,f1,f2,sintol)/fsqav)
                npair+=1

        self.ak=np.zeros((nbpairs,MAXORDER))

        i=0
        while i < nbpairs:
            self.ak[i] = np.fft.irfft(gamQ[i])[0:MAXORDER] * cicj[i]
            i+=1

        sumak=np.zeros(MAXORDER)
        ip=0
        while ip < nbpairs:
            self.trace("ak[{}] = {}".format(ip,str(self.ak[ip])), verb_th=2)
            sumak+= self.ak[ip]
            ip+=1

        self.trace(">>> ci*cj = {}".format(cicj), verb_th=2)
        self.trace(">>> SUM OF AK =  {}".format(sumak), verb_th=2)


    def get_exact_pdf_without_ripples(self):
        """ compute the 'exact' pdf to the order k from the partials functions partials[i] and ak
        J.Appl.Cryst (2013) 46, 461-465 """

        nbpairs = len(self.partials)
        if nbpairs != len(self.ak):
            self.error("pb with the dimensions of partials/ak arrays")

        nbp = len(self.partials[0])
        expdf = np.zeros([nbpairs,nbp])

        i=0
        while i < nbpairs:
            k=0
            while k <= self.nkorder:
                coeff=self.ak[i][k]/self.ak[i][0]  # w_ab(r_k)
                j=0
                while j < nbp:
                    if (k==0):
                        expdf[i][j] = self.partials[i][j]
                    else:
                        j1=j-self.fNy*k
                        j2=j+self.fNy*k

                        if (j1 == -1):  # because g(r=0) not defiined
                            tmp=self.partials[i][j2]
                        elif (j1 < -1):
                            tmp=self.partials[i][j2]-self.partials[i][-2-j1]
                        elif (j2 >= nbp):
                            tmp=self.partials[i][j1]-self.partials[i][2*nbp-(j2+1)]
                        else:
                            tmp=self.partials[i][j1]+self.partials[i][j2]

                        expdf[i][j]+=coeff*tmp
                    j+=1
                k+=1
            i+=1

        exactpdf=np.zeros(nbp)

        ip=0
        for ip in range(nbpairs):
            exactpdf+=self.ak[ip][0]*expdf[ip]

        return exactpdf

    def get_sorted_structure(self, structure_or_file=None):
        """
        Get a pymatgen structure with atoms sorted by species from a filename or a
        from a pymatgen structure
        """
        if isinstance(structure_or_file, str):
            # load Pymatgen structure from file
            structure = Structure.from_file(structure_or_file)
        elif isinstance(structure_or_file, Structure):
            structure = structure_or_file
        elif structure_or_file is None:
            if self.structure is not None:
                structure = self.structure
            else:
                sys.exit('Mehod get_sorted_structure requires a structure input or '
                         'predefined structure property.')
        else:
            sys.exit('structure_or_file should be a pymatgen Structure or a ' +
                     'file name')
        structure.sort()
        return structure

    def set_system_info(self, use_reference_structure=False):
        """
        retrieve typenames and typenames as used in calculate_ak() from structure

        structure (or reference_structure) should be sorted feborehand : use
        get_sorted_structure()

        Args:
            use_reference_structure: bool (default is False)
                typenames and typeindexes will be defined from
                reference_structure property rather than from structure
                property.

        Sets:
            typenames: list of string
                list of atom types as they appear in the sorted pymatgen Structure
            typeindexes: list of int
                gives for each type of atom the first index and at the end the
                number of sites in the system (i.e. the last index of the last
                type + 1). len(typeindexes) = len(typenames) + 1
        """
        self.typenames = []      # order will be the same as in output
        self.typeindexes = []
        if use_reference_structure and self.reference_structure is not None:
            structure = self.reference_structure
        elif self.structure is not None:
            structure = self.structure
        else:
            self.error('set_system_info: error. No (reference) structure set.')
        for index, specie in enumerate(structure.species):
            if specie.name not in self.typenames:
                self.typenames.append(specie.name)
                self.typeindexes.append(index)
        self.typeindexes.append(index+1)

    def set_system_info_from_structure(self, structure_or_file):
        """
        Set sorted structure, typenames, typeindexes properties from structure
        or structure files

        Args:
            structure_or_file: pymatgen Structure instance or str
                file name should correspond to a structure format known by
                pymatgen

        """
        self.structure = self.get_sorted_structure(structure_or_file)
        self.set_system_info()

    def set_system_info_from_reference_structure(self, structure_or_file=None):
        if structure_or_file is None:
            if self.reference_structure is not None:
                self.set_system_info(use_reference_structure=True)

        self.reference_structure = self.get_sorted_structure(structure_or_file)


    def set_reference_structure(self, structure_or_file):
        self.reference_structure = self.get_sorted_structure(structure_or_file)
        # TODO: add warning if struccture and reference_structure have
        # diffferent compositions ???

    def compute_partials(self):
        """
        Compute partials and store them in self.partials
        """
        self.partials = []
        dmd = distanceMatrixData(sigma=self.sigma, R=self.R)
        reduced_partial_RDFs, _ = \
            dmd.calculate_all_reduced_partial_RDFs(self.structure,
                                                   showPlot=False)
        for i in range(len(self.typenames)):
            for j in range(i + 1):
                self.trace('Calculating {}-{} partial RDF.'.format(
                    self.typenames[i], self.typenames[j]), verb_th=2)
                self.partials.append(reduced_partial_RDFs[i, j, :])

    def get_atom_pair_names(self):
        """
        Get list of atom pairs in the same order as in partials from typenames
        """
        atom_pair_names = []
        for i in range(len(self.typenames)):
            for j in range(i + 1):
                atom_pair_names.append('{}-{}'.format(self.typenames[i],
                                                      self.typenames[j]))
        return atom_pair_names

    def set_partials_from_structure(self, structure_or_file=None):
        """
        Compute partials and set info from a pymatgen structure of structure
        file
        """
        self.set_system_info_from_structure(structure_or_file)
        self.compute_partials()

    def add_termination_ripples(self, exact_pdf_without_ripples):
        """
        Add ripples due to the truncation of exp. data at Qmax after Fourier Transform
        """
        Q1 = np.arange(0, self.Qmax + 0.00001, 0.02)
        iexact = self.fourier(self.R, exact_pdf_without_ripples, Q1)   # WARNING : slow Fourier transform
        self.exactPDF = self.invfourier(Q1, iexact, self.R)  # WARNING : slow Fourier transform

    def set_R(self, R_max=None, Qmax=None, fNy=None):
        """
        Set r property as used in nanopdf from fNy, Qmax, and R_max values

        The stored self.property value will be used for all arguments
        set to None. Otherwise all properties will be updated.

        Args:
            R_max: float (default is None)
            Qmax: float (default is None)
            fNy: float (default is None)
        """
        if R_max is not None:
            self.R_max = R_max
        if Qmax is not None:
            self.Qmax = Qmax
        if isinstance(fNy, int):
            self.fNy = fNy
        drNy = np.pi / self.Qmax
        dr = drNy / self.fNy  # Nyquist step / fNy
        self.R = np.arange(dr, R_max, dr)

    def get_partials_in_nanopdf_format(self, extern_partials, types_in_extern_partials):
        """
        Makes the difference between 2D (nanopdf definition) or 3D partials

        3D partials should correspond to an array of dimension
        (len(types), len(types), len(r)) with:
        partials[i, j, :] = partials[j, j, :]
        for pair of species (i-j)
        """
        if isinstance(extern_partials, np.ndarray) and isinstance(
            types_in_extern_partials, (list, tuple)):
            if extern_partials.ndim == 3:
                nanopdf_partials = self.get_sorted_partials_from_3D_partials(extern_partials,
                                                                             types_in_extern_partials)
            elif extern_partials.ndim == 2:
                nanopdf_partials = extern_partials
        elif isinstance(extern_partials, (list, tuple)):
            nanopdf_partials = extern_partials
            # TODO: make sure that types_in_extern_partials match typenames

        return nanopdf_partials


    def get_sorted_partials_from_3D_partials(self, partials_3d,
                                             types_in_partials_3d):
        """
        Helper function to get partials from a 3D array of partials_3d

        partials_3d and types_in_partials_3d should be as obtained with
        pyama.structureComparisonsPkg.distanceMatrixData method
        calculate_all_partial_reduced_RDFs()

        Example:
            from pyama.structureComparisonsPkg.distanceTools import distanceMatrixData

            nd = nanopdfData()
            dmd = distanceMatrixData(R = nd.R)
            partials_3d, types_in_partials_3d =
            dmd.calculate_all_partial_reduced_RDFs(structure)
            nd.set_system_info_from_structure(structure)
            partials_in_nanopdf_format = get_sorted_partials_from_3D_partials(partials_3d)

        Args:
            partials_3d : np array of dimension 3
                Shape should be (len(types), len(types), len(r)).
            types_in_partials_3d: list of str
                list of atom types as returned by distanceMatrixData
                calculate_all_partial_reduced_RDFs() method

        Returns:
            Flattened list of partials arrays as used by nanopdf
        """
        sorted_partials = []

        # Check that typenames and typeindexes are present and retreieve them
        # if possible from a reference strucure
        if self.reference_structure is not None:
            self.set_system_info(use_reference_structure=True)
        if self.typenames is None or self.typeindexes is None:
            if self.structure is None:
                self.error('Error in function get_sorted_partials_from_3D_partials.'
                           'typenames and typeindexes should be defined before '
                           'from a reference structure of identical composition and '
                           'method: set_system_info_from_structure()')
            else:
                self.set_system_info()
        if len(self.typenames) != len(types_in_partials_3d):
            self.error('Property typenames and types_in_partials_3d have ' +
                       'different lengths.')

        for i, t_i in enumerate(self.typenames):
            for t_j in self.typenames[:i+1]:
                j = self.typenames.index(t_j)
                sorted_partials.append(partials_3d[
                    types_in_partials_3d.index(t_i),
                    types_in_partials_3d.index(t_j), :])
        return sorted_partials

    def calculate_final_exact_pdf_from_partials(self, partials, types_in_partials=None):
        """
        Calculate final exact x-ray PDF from pre-calculated partials.

        The typical case is the caclulation of a PDF from snaphots of a
        molecular dynamics run. In such cases the partials should be calculated
        first by averaging the partials of the different snaphots, using the
        same R vector and atom-type orderings.

        This requires setting a reference structure (e.g. the initial ionic
        step of the molcular dynamic rnun) with set_reference_structure()
        method.

        Partials should typically be calculated with:
        distanceMatrixData.calculate_all_reduced_partial_RDFs()
        with a R vector calculated previously with nanopdf

        Example:
        # Assume that structures is a list of pymatgen structures corresponding
        # to snaphots of a molecular dynamics run.
        from nanopdf.nanopdf import nanopdfData
        from structureComparisonsPkg.distanceTools import distanceMatrixData
        nd = nanopdfData(R_max=10, sigma=0.02)
        dmd=distanceMatrixData(R=nd.R, sigma=nd.sigma)
        # Initialize averaged partials
        averaged_partials = None  # Initialize
        partials_count = 0
        for structure in structures:
            partials, types = dmd.calculate_all_reduced_partial_RDFs(structure)
            if averaged_partials is None:
                # Initalize averaged_partials
                averaged_partials = partials
            else:
                averaged_partials += partials
            partials_count += 1
        averaged_partials = averaged_partials / partials_count
        nd.set_reference_structure(structures[0])
        nd.calculate_final_exact_pdf_from_partials(averaged_partials, types)
        fig, ax = nd.plot_partials()
        plt.show()

        args:
            partials: 3D numpy darray of list of numpy darrays
                Either a numpy darray of shape (len(types), len(types), len(R))
                as generated with
                distanceMatrixData.calculate_all_reduced_partial_RDFs() or a
                list of 1D partials as used in nanopdf.

            types_in_partials: list or None (default is None)
                list of types in the order corresponding to partials. Typically
                obtained from
                distanceMatrixData.calculate_all_reduced_partial_RDFs().

        Sets:
            self.exactPDF
        """
        self.partials = self.get_partials_in_nanopdf_format(partials,
            types_in_partials)

        self.calculate_ak()
        pdf_without_ripples = self.get_exact_pdf_without_ripples()
        self.add_termination_ripples(pdf_without_ripples)
    
    def calculate_final_exact_pdf_from_structure(self, structure_or_file=None):
        """
        Calculate the total X-ray PDF for the given structure file or pymatgen structure

        Args:
            partials: None or array of partials

        Returns:
            exact_pdf
        """
        self.set_system_info_from_structure(structure_or_file)
            
        if self.print_performance:
            performance = {}
            tic = time.perf_counter()
            init_tic = tic

        # define chemical system and load partials from file
        self.set_partials_from_structure()
        if self.print_performance:
            performance['partial_rdfs'] = time.perf_counter() - tic

        # compute ak coefficients
        if self.print_performance: tic = time.perf_counter()
        self.calculate_ak()
        if self.print_performance:
            performance['ak_coeffs'] = time.perf_counter() - tic

        # Compute exact PDF from partials
        if self.print_performance: tic = time.perf_counter()
        pdf_without_ripples = self.get_exact_pdf_without_ripples()

        if self.print_performance:
            performance['total_pdf_calc'] = time.perf_counter() - tic

        # add termination ripples: modulation by sin(Q_max*r)/(pi*r)
        if self.print_performance: tic = time.perf_counter()
        self.add_termination_ripples(pdf_without_ripples)

        if self.print_performance:
            performance['riddles'] = time.perf_counter() - tic
            performance['total'] = time.perf_counter() - init_tic
            print('Calculation of partial RDFs: {:.3f} ms.'.format(
                1000*performance['partial_rdfs']))
            print('Computation of ak coefficients: {:.3f} ms.'.format(
                1000*performance['ak_coeffs']))
            print('Computation of the total exact PDF: {:.3f} ms.'.format(
                1000*performance['total_pdf_calc']))
            print('Addition of riddles: {:.3f} ms.'.format(
                1000*performance['riddles']))
            print('Total nanopdf execution: {:.3f} s.'.format(
                performance['total']))


    def export_exact_pdf(self, file_name="exactPDF.out",
                         include_partials=True, show_plot=False):
        with open(file_name, 'w') as f:
            line = '{:10}{:12}'.format('R(\u212B)', 'exactPDF')
            if include_partials:
                pair_names = self.get_atom_pair_names()
                for pair_name in pair_names:
                    line += '{:12}'.format('g({})'.format(pair_name))
            f.write('{}\n'.format(line))
            for i in range(len(self.R)):
                line = '{:10.6f}{:12.5f}'.format(self.R[i], self.exactPDF[i])
                if include_partials:
                    for partial in self.partials:
                        line += '{:12.5f}'.format(partial[i])
                f.write('{}\n'.format(line))

    def _plot_exported_data(self, file_name="exactPDF.out"):
        """
        FOR DEBUGGING¨PURPOSES
        """
        data = np.genfromtxt(file_name, names=True)
        tot_y_shift = 0
        y_shift = 0.25*(max(data[data.dtype.names[1]]) -
                        min(data[data.dtype.names[1]]))
        fig, ax = plt.subplots()
        x = data[data.dtype.names[0]]
        lgd = data.dtype.names[1:]
        for name in data.dtype.names[1:]:
            ax.plot(x, data[name] + tot_y_shift)
            tot_y_shift += y_shift
        ax.set(xlabel=data.dtype.names[0], ylabel='Intensity (AU)',
               title='Data plotted from file: {}'.format(file_name))
        ax.legend(lgd)
        return fig, ax

    def set_experimental_title(self, experimental_title=None):
        if isinstance(experimental_title, str):
            self.experimental_title = experimental_title
        elif experimental_title is None and self.experimental_title == '':
            self.experimental_title = 'Experiment'
        elif experimental_title is None:
            pass
        else:
            raise TypeError('experimental_title should be a str or None.')


    def set_experiment_from_data_file(self, experimental_file_or_data,
                                      experimental_title=None, **kwargs):
        """
        Set experimental_data property from a file using numpy.loadtxt()

        The file should contain 2 columns with r and PDF data, respectively

        All keyword arguments of numpy.loadtxt() may be used, which includes
        in particular (see help(numpy.loadtxt) for details) :
            delimiter: str or None (default is None)
            skiprows: int (default is 0)
                the first skpirows lines will be ignored
            comments: str or sequence of str or None (default is '#')
                mark indicating the start of a comment

        Args:
            experimental_file_or_data:
                name of file containing the data.
                function will be transparent if experimental_file_or_data
                is already a numpy.darray
            experimental_title: str or None (default is None)
                If not None the experimental_title property will be updated.
        """
        if experimental_file_or_data is None:
            pass
        if isinstance(experimental_file_or_data, np.ndarray):
            self.experimental_data = experimental_file_or_data
        elif isinstance(experimental_file_or_data, str):
            self.experimental_data = np.loadtxt(experimental_file_or_data)
        else:
            raise TypeError(('experimental_file_or_data should be a 2D '
                             'numpy.darray or file name'))
        # TODO: Test shape
        error_msg = 'Experimental data should be a numpy.darray of shape (N, 2).'
        if len(self.experimental_data.shape) != 2:
            raise ValueError(error_msg)
        elif self.experimental_data.shape[1] != 2:
            raise ValueError(error_msg)

        # Set experimental_title property
        self.set_experimental_title(experimental_title)

    def add_experiment_to_plot(self, fig, ax, experimental_file_or_data=None,
                               experimental_title=None, y_shift=0):
        if experimental_file_or_data is not None:
            self.set_experiment_from_data_file(experimental_file_or_data,
                experimental_title=experimental_title)
        ax.plot(self.experimental_data[:,0],
                self.experimental_data[:,1] + y_shift, color='black')
        ax.set(xlim=[0, max(self.R)])
        # append legend if any

        return fig, ax


    def calculate_difference_to_experiment(self, experimental_file_or_data=None):
        """
        Calculate difference to an experiment using interpolation.

        Sets difference, difference_x and difference_rmsd properties
        """
        self.trace('Function calculate_difference_to_experiment.', verb_th=2)
        if experimental_file_or_data is not None:
            self.set_experiment_from_data_file(experimental_file_or_data)

        x = self.experimental_data[:,0]
        y = self.experimental_data[:,1]
        # Set x, y limits in the [x_min, x_max] range
        x_min = np.maximum(min(self.R), min(self.experimental_data[:,0]))
        x_max = np.minimum(max(self.R), max(self.experimental_data[:,0]))
        _x = x[x >= x_min]
        _y = y[x >= x_min]
        x = _x[_x <= x_max]
        y = _y[_x <= x_max]
        # interpolate self.exactPDF onto x
        spl = splrep(self.R, self.exactPDF, xb=x_min, xe=x_max)
        interp_pdf = splev(x, spl)
        self.difference = y - interp_pdf
        self.difference_x = x
        self.difference_rmsd = np.sqrt(np.mean(np.square(self.difference)))

    def optimize_sigma_vs_expt(self, structure_or_file, experimental_file_or_data=None,
                               update_sigma=True, show_plot=False, title=None, data_name=None,
                               diff_rel_y_shift=-0.75, expt_rel_y_shift=0, push_legend=True):
        """
        Optimize the sigma value to minimize the difference to experimental pdf data for a structure

        If show_plot is True the function takes the same arguments as method plot_exact_pdf()
        Use update_sigma=True (the default) to update the sigma property and corresponding exactPDF,
        difference, difference_x and difference_rmsd properties.

        Args:
            structure_or_file: pymatgen Structure, str
                Structure used to simulate the X-ray total scattering data.
                Any file format known by pymatgen may be used.
            experimental_file_or_data: str, numpy.darray or None
                name of file containing the experimental data.
                function will be transparent if experimental_file_or_data
                is already a numpy.darray. If None the experimental_data
                property will be used (if available).
            update_sigma: bool (default is True)
                Whether the sigma property should be set to the
                optimized value. If true the exactPDF, difference,
                difference_x and difference_rmsd properties will also be updated.
            show_plot: bool (default is False)
                Whether a plot of experiment, simulation with optimized
                sigma value and corresponding difference should be shown.
            title: str or None (default is None)
                Title of the plot. If None title will be set to :
                'Simulated exact total X-ray scattering'.
            data_name: str or None (default is None)
                Name of the simulated data in legend. If None data_name
                will be set to 'Simulated PDF'.
            experimental_file: str (default is None)
                Name of file containing experimental total pdf function
            experimental_title: str (default is None)
                Title of experimental data to be included in legend
            show_experiment: bool (default is False)
                Whether experiment will be shown on plot.
                Automatically set to True if experimental_file
                is not None.
            expt_rel_y_shift: float (default is 0.0)
                Shift of experimental data with respect simulation
                relative to the simulated intensity range.
            diff_rel_y_shift: float (default is -0.75)
                Shift of difference plot with respect simulation
                relative to the simulated intensity range.
            push_legend: bool (default is True)
                Whether legend should be pushed to the right side of
                the plot. plt.tight_layout() is automatically used
                in this case. if False the plotted y_lim[1] is increased
                to leave room on the top for legend.

        Returns:
            sigma, opt_result, fig, ax if show_plot is True
            sigma, opt_result if show_plot is False

        """

        self.trace('Function optimize_sigma_vs_expt.', verb_th=2)
        if experimental_file_or_data is not None:
            self.set_experiment_from_data_file(experimental_file_or_data)

        def f(x):
            _nd = deepcopy(self)
            _nd.sigma = x
            _nd.calculate_final_exact_pdf_from_structure(structure_or_file)
            _nd.calculate_difference_to_experiment()
            return _nd.difference_rmsd

        opt_result = minimize_scalar(f, bounds=(0.001, 0.5), method='bounded')
        sigma = opt_result.x

        self.trace('Result of the optimization of sigma:', verb_th=1)
        self.trace(('  sigma_opt = {:.4f} \u212B\n  success = {}\n  message = '
                    '{}').format(opt_result.x, opt_result.success,
                                 opt_result.message), verb_th=1)

        if show_plot:
            if data_name is not None:
                _data_name = '{} - \u03C3 = {:.3f} \u212B'.format(data_name, sigma)
            else:
                _data_name = 'Simulated PDF -  \u03C3 = {:.3f} \u212B'.format(sigma)
            if title is None:
                title='Simulated pdf with optimized sigma value'

        if update_sigma:
            self.sigma=sigma
            self.calculate_final_exact_pdf_from_structure(structure_or_file)
            self.calculate_difference_to_experiment()

            if show_plot:
                fig, ax = self.plot_exact_pdf(title=title, data_name=_data_name,
                    show_experiment=True, show_difference=True, 
                    expt_rel_y_shift=expt_rel_y_shift, diff_rel_y_shift=diff_rel_y_shift,
                    push_legend=push_legend)
        elif show_plot:
            # A copy of self should be created
            _nd = deepcopy(self)
            _nd.sima = sigma
            _nd.calculate_final_exact_pdf_from_structure(structure_or_file)
            _nd.calculate_difference_to_experiment()
            fig, ax = _nd.plot_exact_pdf(title=title, data_name=_data_name,
                show_experiment=True, show_difference=True, 
                expt_rel_y_shift=expt_rel_y_shift, diff_rel_y_shift=diff_rel_y_shift,
                push_legend=push_legend)

        if show_plot:
            return sigma, opt_result, fig, ax
        else:
            return sigma, opt_result

    def add_difference_to_plot(self, fig, ax,
                               experimental_file_or_data=None,
                               experimental_title=None,
                               rel_y_shift = -0.25):
        if self.difference is None or experimental_file_or_data is not None:
            self.calculate_difference_to_experiment(experimental_file_or_data)

        if rel_y_shift == 0.0:
            y_shift = 0
        else:
            y_shift = rel_y_shift * (self.exactPDF.max() - self.exactPDF.min())
        ax.plot(self.difference_x, self.difference + y_shift)
        ax.set(xlim=[0, max(self.R)])
        # append legend if any

        return fig, ax


    def plot_exact_pdf(self, title=None, data_name=None, experimental_file=None,
                       experimental_title=None, show_experiment=False,
                       expt_rel_y_shift=0.0, show_difference=False,
                       diff_rel_y_shift=-0.75, show_partials=False, 
                       partials_rel_y_shift=None, rel_shift_between_partials=0.25, 
                       push_legend=True):
        """
        Plot exact x-ray total scattering function, possibly with experiment

        Args:
            title: str or None (default is None)
                Title of the plot. If None title will be set to :
                'Simulated exact total X-ray scattering'.
            data_name: str or None (default is None)
                Name of the simulated data in legend. If None data_name
                will be set to 'Simulated PDF'.
            experimental_file: str (default is None)
                Name of file containing experimental total pdf function
            experimental_title: str (default is None)
                Title of experimental data to be included in legend
            show_experiment: bool (default is False)
                Whether experiment will be shown on plot.
                Automatically set to True if experimental_file
                is not None.
            expt_rel_y_shift: float (default is 0.0)
                Shift of experimental data with respect simulation
                relative to the simulated intensity range.
            show_difference: bool (default is False)
                Whether difference plot relative to experiment.
                Needs experimental_data defined via experimental_file
                or the set_experiment_from_data_file() method.
            diff_rel_y_shift: float (default is -0.75)
                Shift of difference plot with respect to simulation, 
                relative to the simulated intensity range.
            show_partials: bool (default is False)
                Whether partial RDFs should be shown.
            partials_rel_y_shift: float or None (default is None)
                If None partials will automatically be plotted above
                experimental data.
            rel_shift_between_partials: float (default is 0.25)
            push_legend: bool (default is True)
                Whether legend should be pushed to the right side of
                the plot. plt.tight_layout() is automatically used
                in this case. if False the plotted y_lim[1] is increased
                to leave room on the top for legend.

        Returns:
            fig and axes handles.
        """
        if self.exactPDF is None:
            if self.structure is not None:
                self.calculate_final_exact_pdf_from_structure(self.structure)
            elif self.partials is not None:
                self.calculate_final_exact_pdf_from_partials(self.partials)
                
        fig, ax = plt.subplots()
        ax.plot(self.R, self.exactPDF)
        if isinstance(data_name, str):
            lgd = [data_name]
        else:
            lgd = ['Simulated PDF']

        if not isinstance(title, str):
            title = 'Simulated exact total X-ray scattering'

        ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)', title=title)
        self.set_experimental_title(experimental_title)
        if experimental_file is not None:
            self.set_experiment_from_data_file(experimental_file)
            show_experiment = True

        if show_experiment and self.experimental_data is not None:
            if expt_rel_y_shift != 0:
                y_shift = expt_rel_y_shift*(max(self.exactPDF) - min(self.exactPDF))
            else:
                y_shift = 0
            fig, ax = self.add_experiment_to_plot(fig, ax, y_shift=y_shift)
            lgd.append(self.experimental_title)

        if show_difference:
            fig, ax = self.add_difference_to_plot(fig, ax,
                                                  rel_y_shift=diff_rel_y_shift)
            lgd.append('Difference (RMSD = {:.3f})'.format(self.difference_rmsd))
        
        if show_partials:
            if self.partials is None and self.structure is not None:
                self.calculate_all_reduced_partial_RDFs(self.structure)
            if self.partials is not None:
                if partials_rel_y_shift is None:
                    y_max = max([max(l.get_ydata()) for l in ax.lines])
                    y_min = min([min(l.get_ydata()) for l in ax.lines])
                    print('y_min = {}; y_min = {}'.format(y_min, y_max))
                    tot_y_shift = y_max + 0.05 * (y_max - y_min)
                else:
                    tot_y_shift = partials_rel_y_shift*(max(self.exactPDF) - min(self.exactPDF))
                
                atom_pairs = self.get_atom_pair_names()
                y_shift = rel_shift_between_partials * max([max(partial) - min(partial) for 
                                                            partial in self.partials])
                for pair_index, partial in enumerate(self.partials):
                    ax.plot(self.R, partial + tot_y_shift)
                    lgd.append(atom_pairs[pair_index] + ' partial')
                    tot_y_shift += y_shift
            else:
                self.trace('WARNINGS: No partials available for plot.')
            
        if push_legend:
            ax.legend(lgd, loc='upper left', bbox_to_anchor=[1.05, 1.0])
            fig.tight_layout()
        else:
            ax.legend(lgd)
            # Leave space above for legend
            y_lim = ax.get_ylim()
            ax.set_ylim([y_lim[0], y_lim[1] + 0.25*(y_lim[1]-y_lim[0])])

        # plt.show() better to use plt.show once all figures are set or they will
        # open one after the other, only when the former is closed.

        return fig, ax


    def plot_multiple_exact_pdfs(self, structures_or_files,
                                 data_names=None,
                                 title=None,
                                 experimental_file=None,
                                 experimental_title=None,
                                 relative_y_shift=0.5,
                                 constrained_layout=True,
                                 show_experiment=False):
        """
        Plot multiple exact total x-ray scattering

        Args:
            structures:
        """
        structures = []
        lgd = []
        y_shift=None
        y_range=None
        total_y_shift = 0
        fig, ax = plt.subplots(constrained_layout=constrained_layout)
        for index, struct_or_file in enumerate(structures_or_files):
            if isinstance(struct_or_file, (Structure, IStructure,
                                           Molecule, IMolecule)):
                structure = struct_or_file
            elif os.isfile(struct_or_file):
                structure = Structure.from_file(struct_or_file)
            else:
                self.trace(('plot_multiple_exact_pdfs, iteration {}: unknown '
                            'structure type or invalid structure file: {} (of '
                            'type {})').format(index, struct_or_file,
                                               type(struct_or_file)), verb_th=1)
                continue # jump to next for loop step
            if data_names is not None:
                lgd.append(data_names[index])
            elif os.isfile(struct_or_file):
                lgd.append(os.path.basename(struct_or_file))
                #TODO: use full name if basename is contcar or poscar
            else:
                lgd.append('Simulated pdf # {} - {}'.format(index,
                                                            structure.formula))
            self.calculate_final_exact_pdf_from_structure(structure)
            if y_shift is None:
                y_range = max(self.exactPDF) - min(self.exactPDF)
                if y_range > 0:
                    y_shift = relative_y_shift * y_range
            ax.plot(self.R, self.exactPDF + total_y_shift)
            if y_shift is not None:
                total_y_shift += y_shift

        self.set_experimental_title(experimental_title)
        if experimental_file is not None:
            self.set_experiment_from_data_file(experimental_file)
            show_experiment = True

        if show_experiment and self.experimental_data is not None:
            fig, ax = self.add_experiment_to_plot(fig, ax,
                                                  y_shift=total_y_shift)
            lgd.append(self.experimental_title)

        ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)',
               title='Simulated exact total X-ray scattering')
        ax.legend(lgd, bbox_to_anchor=(1.1, 1.0), loc='upper left')

        return fig, ax


    def plot_partials(self, include_total_pdf=True, experimental_file=None,
                      experimental_title=None, relative_y_shift=0.5):
        """
        Plot partial radial distribution functions

        Args:
            include_total_pdf: bool (dfefault is True
                Whether final_total_pdf should be shown alongside partials
            experimental_file: str (default is None)
                Name of file containing experimental total pdf function
            experimental_title: str (default is None)
                Title of experimental data to be included in legend
            relative_y_shift: float (default is 0.5)
                y_shift between partials in fraction of the first partial
                amplitude range.

        Returns:
            Figure and axes handles
        """
        y_shift = relative_y_shift * (max(self.partials[0]) -
                                      min(self.partials[0]))
        total_y_shift = 0
        if include_total_pdf:
            fig, ax = self.plot_exact_pdf(experimental_file=experimental_file,
                                          experimental_title=experimental_title)
            lines = ax.get_lines()
            shift_y = max(lines[0].get_ydata())
            legend_hdle = ax.get_legend()
            if legend_hdle is not None:
                legend = legend_hdle.label
            else:
                if len(lines) >= 1:
                    legend = ['Total X-ray pdf']
                if len(lines) >= 2:
                    legend.append('Experiment')
            total_y_shift = max([line.get_ydata()[0] for line in lines]
                                ) + y_shift
        else:
            fig, ax = plt.subplots()
            plt.xlabel('r (\u212B)')
            plt.ylabel('Intensity (AU)')
            legend=[]

        atom_pairs = self.get_atom_pair_names()
        for index, partial in enumerate(self.partials):
            ax.plot(self.R, partial + total_y_shift)
            legend.append(atom_pairs[index] + ' partial')
            total_y_shift += y_shift
            ax.legend(legend)
        return fig, ax


# Set label of the group containing the structures on which analyses should be
# performed.
@click.command('cli')
@click.option('-i', '--structure_file_name', type=str,
              help='Structure file name (CIF, XYZ, POSCAR, etc.)')
@click.option('-l', '--wave_length',
              default=0.709318, type=float,
              help='Wave length in Angstroms (default is 0.709318 A).' +
                   'use 0.559422 for silver')
@click.option('-q', '--q_max',
              default=21.795972,
              type=float,
              help='Experimental max Q value in Angstrom^-1. For silver ' +
                   'use 21.795972 Angstrom^-1')
@click.option('-k', '--k_order', default=10, type=int,
              help='Maximum order of PDF correction terms. Test convergence.')
@click.option('-f', '--nyquist_fraction', default=5,
              type=int,
              help='Oversampling relative to the Nyquist factor.')
@click.option('-r', '--r_max', default=20, type=int,
              help='Maximum radius for the calculation of partial radial' +
                   'density fucntions. Default is 20.')
@click.option('-s', '--sigma', default=0.01, type=float,
              help='Gaussian smoothing factor for partial RDFs in Angtroms' +
              '. Default is 0.01')
@click.option('-o', '--output_file', default='exactPDF.out', type=str,
              help='Output file containing the final r and calculated total' +
                   'total scattering function.')
@click.option('-v', '--verbosity', default=1, type=int,
              help='Verbosity level (0, 1, 2, etc.). Default is 1.')
def cli(structure_file_name, wave_length, q_max, k_order, nyquist_fraction,
        r_max, sigma, output_file, verbosity):
    # test whether structure file may be loaded within pymatgen.
    structure = Structure.from_file(structure_file_name)

    nd = nanopdfData(lambda1=wave_length, Qmax=q_max, nkorder=k_order,
                     fNy=nyquist_fraction, R_max=r_max, sigma=sigma,
                     verbosity=verbosity)
    nd.calculate_final_exact_pdf_from_structure(
        structure_or_file=structure)
    nd.export_exact_pdf(file_name=output_file)
    fig = nd.plot_exact_pdf()
    plt.show()

#########################################################################
######################  TEST - MAIN #####################################
#########################################################################

if __name__ == '__main__':
    cli()

