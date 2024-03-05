# -*- coding: utf-8 -*-
"""
Module to calculate the exact total X-ray scattering as described by Masson & Thomas

The implementation is based on :
Masson O. and Thomas P., J. Appl. Cryst. (2013). 46, 461–465

Created on Wed Nov 18 09:30:52 2020

@author: Sylvian Cadars, Institut de Recherche sur les Céramiques,
        CNRS, Université de Limoges, France
"""
import numpy as np
import structureComparisonsPkg.distanceTools as dt
import diffractionPkg.utils as dutils
import diffractionPkg.atomicFormFactors as aff
import copy
import matplotlib.pyplot as plt
import scipy.fft

def calculate_total_xray_scattering(structure,kmax=20,waveLength=1.5406,**kwargs) :
    """
    A function to calculate the total xray scattering for an input structure

    IMPORTANT NOTE : the partial radiatial distribution functions g_ij(r) used
        in Masson & Thomas in fact corresponds to the reduced RDF, generally 
        designated as G_ij(r) and obtained as :
            G_ij(r) = 4*pi*r*rho_0*(g_ij(r)-1)
        where g_ij(r) is the partial RDF and rho_0 is the system average
        density (in atoms/volume Units). The multiplicatio by r increases the
        weight of the RDF at longer range, making it more sensitive to medium-
        range order.    

    TODO : check what is wrong with the calculation of w_ij coefficents 
            (slightly different from the paper for some reason...)

    TODO : include the convolution product by taking the Fourier transform and
            multiplying by a step function (=1 for Q < Qmax and =0 beyond) to 
            then Fourier transform back.

    Parameters
    ----------
    structure : pymatgen.core.structure.IStructure or Structure object
        DESCRIPTION.
    kmax : int, OPTIONAL (default = 20)
        Maximum number of steps for the Fourier expansion
    waveLength : float, OPTIONAL (default : 1.5406, in Angstroms)
        
    **kwargs : optional arguments include :
        Q : 
        Qmax :
        Qmin :
        Qsteps :
        Qstep :
        r :
        rMax :
        rMin :
        rSteps :
        rStep :
        showPlot :

    Returns
    -------
    totalScatteringFunction : numpy.ndarray
        Array of length len(Q).
    Q : numpy.ndarray
        DESCRIPTION.

    """
    
    if 'showPlot' in kwargs :
        showPlot = kwargs['showPlot']
    else:
        showPlot = False
    
    print('Hello from calculate_total_xray_scattering function')
    
    Q,Qmin,Qmax,Qstep,Qsteps = dutils.get_Q_vector(**kwargs)
    r_initial,rMin,rMax,rStep,rSteps = dutils.get_r_vector(**kwargs)
    
    r_k = (np.pi/Qmax)*np.arange(kmax+1)
    # Create a copy of requested r vector and expand it up to rMax+max(r_k)
    r = copy.copy(r_initial)
    while r[-1] < r_initial[-1]+r_k[-1] :
        r = np.append(r,r[-1 ]+rStep)
    print('r extended to [',r[0],'...',r[-1],'] of length ',r.size,
          ' to allow calculation of g_ij(r+r_k)')
    
    listOfSpecies = list(structure.symbol_set)
    nbOfSpecies = len(listOfSpecies)

    # Calculate atomic factor with dispersion corrections : 
    #    f = f0(Q,E) + f1(E) + 1j*f2(E)
    # with E = E_eV property of formFactor object
    listOfFormFactors,listOfSpecies = \
        aff.get_list_of_formFactor_objects_from_list_of_species(
            listOfSpecies,Q=Q,waveLength=waveLength,
            plotDispersionCorrections=showPlot,plotf0=showPlot)
    
    f = np.zeros((len(listOfSpecies),len(Q)),dtype=float)   
    for specieIndex,ff in enumerate(listOfFormFactors):
        f[specieIndex][:] = ff.get_corrected_form_factor()
        print('f_',listOfSpecies[specieIndex],'[0] = ',f[specieIndex][0])
    
    # Creating distanceMatrixData class object for the calculation of partial pdfs.
    dmd = dt.distanceMatrixData(sigma=0.05,R=r)
    
    comp = structure.composition.element_composition
    
    # Get concentrations of species and calculate
    # |<f(Q)>|^2 = ( sum_over_species_i(c_i*f_i(Q)) )^2
    # as a matrix of dimension (nbOfSpecies,len(Q))
    c = np.asarray(
        [comp.get_atomic_fraction(specieName) for specieName in listOfSpecies])
    cArray = np.transpose(np.resize(c,(len(Q),nbOfSpecies)))
    f_av_square = np.square( np.sum(np.abs(cArray*f),axis=0) )
       
    # Initializing variables for a loop over pairs of species i,j taking into
    # account that (i,j) and (j,i) will be identical
    # -> retrieve individual indexes with : pair_index = (nbOfSpecies*i)+j
    # -> Retrieve (i,j) from pair index : j = mod(pair_index,
    gamma = np.zeros( (nbOfSpecies*nbOfSpecies,len(Q)) )
    a = np.zeros( (nbOfSpecies*nbOfSpecies,kmax+1) )
    
    # Create matrix of cos(Q*r_k) values with dimension (Q.size,r_k.size)
    cosQr_k = np.cos( np.transpose(np.resize(Q,(r_k.size,Q.size))) * \
                     np.resize(r_k,(Q.size,r_k.size)) )
    
    # Initializing expression (15a) in Masson & Thomas, i.e. totalScattering
    # function before convolution by sin(Qmax*r)/(pi*r)
    eq15a = np.zeros(r_initial.size,dtype=float)
    eq18 = np.zeros(r_initial.size,dtype=float)
    sum_of_a_ij_r_k = np.zeros(len(r_k),dtype=float)
    
    if showPlot == True :
        # Initializing plots of partial RDFs and gamm_ij weighting factors
        fig1 = plt.figure('Partial RDFs')
        ax1 = fig1.add_subplot(1, 1, 1)
        shift = 0.0
        legend1=[]
        
        fig4 = plt.figure('gamma_ij weighting fators')
        ax4 = fig4.add_subplot(1, 1, 1)
        legend4=[]
    
    pair_index = 0
    # Loop on (i,j) species
    for i_index,i_specie in enumerate(listOfSpecies) :
        # Loop over second species up to i_index because for all quantities x_ij = x_ji
        for j_index,j_specie in enumerate(listOfSpecies[:i_index+1]) :             
            f_i = f[i_index][:]
            f_j = f[j_index][:]
            
            #TODO : solve the issue below when using f with dispersion corrections
            # WARNING : This part is NOT symmetric with respect to (i,j)
            # permutation : f_j*conj(f_i) = conj(f_j*conj(f_i))
            gamma = c[i_index]*c[j_index] * f_i*np.conj(f_j) / f_av_square
            
            # Calculate reduced partial radial distribution function (RDF) 
            # G_ij(r) = 4*pi*rho_0*r*(g_ij(r)-1)
            # with r extended up to max(r_initial)+max(r_k)
            G_ij = dmd.calculate_reduced_partial_RDF(structure,i_specie,
                                                     j_specie,showPlot=False)

            if showPlot == True :
                ax1.plot(r,G_ij+shift)
                legend1.append('G_'+i_specie+'_'+j_specie)
                shift += max(G_ij)
                
                ax4.plot(Q,gamma)
                # To 
                if i_index == j_index :
                    ax4.plot(Q,gamma)
                    legend4.append('\u03B3_'+i_specie+'_'+j_specie)
                elif j_index < i_index:
                    ax4.plot(Q,2*gamma)
                    legend4.append('\u03B3_'+i_specie+'_'+j_specie+' + \u03B3_'+\
                                   j_specie+'_'+i_specie)
            
            # Create matrix of gamma_ij(Q)*cos(Qr_rk) values of dim (Q.size,r_k.size)
            gammaCosQr_k = np.transpose(np.resize(gamma,(r_k.size,Q.size))) * \
                cosQr_k
            
            # integrate  gamma_ij(Q)*cos(Qr_rk) btwn 0 and Qmax and multiply by
            # 1/Qmax is equivalent to taking the average gamma_ij(Q)*cos(Qr_rk)
            # over the Q interval
            # ??? Is this correct ???
            a_ij = np.average(gammaCosQr_k , axis=0 )
            
            # eq15a += a_ij[0]*G_ij[:rSteps]
            if j_index < i_index :
                eq15a += 2*a_ij[0]*G_ij[:rSteps]
            elif j_index == i_index :
                eq15a += a_ij[0]*G_ij[:rSteps]

            gX_ij = G_ij[:rSteps]
            
            sum_of_a_ij_r_k[0] += a_ij[0]
            for k_index,rkval in enumerate(r_k[1:]) :
                k = k_index+1
                # Calculate partial RDF g_ij(r-r_k) and g_ij(r+rk) using interpolation
                # Question : should right be 1 instead of 0 since lim_r->inf(g) = 1 ?
                G_r_plus_rk = np.interp(r_initial+rkval,r,G_ij,left=0,right=0)
                G_r_minus_rk = np.interp(r_initial-rkval,r,G_ij,left=0,right=0)
                
                # eq15a += a_ij[k_index]*(G_r_minus_rk + G_r_plus_rk)
                if j_index < i_index :
                    eq15a += 2*a_ij[k_index]*(G_r_minus_rk + G_r_plus_rk)
                    sum_of_a_ij_r_k[k] += 2*a_ij[k]
                elif j_index == i_index :
                    eq15a += a_ij[k_index]*(G_r_minus_rk + G_r_plus_rk)
                    sum_of_a_ij_r_k[k] += a_ij[k]
                    
                w_ij_rk = a_ij[k]/a_ij[0]
                print('w_'+i_specie,'-'+j_specie+'(r_'+str(k)+') = '+\
                      str(w_ij_rk)+' ; r_'+str(k)+' = '+str(rkval))
                gX_ij += w_ij_rk*(G_r_minus_rk + G_r_plus_rk)
                
            # None of this but eq15a is re-used afterwards. No need to store 2D variables.                        
            pair_index = pair_index+1
    
            if j_index < i_index :
                eq18 += 2*a_ij[0]*gX_ij
            elif j_index == i_index :
                eq18 += a_ij[0]*gX_ij
    
    print('**** Verification : *****\nsum over (i,j) species of a_ij[k] :\n',
          sum_of_a_ij_r_k,'\n********************************')
    
    # TODO : (work in progress...)
    # Convolution by sin(Qmax*r)/(pi*r) to obtain final result of equation (15)
    # Fourier transform, multiply by step function (=1 for Q<=Qmax, =0 beyond)
    # then Fourier-tranform back
    # Consider a "door" function f = 1 in the [-1/2 1/2] interval and 0 outside
    # the FT(f) = sin(pi*x)/(pi*x)
    # The same for a door step in the f2(x) = 1 for x in [-Qmax Qmax] ; O otherwise
    # FT(f(2*Qmax*x)) = sin(2*pi*Qmax)/(2*pi*Qmax)
    Q_fftLength = scipy.fft.next_fast_len(len(Q))
    FT = scipy.fft.fft(eq15a,n=Q_fftLength)
    Q_padded = scipy.fft.fftfreq(Q_fftLength,rStep)
    print('Q of length ',len(Q),' padded to ',len(Q_padded),
          ' for Fourrier transform')       
    r_ifftLength = 2*scipy.fft.next_fast_len(len(r_initial))
    IFT = scipy.fft.ifft(FT,n=r_ifftLength)
    r_padded = scipy.fft.fftfreq(r_ifftLength,Q_padded[1]-Q_padded[0])      
    
    
    if showPlot == True :
        ax1.set(xlabel='r (Angstroms)', ylabel='G_ij',
                title='Partial radial distribution functions G_ij')
        ax1.legend(legend1)
        
        fig2 = plt.figure('Equations (15a) and (18)')
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(r_initial,eq15a)
        ax2.plot(r_initial,eq18+0.1*max(eq15a))
        ax2.set(xlabel='r (Angstroms)', ylabel='??',
                title='equations (15a) and (18) before convolution by sin(Qmax*r)/(pi*r)')
        ax2.legend(['equation (15a)','equation (18)'])
        
        fig3 = plt.figure('Fourier transform of eq15a')
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot(Q_padded,FT)
        ax3.set(xlabel='Q (1/A)', ylabel='FT(Q)',
                title='Fourier transform of eq15a')
        
        fig5 = plt.figure('iff(fft(eq15a)')
        ax5 = fig5.add_subplot(1, 1, 1)
        ax5.plot(r_padded,IFT)
        ax5.set(xlabel='r (A)', ylabel='IFFT(FT(eq15a)))(r)',
                title='iff(fft(eq15a)',xlim=[0,max(r_initial)])
        
        ax4.set(xlabel='Q (1/Angstroms)', ylabel='gamma_ij(Q)',
                title='Weighting factors gamma_ij')
        ax4.legend(legend4)
        
        plt.show()
    
    totalScattering = np.array(len(Q),dtype=float)
    return totalScattering,Q


"""



"""