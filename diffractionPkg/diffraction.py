# -*- coding: utf-8 -*-
"""
Module to calculate the exact total X-ray scattering as described by Masson & Thomas

The implementation is based on :
Masson O. and Thomas P., J. Appl. Cryst. (2013). 46, 461–465

Created on Wed Nov 18 09:30:52 2020

@author: Sylvian Cadars, Institut de Recherche sur les Céramiques,
        CNRS, Université de Limoges, France
"""
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction. xrd import XRDCalculator

import numpy as np
import matplotlib.pyplot as plt
import sys


def add_smoothing_lorentzo_gaussian(pattern, fwhm=0.2, eta=0.5,
                                    two_theta_range=[0, 120],
                                    step=None):
    """
    Add pseudo-Voigt broadening to a diffraction pattern (pymatgen)

    Args:
        pattern:
            pymatgen diffraction pattern as obtained with XRDCalculator
            or NDCalculator
        fwhm: float (default=0.2)
            Full width at half maximum in units of 2*theta (°)
        eta: float (default=0.5)
            Lorentzian to Gaussian ratio (1 for pure Lorentzian, 0 for
            pure Gaussian profile)
        two_theta_range:
        step: float or None
            two_theta step for the x axis. Default is 0.1*fwhm

    Returns:
        (x, y): tuple
            x:  two_theta axis between two_theta_range[0] and
                two_theta_range[1] with step.
            y:  simulated diffracted intensities at x.
    """
    H = fwhm  # fwhm in 2(theta) units
    if eta < 0 or eta > 1:
        print('WARNING : eta should be between 0 and 1. Using the default (0.5).')
        eta = 0.5
    if step==None:
        step = 0.1*H

    x = np.arange(two_theta_range[0], two_theta_range[1], step)
    x_hkl = pattern.x
    I_hkl = pattern.y
    # Initialize y
    y = np.zeros(len(x))
    for i in range(len(x_hkl)):
        # Gaussian broadening
        G = (2/H)*np.sqrt(np.log(2)/np.pi) * np.exp(-(4*np.log(2)/(H*H)) * np.square(x - x_hkl[i]))
        # Gaussian broadening
        L = 2/(np.pi * H) / (1 + 4/(H*H) * np.square(x - x_hkl[i]))
        y += I_hkl[i] * (eta * L + (1 - eta) * G)
    # Normalize:
    y = y/max(y)

    return x, y


def plot_diffraction_pattern(structure_or_file,
                              diffraction_type='X-ray',
                              wavelength=None,
                              two_theta_range=(0, 80),
                              fwhm=0.2,
                              eta=0.5,
                              plot_title=None,
                              experimental_file=None,
                              experimental_title=None, 
                              y_shift = 1.05):
    """
    Plot neutron or X-ray diffractogram for a pymatgen Structure or a file
    """
    if isinstance(structure_or_file, Structure):
        pmg_struct=structure_or_file
    elif isinstance(structure_or_file, str):
        pmg_struct=Structure.from_file(structure_or_file)

    # Plot simulated neutron or x-ray diffraction pattern with Pymatgen
    if diffraction_type.lower() in ['xrd', 'x-ray', 'x-rays']:
        diffraction_type_name = 'X-ray diffraction'
        if wavelength is None:
            wavelength = 'CuKa'
        calculator=XRDCalculator(wavelength=wavelength, symprec=0,
                                 debye_waller_factors=None)
    elif diffraction_type.lower() in ['nd', 'neutron', 'neutrons']:
        diffraction_type_name = 'neutron diffraction'
        if wavelength is None:
            wavelength = 2.522
        calculator=NDCalculator(wavelength=wavelength, symprec=0,
                                debye_waller_factors=None)
    else:
        sys.exit('diffraction_type should be neutron/ND or X-ray/XRD.')

    pattern=calculator.get_pattern(pmg_struct, scaled=True,
                                   two_theta_range=two_theta_range)

    x, y = add_smoothing_lorentzo_gaussian(pattern, fwhm=fwhm,
        eta=eta, two_theta_range=two_theta_range)
    fig, ax = plt.subplots()
    # ax.plot(pattern.x, pattern.y, color='blue')

    ax.plot(x, y, color='blue')
    legend = ['Simulation : {}'.format(pmg_struct.formula)]

    if experimental_file is not None:
        # Load experimental ND pattern
        expt_data = np.loadtxt(experimental_file,delimiter=None,skiprows=0)
        # Adding experimental ND data to plot
        ax.plot(expt_data[:,0], expt_data[:,1]/max(expt_data[:,1])+
                 y_shift*max(y),color='black')
        if experimental_title is None:
            legend.append('Experiment')
        else:
            legend.append(experimental_title)

    if isinstance(plot_title, str):
        ax.set_title(plot_title)
    else:
        ax.set_title(('Simulated {} pattern for {}').format(
            diffraction_type_name, pmg_struct.formula))
    ax.set(xlabel='2\u03B8 (°)',ylabel='Normalized Intensity (AU)',
           xlim=two_theta_range)
    ax.legend(legend,loc='upper left', bbox_to_anchor= (1.0, 1.0))
    plt.tight_layout()
    plt.show()

    return fig


