"""
aiida_user_utils.nmr module

User utilities to work with computed and experimental NMR data within AiiDA

Author: Sylvian Cadars
Institution: Institute for Reasearch on Ceramics, CNRS, University of Limoges, France


USAGE:
    from aiida_user_utils import nùr

"""

from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
import os
import numpy as np
from pymatgen.analysis.nmr import ChemicalShielding, ElectricFieldGradient
from ase.io import read, write
from ase.atoms import Atoms
from pyama.utils import visualize_clusters, get_ase_atoms, get_pymatgen_structure
import json
import re

# The following imposes that torch is installed alongside pyama...
import torch
# The step below is need to avoid UnpicklingError due to torch/mace/e3nn version incompatibility
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass
from e3nn.io import CartesianTensor


def decompose_tensor(full_tensor, check_reconstruction=False,
                     verbosity=1):
    """
    Decompose a generic rank-2 tensor using e3nn o3 convention
    Scalar part:
    scalar_0e = T.trace() / sqrt(3)

    Antisymmetric pseudovector 1e part
    A=0.5 * (T-T^T)
    antisymm_1e = (1/np.sqr(2)) * [A[2,1], A[0,2], A[1,0]]

    SYmmetric traceless 2e:
    S = 0.5 * (T+T^T)
    Q = S - (1/3) * T.trace() * np.eye(3)
    q1 = (1/sqrt(2)) * Qxx-Qyy
    q2 = (1/sqrt(6)) * (2*Qzz-Qxx-Qyy)
    q3 = (1/sqrt(2)) * Qxy
    q4 = (1/sqrt(2)) * Qxz
    q5 = (1/sqrt(2)) * Qyz

    irrep_decomposed_vect = np.concatenate([scalar_0e], antisymm_1e, [q1, q2, q3,q4, q5]]

    """

    T = torch.tensor(full_tensor)
    cart = CartesianTensor('ij')  # for a generic tensor
    decomposed_vector = np.array(cart.from_cartesian(T))

    if check_reconstruction:
        reconstructed_tensor = reconstruct_tensor(decomposed_vector)
        if not np.all(np.isclose(reconstructed_tensor, full_tensor)):
            raise ValueError(f"Reconstructed_tensor:\n{reconstructed_tensor} does not match "
                             f"with original tensor:\n{full_tensor}")
        elif verbosity >= 2:
            print("Reconstructed_tensor and original tensor match")

    if verbosity >= 1:
        print((f"Tensor:\n{full_tensor} decomposes into irreducible-representation "
               f"vector:\n{decomposed_vector}"))

    return decomposed_vector


def reconstruct_tensor(decomposed_tensor: np.ndarray,
                       verbosity: int=1) -> np.ndarray:   # shape (5,) → [t1,…,t5]
    """
    Re‑assemble a 3×3 tensor from its o3("0e + 1e + 2e") components.

    Parameters
    ----------
    decomposed_tensor: np.ndarray
        Concatenation of trace (1), antisymmetric (3) and traceless_symmetric (5)
        components, corresponding to shape (9,).

    return_all_tensors_by_rank: bool (default is False)
        Whether all-rank tensors (V^(0), V^(1), V^(2)) should be returned in addition
        to the recontructed full tensor.

    Returns
    -------
    np.ndarray
        Reconstructed tensor of shape (3, 3).
    """
    decomposed_tensor = torch.tensor(decomposed_tensor)

    cart = CartesianTensor('ij')
    reconstructed_tensor = np.array(cart.to_cartesian(decomposed_tensor))
    if verbosity >= 1:
        print(f"Irrep-decomposed tensor {decomposed_tensor} reconstructs into "
              f"full tensor: {reconstructed_tensor}")

    return reconstructed_tensor


def decompose_atomic_tensors(atomic_tensors=None, atoms=None, array_name="nmr_cs_tensors"):

    if atomic_tensors is None:
        atomic_tensors = atoms.arrays[array_name]
    else:
        atomic_tensors = torch.tensor(atomic_tensors)

    cart = CartesianTensor('ij')
    decomposed_tensors = cart.from_cartesian(atomic_tensors)

    return np.array(decomposed_tensors)


def reconstruct_cartesian_tensors(irreps_decomposed_tensors=None, atoms=None,
                                  array_name="nmr_cs_tensors"):

    if irreps_decomposed_tensors is None:
        irreps_decomposed_tensors = atoms.arrays[array_name]
    else:
        irreps_decomposed_tensors = torch.tensor(irreps_decomposed_tensors)

    cart = CartesianTensor('ij')
    cartesian_tensors = cart.to_cartesian(irreps_decomposed_tensors)

    return np.array(cartesian_tensors)


def add_smoothing_lorentzo_gaussian(peak_positions, peak_intensities=None,
                                    fwhm=1.0, eta=0.5, frequencies=None,
                                    steps=1024, normalize=True, x_rel_margins=0.10):
    """
    Add a combination of Lorentzian and Gaussan broadening to an NMR spectrum

    Args:
        fwhm: float (default is 1.0)
            Full width at half maximum in ppm.
        eta: float (default is 0.5
            Lorentzian to Gaussian ratio.
    """
    peak_positions = np.array(peak_positions)

    # Calculate frequency scale automatically
    if frequencies is None:
        x_min = np.max(peak_positions)
        x_max = np.min(peak_positions)
        x = np.linspace(x_min - x_rel_margins*(x_max-x_min), x_max + x_rel_margins*(x_max-x_min),
                       num=steps)
    elif len(frequencies) == 2:
        x = np.linspace(np.min(frequencies), np.max(frequencies),
                           num=steps)
    else:
        x = np.array(frequencies)

    H = fwhm  # fwhm in 2(theta) units
    if eta < 0 or eta > 1:
        raise ValueError('WARNING : eta should be between 0 and 1.')

    if peak_intensities is None:
        peal_intensities = np.ones(peak_positions.shape)
    # Initialize y
    y = np.zeros(len(x))
    for x_0 in peak_positions:
        # Gaussian broadening
        G = (2/H)*np.sqrt(np.log(2)/np.pi) * np.exp(-(4*np.log(2)/(H*H)) * np.square(x - x_0))
        # Gaussian broadening
        L = 2/(np.pi * H) / (1 + 4/(H*H) * np.square(x - x_0))
        y += eta * L + (1 - eta) * G
    
    if normalize:
        y = y / np.max(y)

    return x, y


def calc_nmr_spectrum(nmr_iso, fwhm=1.0, eta=0.5,
                      frequencies=None, steps=1024, x_rel_margins=0.10):
    """
    TODO: add default parameters
    """
    x, y = add_smoothing_lorentzo_gaussian(nmr_iso, fwhm=fwhm, eta=eta,
        frequencies=frequencies, steps=steps, x_rel_margins=x_rel_margins)
    return x, y


def get_iso_nmr_spectrum(structure_or_file, atom_type,
                         nmr_tensors=None, nmr_tensors_key=None,
                         are_irreps_decomposed_tensors=True,
                         frequencies=None, steps=1024,
                         fwhm=1.0, eta=0.5, sigma_to_delta=None, 
                         verbosity=1):
    """
    Get isotropic NMR spectrum with braodeing from a structure containing (or a separate array of) NMR tensors
    """
    if isinstance(structure_or_file, str):
        ext = os.path.splitext(structure_or_file)[-1]
        if verbosity >= 2:
            print(f"Reading ASE Atoms from file {structure_or_file} with extension {ext}")
        if ext in [".xyz", ".extxyz"]:
            ase_atoms = read(structure_or_file, format="extxyz")
            print(f"ASE Atoms of length {len(ase_atoms)} read from file {structure_or_file} "
                  f"in extended xyz format")
        else:
            ase_atoms = read(structure_or_file)
    else:
        ase_atoms = get_ase_atoms(structure_or_file)
    
    # TODO: use update_structure_with_nmr_data
    if nmr_tensors is None:
        if nmr_tensors_key is None:
            raise ValueError(f"Set either nmr_tensors or an nmr_tensors_key corresponding to "
                             f"a valid key in ase_atoms.arrays (currently includes: "
                             f"{list(ase_atoms.arrays.keys())})")
        else:
            nmr_tensors = ase_atoms.arrays[nmr_tensors_key]

    if verbosity >= 1:
        print(f"NMR {atom_type} isotropic CS spectrum will be plotted for "
              f"{ase_atoms.get_chemical_formula()} structure.")

    print(f"DEBUG: ase_atoms.info: {ase_atoms.info}")
    print(f"DEBUG: ase_atoms.cell: {ase_atoms.cell}")

    # TODO: select relevant sites matching atom_type and
    mapping = {
        'site_index': [],
        'index_by_type': [],
        'nmr_iso': [],
        'position': [],
    }
    index_by_type = -1
    for site_index, atom in enumerate(ase_atoms):
        if atom.symbol == atom_type:
            index_by_type += 1
            mapping['site_index'].append(site_index)
            mapping['index_by_type'].append(index_by_type)
            mapping['position'].append(atom.position)

    selected_indexes = np.array(mapping['site_index'])
    if are_irreps_decomposed_tensors:
        cartesian_tensors = reconstruct_cartesian_tensors(nmr_tensors[selected_indexes])
    else:
        cartesian_tensors = nmr_tensors[selected_indexes]
    
    nmr_iso = []
    for T in cartesian_tensors:
        cs = ChemicalShielding(T)
        nmr_iso.append(cs.mehring_values.sigma_iso)

    nmr_iso = np.array(nmr_iso)

    if sigma_to_delta is not None:
        nmr_iso = (sigma_to_delta['slope'] * nmr_iso) +  sigma_to_delta['intercept']

    x, y = add_smoothing_lorentzo_gaussian(nmr_iso, fwhm=fwhm, eta=eta,
                                           frequencies=frequencies, steps=steps, normalize=False)
    
    return x, y, ase_atoms, nmr_iso, mapping



def plot_iso_nmr_spectrum(structure_or_file, atom_type,
                          nmr_tensors=None, nmr_tensors_key=None,
                          are_irreps_decomposed_tensors=True,
                          frequencies=None, steps=1024,
                          fwhm=1.0, eta=0.5, show_clusters=True,
                          cluster_cutoff_radius=4.0, show_plot=True,
                          sigma_to_delta=None, title=None, 
                          plot_local_env_subspectra=True, 
                          sorted_site_labels_and_counts=None, 
                          sorted_site_labels_and_counts_file=None, 
                          max_local_env_proportion=0.01, 
                          max_n_local_envs = 10, 
                          y_rel_shift = -1.05, 
                          full_spectrum_scale = 1.0, 
                          verbosity=1):
    """
    Load an ASE Atoms structure and associated or included NMR tensors and plot

    TODO: describe expected sorted_site_labels_and_counts architecture:

    Args:
        TO BE COMPLETED

    Returns:

    """
    x, y, ase_atoms, nmr_iso, mapping = get_iso_nmr_spectrum(structure_or_file, atom_type,
        nmr_tensors=nmr_tensors, nmr_tensors_key=nmr_tensors_key,
        are_irreps_decomposed_tensors=are_irreps_decomposed_tensors,
        frequencies=frequencies, steps=steps,
        fwhm=fwhm, eta=eta, 
        sigma_to_delta=sigma_to_delta, verbosity=verbosity)

    pmg_struct = get_pymatgen_structure(ase_atoms)

    sum_full_spec = np.sum(y)
    fig, ax = plt.subplots(1, 1)

    full_spectrum_label = '{} simulated spectrum'.format(atom_type)
    if full_spectrum_scale is not None and full_spectrum_scale != 1:
        full_spectrum_label += " (scale: {full_spectrum_scale})"
    
    [line] = ax.plot(x, y * full_spectrum_scale, 'r-', label=full_spectrum_label)

    sites_y = np.interp(nmr_iso, x[::-1], y[::-1])

    peak_pos_lines = []
    for i, site_index in enumerate(mapping['site_index']):
        _lines = ax.plot([nmr_iso[i], nmr_iso[i]], [0, sites_y[i] * full_spectrum_scale],
                         '-', color='grey', linewidth=0.5, picker=True,
                         label=f"site_index: {site_index}, index_by_type: {mapping['index_by_type'][i]}")
        peak_pos_lines.append(_lines[0])

    if sigma_to_delta is not None:
        ax.set_xlabel('{} isotropic chemical shift (ppm)'.format(atom_type))
        ax.xaxis.set_inverted(True)
    else:
        ax.set_xlabel('{} isotropic shielding (ppm)'.format(atom_type))
        ax.xaxis.set_inverted(True)

    if plot_local_env_subspectra:
        # Open file:
        if sorted_site_labels_and_counts is None and sorted_site_labels_and_counts_file is not None:
            with open(sorted_site_labels_and_counts_file, "r") as f:
                sorted_site_labels_and_counts = json.load(f)
        elif sorted_site_labels_and_counts is None:
            raise NotImplementedError("sorted_site_labels_and_counts_file should be provided until "
                                      "local env exploration is implemented in pyama.")
        subspec_lines = []
        subspec_texts = []
        subspec_peak_pos_lines = []            
        y_shift = 0
        for env_index, site_label in enumerate(
                sorted_site_labels_and_counts[atom_type]["site_labels"]):
            env_proportion = sorted_site_labels_and_counts[atom_type]["proportions"][env_index]
            if env_proportion < max_local_env_proportion or env_index >= max_n_local_envs:
                if verbosity >= 1:
                    print(f"No further {atom_type} will be considered.")
                    break
            env_count = sorted_site_labels_and_counts[atom_type]["counts"][env_index]
            site_indexes = np.array(sorted_site_labels_and_counts[atom_type]["site_indexes"][env_index])
            indexes_by_type = np.array(sorted_site_labels_and_counts[atom_type]["indexes_by_type"][env_index])
            if verbosity >= 1:
                print(f"Plotting {atom_type} subspectrum for {env_count} {site_label} environments "
                      f"({100 * env_proportion:.2f} %)")

            sel_nmr_iso = nmr_iso[indexes_by_type]
            x, y = add_smoothing_lorentzo_gaussian(sel_nmr_iso, fwhm=fwhm, eta=eta,
                frequencies=frequencies, steps=steps, normalize=False)
            
            # TODO: normalize vs env_proportion * sum_full_spec / sum(y)
            # Adjust area (in case spextra are normalized)
            y *= env_proportion * sum_full_spec / np.sum(y)

            sel_sites_y = np.interp(sel_nmr_iso, x[::-1], y[::-1])

            max_y = np.max(y)
            y_shift += y_rel_shift * max_y
            min_x = np.min(x)
            max_x = np.max(x) 

            for i, site_index in enumerate(site_indexes):
                _lines = ax.plot([sel_nmr_iso[i], sel_nmr_iso[i]], 
                                 [y_shift, sel_sites_y[i] + y_shift],
                                '-', color='grey', linewidth=0.5, picker=True,
                                label=f"site_label: {site_label}, site_index: {site_index}, index_by_type: {mapping['index_by_type'][i]}")
                subspec_peak_pos_lines.append(_lines[0])

            _lines = ax.plot(x, y + y_shift, ls="-", marker="none", label=f"{site_label} sites")
            color = _lines[0].get_color()
            text = f"{site_label} ({100 * env_proportion:.1f} %))"
            subspec_lines.append(_lines[0])
            
            # TODO: alternate lext/right text position
            txt = ax.text(min_x + 0.1 * (max_x - min_x), 
                                         y_shift + 0.05 * max_y, text, color=color)
            subspec_texts.append(txt)

            # TODO: add pickable individual-site lines

    ax.set(ylabel='Intensity (AU)')

    if not title:
        ax.set(title='{} CS_iso'.format(atom_type))
    else:
        ax.set(title=title)

    # Add sliders to adjust eta and fwhm
    fig.subplots_adjust(bottom=0.30)
    axfreq = fig.add_axes([0.20, 0.15, 0.70, 0.03])
    fwhm_slider = Slider(ax=axfreq, label='FWHM (ppm)',
        valmin=0.1, valmax=50, valinit=fwhm,
    )

    axfreq = fig.add_axes([0.20, 0.10, 0.70, 0.03])
    eta_slider = Slider(ax=axfreq, label='eta (L/G ratio)',
        valmin=0.0, valmax=1.0, valinit=eta,
    )

    axfreq = fig.add_axes([0.20, 0.05, 0.70, 0.03])
    r_cut_slider = Slider(ax=axfreq, label='cluster r_cut (A)',
        valmin=0.5, valmax=10.0, valinit=cluster_cutoff_radius,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        y = calc_nmr_spectrum(atom_type, nmr_iso, fwhm_slider.val,
            eta_slider.val, frequencies, steps)
        line.set_ydata(y)
        sites_y = np.interp(nmr_iso, x[::-1], y[::-1])
        for i, peak_pos_line in enumerate(peak_pos_lines):
            peak_pos_line.set_ydata([0, sites_y[i]])

        fig.canvas.draw_idle()

    # register the update function with each slider
    fwhm_slider.on_changed(update)
    eta_slider.on_changed(update)

    def onpick(event):
        # TODO: identify artist to introduce a different behavior for lines picked in subspectra
        line = event.artist
        print(f"Picked line label: {line.get_label()}")
        # ind = event.ind
        position = line.get_xdata()[0]
        
        if plot_local_env_subspectra and line in subspec_peak_pos_lines:
            pattern = r"site_label:\s*([A-Za-z0-9-]+),\s*site_index:\s*(\d+),\s*index_by_type:\s*(\d+)"
            match = re.match(pattern, line.get_label())
            if match:
                site_label = match.group(1)
                site_index = int(match.group(2))
                index_by_type = int(match.group(3))
            else:
                raise ValueError("Regex pattern does not match with picked line label.")
        
            print(f"Picked {site_label} site {site_index} at {position} ppm (index_by_type: {index_by_type})")

        elif line in peak_pos_lines:
            pattern = r"site_index:\s*(\d+),\s*index_by_type:\s*(\d+)"
            match = re.match(pattern, line.get_label())
            if match:
                site_index = int(match.group(1))
                index_by_type = int(match.group(2))
            else:
                raise ValueError("Regex pattern does not match with picked line label.")
            
            print(f"Picked site {site_index} at {position} ppm (index_by_type: {index_by_type})")
        
        # Retrieve site index and index_by_type from labal
        
        # TODO: explore local environment and/or show cluster
        if show_clusters:
            visualize_clusters(pmg_struct, [site_index], r_cut=r_cut_slider.val)

    fig.canvas.mpl_connect('pick_event', onpick)

    if show_plot:
        plt.show()

    return fig, ax


def plot_multi_iso_nmr_spectra(structures_or_files, atom_type,
                            nmr_tensors_list=None, nmr_tensors_key=None,
                            are_irreps_decomposed_tensors=True,
                            frequencies=None, steps=1024,
                            fwhm=1.0, eta=0.5, show_plot=True,
                            sigma_to_delta=None, title=None, 
                            descriptions=None,
                            y_rel_shift=-1.05, 
                            normalize_to_first=False, 
                            verbosity=1):
    """
    Plot multiple NMR spectra from structure (or files) containing (or with a separate list of) NMR tensors

    TO BE COMPLETED
    """
    fig, ax = plt.subplots()
    y_shift = 0
    lines = []
    for struct_index, structure_or_file in enumerate(structures_or_files):
        if nmr_tensors_list is not None:
            nmr_tensors = nmr_tensors_list[struct_index]
        else:
            nmr_tensors = None
        x, y, ase_atoms, nmr_iso, mapping = get_iso_nmr_spectrum(structure_or_file, atom_type,
            nmr_tensors=nmr_tensors, nmr_tensors_key=nmr_tensors_key,
            are_irreps_decomposed_tensors=are_irreps_decomposed_tensors,
            frequencies=frequencies, steps=steps,
            fwhm=fwhm, eta=eta, 
            sigma_to_delta=sigma_to_delta, verbosity=verbosity)

        # Normalize (add an option to normalize all spectra to the first)
        if normalize_to_first:
            if struct_index == 0:
                first_spec_sum = np.sum(y)
            y /= first_spec_sum
        else:
            y /= np.max(y)

        description = None
        if descriptions is None:
            for description_key in ["system_name", "description"]:
                if description_key in ase_atoms.info:
                    description = ase_atoms.info[description_key]
                    break
            if description is None:
                description = ase_atoms.get_chemical_formula()
        else:
            description = descriptions[struct_index]

        lines.append(ax.plot(x, y + y_shift, label=description))

        # TODO: add the possibility to plot on the spectrum to get info on underlying sites
        # including local env if available

        y_shift += y_rel_shift * np.max(y)

    ax.xaxis.set_inverted(True)
    
    if sigma_to_delta is not None:
        ax.set_xlabel('{} isotropic chemical shift (ppm)'.format(atom_type))    
    else:
        ax.set_xlabel('{} isotropic shielding (ppm)'.format(atom_type))

    ax.set(ylabel='Intensity (AU)')

    if not title:
        ax.set(title='{} CS_iso'.format(atom_type))
    else:
        ax.set(title=title)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    return fig, ax


def update_structure_file_with_nmr_data(struct_or_file, 
                                        cs_tensors_irreps=None,
                                        cs_tensors_cartesian=None, 
                                        cs_tensors_irreps_key="nmr_cs_tensors", 
                                        cs_basename="nmr_cs", 
                                        efg_tensors_irreps=None,
                                        efg_tensors_cartesian=None, 
                                        efg_tensors_irreps_key="nmr_efg_tensors",
                                        efg_basename="nmr_efg", 
                                        xyz_file_suffix="standard_nmr", 
                                        updated_structure_file=None, 
                                        replace_arrays=False, 
                                        verbosity=1):
    """
    Create (or duplicate) an ASE Atoms instance with experimentally-relevant NMR parameters in atoms.arrays

    TODO: Add possibility to update ASE Atoms arrays rather than overwrite, to compile results obtained
          independently for different atom types.

    TODO: Add chemical shift for all atom types with a shielding_to_shift correction in format:
        {
            "H": {
                "slope": SLOPE, 
                "intercept": INTERCEPT
            }, 
            "C": {
                "slope": SLOPE, 
                "intercept": INTERCEPT
            },
        }
    """
    if isinstance(struct_or_file, str) and updated_structure_file is None:
        folder, basemame = os.path.split(os.path.abspath(struct_or_file))
        basename_noext = os.path.splitext(basemame)[0]
        updated_structure_file = os.path.join(folder, f"{basename_noext}_{xyz_file_suffix}.extxyz")
    elif not updated_structure_file:
        raise ValueError('Updated_structure_file must be provided struct_or_file is not a structure file.')
    else:
        updated_structure_file = os.path.abspath(updated_structure_file)

    if isinstance(struct_or_file, Atoms):
        ase_atoms = struct_or_file
    else:
        ase_atoms = get_ase_atoms(struct_or_file)
   
    # Get cartesian chemical-shielding tensors from arguments or ase_atoms.arrays
    # Convert from irreps-decomposed to cartesian idf 
    if cs_tensors_cartesian is not None:
        pass
    elif cs_tensors_irreps is not None:
        cs_tensors_cartesian = reconstruct_cartesian_tensors(cs_tensors_irreps)
    elif cs_tensors_irreps_key in ase_atoms.arrays:
        cs_tensors_cartesian = reconstruct_cartesian_tensors(ase_atoms.arrays[cs_tensors_irreps_key])
    
    if cs_tensors_cartesian is not None:
        cs_sigma_iso_list = []
        cs_haeberlen_delta_sigma_list = []
        cs_haeberlen_eta_list = []
        cs_haeberlen_zeta_list = []
        cs_maryland_omega_list = []
        cs_maryland_kappa_list = []
        cs_mehring_sigma_11_list = []
        cs_mehring_sigma_22_list = []
        cs_mehring_sigma_33_list = []
        
        lists = [cs_sigma_iso_list, 
                 cs_haeberlen_delta_sigma_list, cs_haeberlen_eta_list, cs_haeberlen_zeta_list, 
                 cs_maryland_omega_list, cs_maryland_kappa_list, 
                 cs_mehring_sigma_11_list, cs_mehring_sigma_22_list, cs_mehring_sigma_33_list, 
                 ]
        
        not_nan_indexes = []
        for site_index, T in enumerate(cs_tensors_cartesian):
            cs = ChemicalShielding(T)
            if np.any(np.isnan(ChemicalShielding(cs))):
                [l.append(np.nan) for l in lists]
            else:
                not_nan_indexes.append(site_index)
                cs_sigma_iso_list.append(cs.haeberlen_values.sigma_iso)
                cs_haeberlen_delta_sigma_list.append(cs.haeberlen_values.delta_sigma_iso)
                cs_haeberlen_eta_list.append(cs.haeberlen_values.eta)
                cs_haeberlen_zeta_list.append(cs.haeberlen_values.zeta)
                cs_maryland_omega_list.append(cs.maryland_values.omega)
                cs_maryland_kappa_list.append(cs.maryland_values.kappa)
                cs_mehring_sigma_11_list.append(cs.mehring_values.sigma_11)
                cs_mehring_sigma_22_list.append(cs.mehring_values.sigma_22)
                cs_mehring_sigma_33_list.append(cs.mehring_values.sigma_33)
        
        not_nan_indexes = np.array(not_nan_indexes, dtype=int)
        # Add arrays to ASE Atoms
        
        keys = [f"{cs_basename}_{prop}" for prop in [
                                                        "sigma_iso", 
                                                        "haeberlen_delta_sigma", 
                                                        "haeberlen_eta", 
                                                        "haeberlen_zeta", 
                                                        "maryland_omega", 
                                                        "maryland_kappa", 
                                                        "mehring_sigma_11", 
                                                        "mehring_sigma_22", 
                                                        "mehring_sigma_33"
                                                    ]]

        for key, l in zip(keys, lists):
            if key in ase_atoms.arrays and not replace_arrays:
                ase_atoms.arrays[key][not_nan_indexes] = np.array(l)[not_nan_indexes]
            else:
                ase_atoms.arrays[key] = np.array(l)
        

    elif verbosity >= 1:
        print(f"No chemical shielding tensors provided or found for structure {ase_atoms.get_chemical_formula()}")

    # Get cartesian electricv-field-gradient tensors from arguments or ase_atoms.arrays
    # Convert from irreps-decomposed to cartesian 
    # TODO: A specific reconstruction 
    if efg_tensors_cartesian is not None:
        pass
    elif efg_tensors_irreps is not None:
        raise NotImplementedError("The conversion from irreps(2e)-decomposed EFG tensors to cartesian "
                                  "is not yet implemented...")
        # efg_tensors_cartesian = reconstruct_cartesian_tensors(efg_tensors_irreps)
    elif efg_tensors_irreps_key in ase_atoms.arrays:
        raise NotImplementedError("The conversion from irreps(2e)-decomposed EFG tensors to cartesian "
                                  "is not yet implemented...")
        # efg_tensors_cartesian = reconstruct_cartesian_tensors(ase_atoms.arrays[efg_tensors_irreps_key])
        
    if efg_tensors_cartesian is not None:
        efg_v_xx_list = []
        efg_v_yy_list = []
        efg_v_zz_list = []
        efg_asymmetry_list = []  
        # TODO: implement C_Q based on latest Q values

        lists = [efg_v_xx_list, efg_v_yy_list, efg_v_zz_list, efg_asymmetry_list]
                 
        not_nan_indexes = []
        for site_index, T in enumerate(efg_tensors_cartesian):
            efg = ElectricFieldGradient(T)
            if np.any(np.isnan(efg)):
                [l.append(np.nan) for l in lists]
            else:
                not_nan_indexes.append(site_index)
                efg_v_xx_list.append(efg.V_xx)
                efg_v_yy_list.append(efg.V_yy)
                efg_v_zz_list.append(efg.V_zz)
                efg_asymmetry_list.append(efg.asymmetry)
                # TODO: add C_Q
        
        # Add arrays to ASE Atoms
        ase_atoms.arrays[f"{efg_basename}_v_xx"] = np.array(efg_v_xx_list)
        ase_atoms.arrays[f"{efg_basename}_v_yy"] = np.array(efg_v_yy_list)
        ase_atoms.arrays[f"{efg_basename}_v_zz"] = np.array(efg_v_zz_list)
        ase_atoms.arrays[f"{efg_basename}_asymmetry"] = np.array(efg_asymmetry_list)

        keys = [f"{cs_basename}_{prop}" for prop in ["v_xx", "v_yy", "v_zz", "asymmetry"]]

        for key, l in zip(keys, lists):
            if key in ase_atoms.arrays and not replace_arrays:
                ase_atoms.arrays[key][not_nan_indexes] = np.array(l)[not_nan_indexes]
            else:
                ase_atoms.arrays[key] = np.array(l)

    elif verbosity >= 1:
        print(f"No electric-field gradient tensors provided or found for structure {ase_atoms.get_chemical_formula()}")    
    
    if efg_tensors_cartesian is None and cs_tensors_cartesian is None:
        raise ValueError("No chemical shielding or EFG information found.")
    
    write(updated_structure_file, ase_atoms, format="extxyz")
    
    print(f"Updated ASE Atoms structure with NMR data in arrays['{cs_basename}_[...]'] and/or "
          f"arrays['{efg_basename}_[...]'] was stored as {updated_structure_file}")
    
    return ase_atoms, updated_structure_file        


def find_equivalent_sites(input_array, site_types=None, delta=None, eta=None, atol=1e-3, 
                          selected_atom_type=None, local_envs=None, verbosity=1):
    """
    Group sites based on the values in an array of dimension (n_sites, M)
    
    Values should be an array of dimension (n_sites, n_properties).
    A typical example would be a concatenation of 3 (n_sites,) arrays
    containing sigma_iso, delta_sigma, eta chemical shielding values.

    Args:
        TO BE COMPLETED:
    """
    keys = np.round(input_array / atol).astype(np.int64)

    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    index_groups = [
        np.flatnonzero(inverse == k)
        for k in np.unique(inverse)
    ]
    
    groups = {}  # {group_label: site_indexes}
    group_label_counts = {}
    for site_indexes in index_groups:
        if (site_types is not None and selected_atom_type is not None
            and site_types[site_indexes[0]] != selected_atom_type):
            if verbosity >= 2:
                print(f"Skipping group of {site_types[site_indexes[0]]} sites: {site_indexes}")
            continue
        
        if local_envs is not None:
            group_label = local_envs[site_indexes[0]]
            for i in site_indexes[1:]:
                if local_envs[i] != local_envs[site_indexes[0]]:
                    raise ValueError(f"Sites {site_indexes[0]} and {i} have similar "
                                     f"values but different local_envs: {local_envs[i]} "
                                     f"and {local_envs[site_indexes[0]]}")
        elif site_types is not None:
            group_label = site_types[site_indexes[0]]
        else:
            group_label = "group"
    
        if group_label not in group_label_counts:
            group_label_counts[group_label] = 0
        else:
            group_label_counts[group_label] += 1
            
        groups[f"{group_label}-{group_label_counts[group_label]}"] = site_indexes
    
    return groups


