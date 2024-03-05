#!/usr/local/miniconda/envs/aiida/bin/python3
"""
Build a polymer atom by atom in a periodic cell

Version using scipy fsolve to find positions of atoms in the polyhedrons
Problems of infinite loops with sympy.solve()

"""

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.sites import Site, PeriodicSite
from pymatgen.core.bonds import get_bond_length

from ase.visualize import view

import numpy as np
from scipy.optimize import fsolve
import click
import sys
import os

"""
import sympy as sym
sym.init_printing()
"""


def class polymerBuilderData():
    """
    Class containing all methods and properties necessary to run polymerBuilder

    properties:
        seed
        rng
        species_properties
        system
        structure
        abs_bond_length_tol
        rel_bond_length_tol
        verbosity

    """



    # initialize:





def visualize(pymatgen_structure, visualizer='ase'):

    if visualizer.lower() == 'ase':
        ase_struct = AseAtomsAdaptor.get_atoms(pymatgen_structure)
        view(ase_struct)
    if visualizer.lower() == 'vesta':
        pymatgen_structure.to(fmt='cif', filename='tmp.cif')
        import subprocess
        sp = subprocess.run(['vesta', 'tmp.cif'], capture_output=True)
        print(sp)


def get_type_from_index(structure, index):
    atom_type = structure.sites[index].species.elements[0].name
    return atom_type


def get_atom_type(site):
    atom_type = site.species.elements[0].name
    return atom_type


def get_bond_angle_from_CN(coord_number):
    if coord_number == 2:
        theta = 180.0
    elif coord_number == 3:
        theta = 120.0
    elif coord_number == 4:
        theta = 109.4712206
    elif coord_number == 6:
        theta = 90.0
    else:
        raise ValueError
    return theta


def get_bond_length_matrix(atom_types, system):
    bond_lengths_matrix = np.zeros(2*[len(system['atom_types'])])
    for i in range(len(system['atom_types'])):
        for j in range(i+1):
            bond_lengths_matrix[i, j] = float(get_bond_length(system['atom_types'][i],
                                                              system['atom_types'][j]))
            if j != i:
                bond_lengths_matrix[j, i] = bond_lengths_matrix[i, j]
    return bond_lengths_matrix


def get_remaining_atoms_by_type(atom_types, nb_of_atoms_by_type, structure,
                                as_dict=False):
    """
    get number of atoms of each type to be inserted in structure

    Args:
        atom_types: list or tuple of str (or int for atomic numbers)
            List of atom types to be inserted in the structure
        nb_of_atoms_by_type: list or tuple of int
            Targeted number of atoms of this type in the final structure
        structure: pymatgen.core.structure.Structure object

        as_dict: bool (default is False)
            if True the function will return a dict mapping of the form
            {
                'atomic_type_1': nb_of_atoms_of_type_1,
                ...
            }

    returns:
        - a list of number of remainingining atoms in the same order as
          atom_types
        - a mapping of atom_type: nb_of_remaining_atoms
    """
    # Make a copy to avoid decreasing system['nb_of_atoms_by_type']
    remaining_atoms = nb_of_atoms_by_type.copy()
    if as_dict:
        mapping = {}
    for type_index, atom_type in enumerate(atom_types):
        for site in structure.sites:
            if site.species.elements[0].name.lower() == atom_type.lower():
                remaining_atoms[type_index] -= 1
        if as_dict:
            mapping[atom_type] = remaining_atoms[type_index]

    if as_dict:
        remaining_atoms = mapping
    return remaining_atoms

# Definition of local functions using the user variables
def pick_coord_number_from_type(at_type, species_properties, rng):
    """
    (a, size=None, replace=True, p=None, axis=0, shuffle=True)
    """
    [coord_number] = rng.choice(
        list(range(len(species_properties[at_type]['coord_proba']))),
        size=1, p=species_properties[at_type]['coord_proba'])
    print('Picked coord number for {} atom: {}.'.format(at_type, coord_number))
    # TODO: Reproducible random with:
    # rng = numpy.random.default_rng()
    # rng.choice()
    return coord_number


def pick_type_from_neighbor_type(neighbor_type, system, species_properties,
                                 rng):
    [atom_type_index] = rng.choice(
        list(range(len(species_properties[neighbor_type]['clustering_proba']))),
        size=1, p=species_properties[neighbor_type]['clustering_proba'])
    atom_type = system['atom_types'][atom_type_index]
    print('Picked type index for neighbor of {} atom: {}.'.format(atom_type,
          neighbor_type))
    return atom_type


def initialize_site_properties(site, **kwargs):
    """
    Initialize the custom properties of a site

    One may use a kwarg to change a specific property to non-default value
    """
    if not isinstance(site, (PeriodicSite, Site)):
        raise(TypeError,
              'Argument site should be of type pymatgen.core.site.(Periodic)Site.')
    site.properties['is_shell_complete'] = False
    site.properties['is_treated'] = False
    site.properties['treatment_attempts'] = 0
    site.properties['target_coord_number'] = None
    site.properties['connected_neighbors'] = []

    return site


def get_connected_neighbors(structure, site_index, search_radius=2.5,
                            bond_length_abs_tol=1e-5,
                            bond_length_rel_tol=0, verbosity=1):
    """
    Find neighbors whose distance to center match the expected bond_length

    Tolerance on bond length may be set based on absolute (in Angstroms)
    or relative value (in fraction of the expected bond length)
    """
    neighbors = structure.get_neighbors(structure.sites[site_index],
                                        search_radius)
    connected_neighbors = []
    for nbr in neighbors:
        dist = np.linalg.norm(nbr.coords - structure.sites[site_index].coords)
        site_type = get_type_from_index(structure, site_index)
        if np.isclose(dist, float(get_bond_length(site_type, get_atom_type(nbr))),
                      atol=bond_length_abs_tol, rtol=bond_length_rel_tol):
            if verbosity >= 2:
                print(('Found {} ({}) atom connected to {} ({}) (at {} \u212B).'
                   ).format(get_atom_type(nbr), nbr.index,
                            site_type, site_index, dist))
            connected_neighbors.append(nbr)

    return connected_neighbors


def build_current_site_shell(structure, site_index,
                             rel_contact_tol=0.0, abs_contact_tol=0.10,
                             contact_search_radius=2.5, MAX_ATTEMPTS=10,
                             NUMERIC_TOLERANCE=1e-6, verbosity=1):
    """
    Construct a polyhedron around selected site based on coordination number
    """
    # Pick target coordination number if none has been selected
    if structure.sites[site_index].properties['target_coord_number'] is None:
        structure.sites[site_index].properties['target_coord_number'] = \
            pick_coord_number_from_type(get_type_from_index(structure,
                                                            site_index),
                                        species_properties, rng)
    site_coord_number = structure.sites[site_index].properties[
        'target_coord_number']


    # Find neighbors connected to site_index
    struct, nbrs = update_connected_neighbors(structure, site_index,
        search_radius=contact_search_radius, verbosity=verbosity)

    while len(nbrs) < site_coord_number and (
            struct.sites[site_index].properties['treatment_attempts'] <
            MAX_ATTEMPTS):
        (structure, nbrs) = add_neighbor(structure, site_index,
                                         site_coord_number, nbrs,
                                         rel_contact_tol=rel_contact_tol,
                                         abs_contact_tol=abs_contact_tol,
                                         contact_search_radius=contact_search_radius,
                                         NUMERIC_TOLERANCE=NUMERIC_TOLERANCE,
                                         verbosity=verbosity)
    if struct.sites[site_index].properties['treatment_attempts'] >= MAX_ATTEMPTS:
        struct.sites[site_index].properties['is_treated'] = True
        print(('Max number of attemps ({}) to complete the {}-coordinated shell of '
               'site {} ({}) has been reached. Switching to another site.').format(
                MAX_ATTEMPTS, site_coord_number, site_index,
                get_type_from_index(structure, site_index)))

        # TODO: print message if MAX_ATTEMPS reached

    if len(nbrs) == site_coord_number:
        if verbosity >= 1:
            print(('Number of neighbors connected to atom {}{} ' +
                   'match coordination number : {}').format(
                   site_index, get_type_from_index(structure, site_index),
                   site_coord_number))
        struct.sites[site_index].properties['is_treated'] = True

    return structure


def add_neighbor(structure, site_index, site_coord_number, nbrs,
                 abs_contact_tol=0.0, rel_contact_tol=0.1,
                 contact_search_radius=2.5, verbosity=1,
                 NUMERIC_TOLERANCE=1e-6):
    """
    Add a neighbor with a method that suits coordination number

    Arguments:
        structure: pymatgen.core.structure.Structure
            Structure object
        site_index: int
            Considered site index
        site_coord_number: int
            Coordination number of the considered site
        nbrs: list of pymatgen.core.structure.Neighbor
            aleardy-identified connected neighbors of the considered site
        abs_contact_tol: float (default: 0.0)
            absolute tolerance in angstrom added to expected bond length between
            detected neigbor and tried position to calcultate the contact criterion.
        rel_contact_tol: float (default: 0.1)
            tolerance in fraction of expected bond length
        contact_search_radius: float (default: 2.5)
            search radius to find contact with tried position insertion
        verbosity: int (default: 1)
            verbosity level

    Returns:
        structure: pymatgen.core.structure.Structure
            Updated structure
        nbrs: list of pymatgen.core.structure.Neighbor
            Updated list of identified connected neighbors
    """
    if len(nbrs) == 0:
        structure, nbrs = add_first_neighbor(structure, site_index,
                                             site_coord_number, nbrs,
                                             abs_contact_tol=abs_contact_tol,
                                             rel_contact_tol=rel_contact_tol,
                                             contact_search_radius=contact_search_radius,
                                             verbosity=verbosity)
    elif len(nbrs) == 1:
        structure, nbrs = add_second_neighbor(structure, site_index,
                                              site_coord_number, nbrs,
                                              abs_contact_tol=abs_contact_tol,
                                             rel_contact_tol=rel_contact_tol,
                                             contact_search_radius=contact_search_radius,
                                             NUMERIC_TOLERANCE=NUMERIC_TOLERANCE,
                                             verbosity=verbosity)
    elif len(nbrs) == 2:
        structure, nbrs = add_third_neighbor(structure, site_index,
                                             site_coord_number, nbrs,
                                             abs_contact_tol=abs_contact_tol,
                                             rel_contact_tol=rel_contact_tol,
                                             contact_search_radius=contact_search_radius,
                                             NUMERIC_TOLERANCE=NUMERIC_TOLERANCE,
                                             verbosity=verbosity)
    elif len(nbrs) == 3:
        structure, nbrs = add_fourth_neighbor(structure, site_index,
                                              site_coord_number, nbrs,
                                              abs_contact_tol=abs_contact_tol,
                                             rel_contact_tol=rel_contact_tol,
                                             contact_search_radius=contact_search_radius,
                                             NUMERIC_TOLERANCE=NUMERIC_TOLERANCE,
                                             verbosity=verbosity)
    return structure, nbrs


def is_space_clear(structure, tried_type, tried_position, known_nbr_indexes=None,
                   abs_tolerance=0.0, rel_tolerance=0.10, search_radius=2.5,
                   verbosity=1):
    """
    Check that no atom exist within distances defined by types

    Parameters:
        structure: pymatgen.core.structure.Structure
        tried_type: str
            type of the atom to be inserted. Will be used to determine
            the contact criteria along with type of detected neighbors
        tried_type: str
            type of the atom to be inserted. Will be used to determine
            the contact criteria along with type of detected neighbors
        known_nbr_indexes: int, list or tuple
            site indexes that should be ignored in the search for contacts
        abs_tolerance: float (default: 0.0)
            absolute tolerance in angstrom added to expected bond length between
            detected neigbor and tried position to calcultate the contact criterion.
        rel_tolerance: float(default: 0.1)
            tolerance in fraction of expected bond length

    returns:
        Bool: False if no contact other than with known_nbr_indexes
              are found
    """
    is_clear = True
    nearby_atoms = structure.get_sites_in_sphere(tried_position, search_radius,
                                                 include_index=True)
    if verbosity >= 3:
        print('{} atoms detected within {} \u212B of tried position {}'.format(
              len(nearby_atoms), search_radius, tried_position))
    # Convert known_nbr_indexes to list if single element
    if not isinstance(known_nbr_indexes, (list, tuple)):
        known_nbr_indexes = [known_nbr_indexes]

    for (site, dist, index) in nearby_atoms:
        if index not in known_nbr_indexes:
            contact_dist = (1 + rel_tolerance) * \
                float(get_bond_length(get_atom_type(site), tried_type)) + \
                abs_tolerance
            if dist < contact_dist:
                if verbosity >= 2:
                    print('Atom {}{} within {} \u212B of tried position {}.'.format(
                          index, get_atom_type(site), contact_dist,
                          tried_position))
                is_clear = False
                break
    return is_clear


def update_connected_neighbors(structure, site_index, search_radius=2.5,
                               include_neighbors=True, verbosity=1):
    """
    Update connected_neighbors property of listed sites
    """
    nbrs = get_connected_neighbors(struct, site_index,
                                   search_radius=search_radius)
    structure.sites[site_index].properties['connected_neighbors'] = [
        nbr.index for nbr in nbrs]

    if verbosity >= 2:
        print('Site {} ({}) connected_neighbors property updated : {}'.format(
              site_index, get_type_from_index(structure, site_index),
              structure.sites[site_index].properties['connected_neighbors']))

    if include_neighbors:
        return struct, nbrs
    else:
        return struct


def add_first_neighbor(struct, site_index, site_coord_number,
                       nbrs, verbosity=1, contact_search_radius=2.5,
                       abs_contact_tol=0.0, rel_contact_tol=0.1,
                       max_local_attempts=10):
    """
    Add a first neighbor to a selected site in the structure
    """
    # Designate site of focus by O for practicity
    index_O = site_index
    X_O = struct.sites[index_O].coords
    type_O = get_type_from_index(struct, index_O)

    type_A = pick_type_from_neighbor_type(type_O)
    remaining_atoms = get_remaining_atoms_by_type(system['atom_types'],
                                                  system['nb_of_atoms_by_type'],
                                                  struct,
                                                  as_dict=True)[type_A]
    if remaining_atoms <= 0:
        print(('No atoms of type {} left. First neighbor of site {} ({}) ' +
               'has not been inserted.').format(type_A, index_O, type_O))
        struct.sites[index_O].properties['treatment_attempts'] += 1
        return struct, nbrs

    elif verbosity >= 2:
        print(('Picking type {} (out of {} remaining) for the 1st neighbor ' +
               'of site {} ({}).').format(type_A, remaining_atoms, index_O,
              type_O))

    # place A atom along a random direction V
    OA = float(get_bond_length(type_O, type_A))
    if verbosity >= 2:
        print('OA distance set to {}-{} bond length: {} A.'.format(type_A,
              type_O, OA))

    local_attempts = 0
    while 1:  # local attempts : trying different A positions withouht changing type_A
        if local_attempts >= max_local_attempts:
            struct.sites[index_O].properties['treatment_attempts'] += 1
            return struct, nbrs
        V = rng.standard_normal(3)
        V = V / np.linalg.norm(V)
        OA_vect = V*OA
        X_A = X_O + OA_vect
        if is_space_clear(struct, type_A, X_A, [index_O],
                          abs_tolerance=abs_contact_tol,
                          rel_tolerance=rel_contact_tol,
                          search_radius=contact_search_radius,
                          verbosity=verbosity):
            break
        else:
            local_attempts += 1

    try:
        struct.insert(len(struct.sites), type_A, X_A,
                      coords_are_cartesian=True, validate_proximity=True)
        # proximity validation uses static struct.DISTANCE_TOLERANCE
        index_A = len(struct.sites) - 1
        struct.sites[index_A] = initialize_site_properties(
            struct.sites[index_A])
        if site_coord_number == 1:
            struct.sites[index_O].properties['is_shell_complete'] = True
            struct.sites[index_O].properties['is_treated'] = True
        # Update list of connected neighbors
        struct, nrbs = update_connected_neighbors(struct, index_O,
            search_radius=contact_search_radius, verbosity=verbosity)

    except ValueError as e:
        print('WARNING: Atom A not inserted, presumably due to proximity issue: ',
              e)
        struct.sites[index_O].properties['treatment_attempts'] += 1

    return struct, nbrs


def add_second_neighbor(struct, site_index, site_coord_number,
                        nbrs, verbosity=1, contact_search_radius=2.5,
                        abs_contact_tol=0.0, rel_contact_tol=0.1,
                        NUMERIC_TOLERANCE=1e-6):

    # Designate site of focus by O and 1st neighbor by A for practicity
    index_O = site_index
    X_O = struct.sites[index_O].coords
    type_O = get_type_from_index(struct, index_O)
    index_A = nbrs[0].index
    X_A = nbrs[0].coords
    type_A = get_atom_type(nbrs[0])

    # Picking type_B based on type_O, exiting function if none remaining
    type_B = pick_type_from_neighbor_type(type_O)
    remaining_atoms = get_remaining_atoms_by_type(system['atom_types'],
                                                  system['nb_of_atoms_by_type'],
                                                  struct,
                                                  as_dict=True)[type_B]
    if remaining_atoms <= 0:
        print(('No atoms of type {} left. 2nd neighbor of site {} ({}) ' +
               'has not been inserted.').format(type_B, index_O, type_O))
        struct.sites[index_O].properties['treatment_attempts'] += 1
        return struct, nbrs

    elif verbosity >= 2:
        print(('Picking type {} (out of {} remaining) for the 2nd neighbor ' +
               'of site {} ({}).').format(type_B, remaining_atoms, index_O,
              type_O))


    # Set level below which imaginary parts will be considered 0 in units of
    # machine epsilon (used by numpy).
    imag_tolerance = NUMERIC_TOLERANCE / np.finfo(float).eps

    # create a random temporary point T to define the AOT plane
    X_T = X_O + rng.standard_normal(3)
    OA_vect = X_A - X_O
    OA = np.linalg.norm(OA_vect)
    n = np.cross(X_T - X_O, OA_vect)  # Normal vector of AOT plane
    # Calculate a, b, c, d parameters in plane equation: ax + by + cz + d = 0
    (a, b, c) = n / np.linalg.norm(n)
    d = -(a*X_T[0] + b*X_T[1] + c*X_T[2])

    theta = (np.pi/180)*get_bond_angle_from_CN(site_coord_number)
    # B is along a vector v such that v.OA = |v|*|OA|*cos(theta)
    # (x_B, y_B, z_B) = sym.symbols('x_B y_B z_B')
    OB = float(get_bond_length(type_B, type_O))
    if verbosity >= 2:
        print('OB distance set to {}-{} bond length: {} \u212B.'.format(type_B,
              type_O, OB))

    def equations(X):
        x, y, z = X
        f1 = a*x + b*y + c*z + d  # B is in OAT plane
        f2 = (x-X_O[0])*OA_vect[0] + (y-X_O[1])*OA_vect[1] + \
             (z-X_O[2])*OA_vect[2] - OB*OA*np.cos(theta)
        f3 = (x-X_O[0])**2 + (y-X_O[1])**2 + (z-X_O[2])**2 - OB**2
        return (f1, f2, f3)

    if verbosity >= 2:
        print(('Looking for solutionsofor atom B of type {} ' +
               'around site {} ({})').format(type_B,
              index_O, type_O))
    solution = fsolve(equations, (0, 0, 0))
    if verbosity >= 2:
        print('Solution found : {}.'.format(solution))

    X_B = None
    X_B_try = np.asarray(solution, dtype=float)
    if verbosity >= 2:
        print('Trying to insert B atom of type {} at position {}.'.format(
              type_B, X_B_try))
    if is_space_clear(struct, type_B, X_B_try, [index_O, index_A],
                      search_radius=contact_search_radius,
                      abs_tolerance=abs_contact_tol,
                      rel_tolerance=rel_contact_tol,
                      verbosity=verbosity):
        try:

            struct.insert(len(struct.sites), type_B, X_B_try,
                          coords_are_cartesian=True,
                          validate_proximity=True)
            X_B = X_B_try
            index_B = len(struct.sites)-1
            OB_vect = X_B - X_O
            if verbosity >= 1:
                print('Site {}:{} added at position {}.'.format(index_B,
                      type_B, X_B))
            struct.sites[index_B] = initialize_site_properties(
                 struct.sites[index_B])
            if site_coord_number == 2:
                struct.sites[index_O].properties['is_shell_complete'] = True
                struct.sites[index_O].properties['is_treated'] = True
            # Update list of connected neighbors
            struct, nrbs = update_connected_neighbors(struct, index_O,
                search_radius=contact_search_radius, verbosity=verbosity)

        except ValueError as e:
            print('B atom could not be inserted at position {}. ValueError: {}'.format(
                  X_B_try, e))
    else:
        if verbosity >= 2:
            print('Solution leads to contact between atoms.')

    if X_B is None:
        print('Exiting function add_second_neighbor with no site B added.')
        struct.sites[index_O].properties['treatment_attempts'] += 1
        # struct is unchanged. No need to update nbrs.

    return struct, nbrs


def add_third_neighbor(struct, site_index, site_coord_number,
                        nbrs, verbosity=1, contact_search_radius=2.5,
                        abs_contact_tol=0.0, rel_contact_tol=0.1,
                        NUMERIC_TOLERANCE=1e-6):

    # Designate site of focus, and 1st and 2nd neighbors by O, A and B for practicity
    index_O = site_index
    X_O = struct.sites[index_O].coords
    type_O = get_type_from_index(struct, index_O)
    # shuffle A-B neighbors order
    pick_order = rng.choice(range(2), replace=False, size=2, shuffle=False)
    index_A = nbrs[pick_order[0]].index
    X_A = nbrs[pick_order[0]].coords
    type_A = get_atom_type(nbrs[pick_order[0]])
    index_B = nbrs[pick_order[1]].index
    X_B = nbrs[pick_order[1]].coords
    type_B = get_atom_type(nbrs[pick_order[1]])

    # Set level below which imaginary parts will be considered 0 in units of
    # machine epsilon (used by numpy).
    imag_tolerance = NUMERIC_TOLERANCE / np.finfo(float).eps

    # Pick type_C based on type_O, exiting function if none remaining
    type_C = pick_type_from_neighbor_type(type_O)
    remaining_atoms = get_remaining_atoms_by_type(system['atom_types'],
                                                  system['nb_of_atoms_by_type'],
                                                  struct,
                                                  as_dict=True)[type_C]
    if remaining_atoms <= 0:
        print(('No atoms of type {} left. 3rd neighbor of site {} ({}) ' +
               'has not been inserted.').format(type_C, index_O, type_O))
        struct.sites[index_O].properties['treatment_attempts'] += 1
        return struct, nbrs

    elif verbosity >= 2:
        print(('Picking type {} (out of {} remaining) for the 3rd neighbor ' +
               'of site {} ({}).').format(type_C, remaining_atoms, index_O,
              type_O))

    theta = (np.pi/180)*get_bond_angle_from_CN(
    site_coord_number)
    OC = float(get_bond_length(type_O, type_C))
    OA_vect = X_A - X_O
    OA = np.linalg.norm(OA_vect)
    OB_vect = X_B - X_O
    OB = np.linalg.norm(OB_vect)

    # (x, y, z) = sym.symbols('x y z')

    def equations(v):
        (x, y, z) = v
        f1 = x*OA_vect[0] + y*OA_vect[1] + z*OA_vect[2] - OC*OA*np.cos(theta)
        f2 = x*OB_vect[0] + y*OB_vect[1] + z*OB_vect[2] - OC*OB*np.cos(theta)
        f3 = x**2 + y**2 + z**2 - OC**2
        return f1, f2, f3

    if verbosity >= 2:
        print(('Looking for solution to place atom C of type {} '
              'around site {} ({})...').format(type_C, index_O, type_O))
    solution = fsolve(equations, (0, 0, 0))
    if verbosity >= 2: # debugging only
        print('Solution found: {}'.format(solution))
    X_C = None
    X_C_try = np.asarray(solution, dtype=float) + X_O
    if is_space_clear(struct, type_C, X_C_try, [index_O, index_A, index_B],
                      search_radius=contact_search_radius,
                      abs_tolerance=abs_contact_tol,
                      rel_tolerance=rel_contact_tol,
                      verbosity=verbosity):
        try:
            struct.insert(len(struct.sites), type_C, X_C_try,
                          coords_are_cartesian=True, validate_proximity=True)
            X_C = X_C_try
            index_C = len(struct.sites)-1
            OC_vect = X_C - X_O
            if verbosity >= 1:
                print('Site {}:{} added at position {}.'.format(index_C, type_C,
                                                                X_C))
            struct.sites[index_C] = initialize_site_properties(struct.sites[index_C])
            if site_coord_number == 3:
                struct.sites[index_O].properties['is_shell_complete'] = True
                struct.sites[index_O].properties['is_treated'] = True
            # Update list of connected neighbors
            struct, nrbs = update_connected_neighbors(struct, index_O,
                search_radius=contact_search_radius, verbosity=verbosity)

        except ValueError as e:
            print(('C atom could not be inserted at position {}: ' +
                   '{}').format(X_C_try, e))
    else:
        if verbosity >= 2:
            print('Solution leads to contact between atoms.')

    if X_C is None:
        print('Exiting function build_current_site_shell with no site C added.')
        struct.sites[index_O].properties['treatment_attempts'] += 1
        # struct is unchanged. No need to update nbrs.

    return struct, nbrs


def add_fourth_neighbor(struct, site_index, site_coord_number,
                        nbrs, verbosity=1, contact_search_radius=2.5,
                        abs_contact_tol=0.0, rel_contact_tol=0.1,
                        NUMERIC_TOLERANCE=1e-6):

    # Designate site of focus, and 1st-3rd neighbors by O, A, B, and C
    # for practicity
    index_O = site_index
    X_O = struct.sites[index_O].coords
    type_O = get_type_from_index(struct, index_O)
    # shuffle A-B neighbors order
    pick_order = rng.choice(range(len(nbrs)), replace=False, size=len(nbrs),
                            shuffle=False)
    index_A = nbrs[pick_order[0]].index
    X_A = nbrs[pick_order[0]].coords
    type_A = get_atom_type(nbrs[pick_order[0]])
    index_B = nbrs[pick_order[1]].index
    X_B = nbrs[pick_order[1]].coords
    type_B = get_atom_type(nbrs[pick_order[1]])
    index_C = nbrs[pick_order[2]].index
    X_C = nbrs[pick_order[2]].coords
    type_C = get_atom_type(nbrs[pick_order[2]])

    # Set level below which imaginary parts will be considered 0 in units of
    # machine epsilon (used by numpy).
    imag_tolerance = NUMERIC_TOLERANCE / np.finfo(float).eps

    # Pick type_D based on type_O, exiting function if none remaining
    type_D = pick_type_from_neighbor_type(type_O)
    remaining_atoms = get_remaining_atoms_by_type(system['atom_types'],
                                                  system['nb_of_atoms_by_type'],
                                                  struct,
                                                  as_dict=True)[type_D]
    if remaining_atoms <= 0:
        print(('No atoms of type {} left. 4th neighbor of site {} ({}) ' +
               'has not been inserted.').format(type_D, index_O, type_O))
        struct.sites[index_O].properties['treatment_attempts'] += 1
        return struct, nbrs

    elif verbosity >= 2:
        print(('Picking type {} (out of {} remaining) for the 4th neighbor ' +
               'of site {} ({}).').format(type_D, remaining_atoms, index_O,
              type_O))

    theta = (np.pi/180)*get_bond_angle_from_CN(site_coord_number)
    OD = float(get_bond_length(type_O, type_D))
    # (x, y, z) = sym.symbols('x y z')
    OA_vect = X_A - X_O
    OA = np.linalg.norm(OA_vect)
    OB_vect = X_B - X_O
    OB = np.linalg.norm(OB_vect)
    OC_vect = X_C - X_O
    OC = np.linalg.norm(OC_vect)

    # Find vector of norm OD forming a theta angle with OA, OB and OC
    def equations(v):
        (x, y, z) = v
        f1 = x*OA_vect[0] + y*OA_vect[1] + z*OA_vect[2] - OD*OA*np.cos(theta)
        f2 = x*OB_vect[0] + y*OB_vect[1] + z*OB_vect[2] - OD*OB*np.cos(theta)
        f3 = x*OC_vect[0] + y*OC_vect[1] + z*OC_vect[2] - OD*OC*np.cos(theta)
        return f1, f2, f3

    if verbosity >= 2:
        print(('Looking for solution to place atom D of type {} '
              'around site {} ({})...').format(type_D, index_O, type_O))

    solution = fsolve(equations, (0, 0, 0))
    if verbosity >= 2:
        print('Solution found: {}.'.format(solution))

    X_D = None
    X_D_try = np.asarray(solution, dtype=float) + X_O
    if verbosity >= 2:
        print('Trying to insert D atom of type {} at position {}.'.format(
              type_D, X_D_try))
    if is_space_clear(struct, type_D, X_D_try,
                      [index_O, index_A, index_B, index_C],
                      search_radius=contact_search_radius,
                      abs_tolerance=abs_contact_tol,
                      rel_tolerance=rel_contact_tol,
                      verbosity=verbosity):
        try:
            struct.insert(len(struct.sites), type_D, X_D_try,
                          coords_are_cartesian=True, validate_proximity=True)
            index_D = len(struct.sites)-1
            X_D = X_D_try
            OD_vect = X_D - X_O
            if verbosity >= 1:
                print('Site {}:{} added at position {}.'.format(index_D, type_D,
                                                                X_D))
            struct.sites[index_D] = initialize_site_properties(struct.sites[index_D])
            if site_coord_number == 4:
                struct.sites[index_O].properties['is_shell_complete'] = True
                struct.sites[index_O].properties['is_treated'] = True
            # Update list of connected neighbors
            struct, nrbs = update_connected_neighbors(struct, index_O,
                search_radius=contact_search_radius, verbosity=verbosity)

        except ValueError as e:
            print(('D atom could not be inserted at position {}: ' +
                   '{}').format(X_D_try, e))

    else:
        if verbosity >= 2:
            print('Solution leads to contact between atoms.')

    if X_D is None:
        print('Exiting function build_current_site_shell with no site D added.')
        struct.sites[index_O].properties['treatment_attempts'] += 1

    return struct, nbrs


"""
def export_vasp_poscar(structure, rng_seed, dir_name):
    # order structure
    # create directory
    # save poscar
    # write system description in first line
    poscar = Poscar(structure,
                    comment='{} generated with seed={}'.format(
                        structure.get_formula(), rng_seed),
                    sort_structure = True))
    poscar.write_file(os.path.join())
"""


def main(SEED=None, VERBOSITY=1, ABSOLUTE_BOND_LENGTH_TOLERANCE=0.0,
         RELATIVE_BOND_LENGTH_TOLERANCE=0.1,
         MAX_ITERATIONS=1000, NUMERIC_TOL=1e-5, CONTACT_SEARCH_RADIUS=3.0,
         FIRST_ATOM_INTERN_COORDS=(0.5, 0.5, 0.5), VISUALIZER='vesta' ):


    # Define system
    species_properties = {
        'Si': {
            'coord_proba': [0, 0, 0, 0, 1],
            'clustering_proba': [0, 0.05, 0.95],
        },
        'C': {
            'coord_proba': [0, 0, 0, 0.2, 0.8],
            'clustering_proba': [0.025, 0.95, 0.025]
        },
        'N': {
            'coord_proba': [0, 0, 0, 0.7, 0.3],
            'clustering_proba': [0.95, 0.05, 0]
        },
    }

    system = {
        'cell_lengths': [12, 12, 12],  # formats accepted by pymatgen
        'cell_angles': [90, 90, 90],
        'atom_types': ['Si', 'C', 'N'],
        'nb_of_atoms_by_type': [40, 40, 40],  # Adjust to match typical densities
    }

    if SEED is None:
        seed = np.random.randint(1000)  # Store seed in program output
    elif isinstance(SEED, int):
        seed = SEED
    # seed = 873  # Set manually (e.g. to reproduce an unexpected behaviour or bug)

    # Precalculate matrix of bond lengths
    BOND_LENGTHS_MATRIX = get_bond_length_matrix(system['atom_types'])

    if VERBOSITY >= 2:
        print('Bond length matrix: \n{}'.format(BOND_LENGTHS_MATRIX))

    nb_of_atoms = sum(system['nb_of_atoms_by_type'])
    remaining_atoms_by_type = system['nb_of_atoms_by_type']

    # Intialize numpy random number generator (RNG) to ensure reproducibility
    print(f'Initialization of random number generator with seed: {seed}')
    rng = np.random.default_rng(seed)

    # Build system:
    lattice = Lattice.from_parameters(system['cell_lengths'][0],
                                      system['cell_lengths'][1],
                                      system['cell_lengths'][2],
                                      system['cell_angles'][0],
                                      system['cell_angles'][1],
                                      system['cell_angles'][2])

    # Construct structure with first atom
    # Pick atom type randomly with probabilities according to target system compo
    nb_of_atoms_by_type = np.asarray(system['nb_of_atoms_by_type'])
    [atom_type_index] = rng.choice(range(len(system['atom_types'])), size=1,
        p=nb_of_atoms_by_type / np.sum(nb_of_atoms_by_type))
    atom_type = system['atom_types'][atom_type_index]

    if FIRST_ATOM_INTERN_COORDS is None:
        struct = Structure(lattice=lattice, species=[atom_type],
                           coords=[rng.random(3)])
    else:
        struct = Structure(lattice=lattice, species=[atom_type],
                           coords=[FIRST_ATOM_INTERN_COORDS])

    # Initialize site properties
    struct.sites[0] = initialize_site_properties(struct.sites[0])

    iteration_index = -1
    while struct.num_sites < np.sum(nb_of_atoms_by_type):
        iteration_index += 1
        if iteration_index >= MAX_ITERATIONS:
            print(f'Maximum number of iterations ({MAX_ITERATIONS}) reached.')
            break
        if VERBOSITY >= 1:
            print(f'Starting iteration number {iteration_index}.')

        # pick first atom not-yet-treated
        sites_not_yet_treated = [index for (index, site) in enumerate(struct.sites)
                                 if (not site.properties['is_treated']) and
                                 (site.properties['treatment_attempts'] < MAX_ATTEMPTS)]
        if VERBOSITY >= 2:
            print('sites_not_yet_treated = ', sites_not_yet_treated)
        if len(sites_not_yet_treated):
            # Treat sites in order of creation until one type of atoms is exhausted
            remaining_atoms = get_remaining_atoms_by_type(system['atom_types'],
                system['nb_of_atoms_by_type'], struct)
            if min(remaining_atoms) > 0:
                current_site_index = sites_not_yet_treated[0]
            else:  # end then randomly
                [current_site_index] = rng.choice(sites_not_yet_treated, size=1)
        else:
            print('All sites have been treated. A new random site must be added.')
            print('IMPLEMENTATION IN PROGRESS.')
            break
            # Add a new atom from scratch at a random position.

        current_site_type = get_type_from_index(struct, current_site_index)
        # Set coord based on probabilities (easiest: ignore other sites)i
        current_site_CN = pick_coord_number_from_type(current_site_type)
        struct.sites[current_site_index].properties['target_coord_number'] = \
            current_site_CN
        if VERBOSITY >= 1:
            print('Current site[{}] : {} with coordination number {}.'.format(
                  current_site_index, current_site_type, current_site_CN))

        struct = build_current_site_shell(struct, current_site_index,
            rel_contact_tol=RELATIVE_BOND_LENGTH_TOLERANCE,
            abs_contact_tol=ABSOLUTE_BOND_LENGTH_TOLERANCE,
            contact_search_radius=CONTACT_SEARCH_RADIUS,
            verbosity=VERBOSITY,
            MAX_ATTEMPTS=MAX_ATTEMPTS)

        if VERBOSITY >= 2:
            print('\nStructure at the end of iteration {}:\n{}\n'.format(
                  iteration_index, struct))

    if VERBOSITY >= 1:
        print('\n' + 20*'*' + '\n' + 'Final structure :' + '\n' + 20*'*' + '\n',
              struct)
        print('Density = {:.3f} g/cm-3'.format(struct.density))

    visualize(struct,visualizer=VISUALIZER.lower())

# Set label of the group containing the structures on which analyses should be
# performed.
@click.command('cli')
@click.option('-r', '--rel_bond_length_tol', default=0.1, type=float,
              help='Relative contact tolerance in fraction of expected bond length')
@click.option('-m', '--max_iterations', default=1000, type=int,
              help='Maximum number of iterations.')
@click.option('-V', '--visualizer', default='vesta', type=str,
              help='Visualizer (vesta, ase)')
@click.option('-v', '--verbosity', default='1', type=int,
              help='Verbosity level. Default is 1')
def cli(group_label, base_label, relax_setting_label, data_description,
        plot_figure, verbosity):
    """
    Add code description here.
    """
    if not isinstance(rel_bond_length_tol, float) or rel_bond_length_tol < 0:
        sys.exit('rel_bond_length_tol should be a positive float')

    if not isinstance(max_iterations, int):
        sys.exit('rel_bond_length_tol should be a positive float')

    verbosity_error_msg = 'Verbosity should be a positive integer'
    if not isinstance(verbosity, int):
        sys.exit(verbosity_error_msg)
    elif verbosity < 0:
        sys.exit(verbosity_error_msg)

    if visualizer.lower() not in ['ase', 'vesta']:
        sys.exit('Currently supported visualizers include : \n- ase, \n- vesta ')

    #TODO : add other user inputs here.

    if verbosity >= 1:
        print('Running script {}'.format(os.path.basename(__file__)))
    main(RELATIVE_BOND_LENGTH_TOLERANCE=rel_bond_length_tol,
         MAX_ITERATIONS=max_iterations, VISUALIZER=visualizer.lower() )


if __name__ == '__main__':
    cli()




