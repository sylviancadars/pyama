from pyama.cpmdTools.utils import CPMDData
from structureComparisonsPkg.distanceTools import distanceMatrixData
import matplotlib.pyplot as plt
from pyama.utils import visualize_structure

dir_name='/data/CPMD/As2Te3-beta/MD_SYM-4_EFREQ-1600'
cpmdd = CPMDData(dir_name, verbosity=3,
                 system_description='\u03B1-As2Te3, 1x4x2 supercell, CPMD dynamics at 300 K')
cpmdd.set_simul_time(initial_simul_time=23500)
cpmdd.parse_output_file('output.out0')
print('Unit cell parameters (in \u212b):\n',
      cpmdd.initial_structure.lattice.lengths,
      cpmdd.initial_structure.lattice.angles)


dmd = distanceMatrixData()
partials, types = dmd.calculate_all_partial_RDFs(cpmdd.initial_structure)

fig, ax = plt.subplots()
y_shift = 0.0
legend = []
for type1 in types:
    for type2 in types:
        ax.plot(dmd.R, partials[types.index(type1), types.index(type2), :] + y_shift)
        y_shift += 10.0
        legend.append('partial {}-{} RDF'.format(type1, type2))

ax.legend(legend)
ax.set(xlabel='r (\u212B)', ylabel='Intensity (AU)',
       title='Initial structure partials')

visualize_structure(cpmdd.initial_structure)
visualize_structure(cpmdd.initial_structure, viewer='vesta')

ionic_iterations = cpmdd.pick_evenly_spaced_frames(max_nb_of_frames=20)
traj_indexes = cpmdd.get_traj_indexes_from_ionic_iterations(ionic_iterations)
partials, types = cpmdd.get_averaged_partials(traj_indexes)
pdf, r = cpmdd.get_exact_xrd_pdf_from_partials(partials, types)

fig2, ax2 = plt.subplots()
ax2.plot(r, pdf)
legend = ['Total pdf obtained from {} evenly-spaced frames'.format(len(
    ionic_iterations))]
y_shift = 0.5*max(pdf)
for type1 in types:
    for type2 in types:
        ax2.plot(cpmdd.r, partials[types.index(type1), types.index(type2), :] + y_shift)
        y_shift += 10.0
        legend.append('partial {}-{} RDF'.format(type1, type2))


