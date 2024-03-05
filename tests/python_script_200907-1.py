from pymatgen import Lattice, Structure, Molecule

InputFileName = "D:\cadarp02\Documents\CristalStructures\TeO2-alpha_P41212_1968_COD-1537586.cif"
outputFileName = "POSCAR"

# Read a CIF file
structure = Structure.from_file(InputFileName)
structure.to("poscar",outputFileName)
f = open(outputFileName, 'r')
file_contents = f.read()
print (file_contents)
f.close()

sites = structure.sites
print(sites)

for site in sites:
#   print('variable site is of type : ',type(site))
    if site.is_ordered:
        print (site.specie,' ',site.a,' ',site.b,' ',site.c)
    else:
        print (site.species,' ',site.a,' ',site.b,' ',site.c,' (mixed-occupancy site)')
    print (site.coords)

print("**** Using get_all_neighbors structure method *******")

cutoffDistance = 2.5 ;
neighborsList = structure.get_all_neighbors(cutoffDistance)
print(neighborsList)

"""
print('Distance between sites ',sites[1].specie,' and  ',sites[5].specie,' : ',sites[1].distance(sites[5],None))
print(sites[1].lattice)

print("***** Using CrystalNN class ******")
from pymatgen.analysis.local_env import CrystalNN
myCrystalNN = CrystalNN()
siteNb = 1 ;
print('Coordination number of site ',siteNb,' : ',myCrystalNN.get_cn(structure,siteNb))
myNNData = myCrystalNN.get_nn_data(structure,siteNb)
# print(myNNData)
print(len(myNNData.all_nninfo))
print(myNNData.all_nninfo)
nnNb = 1
print(type(myNNData.all_nninfo[nnNb]))
print(type(myNNData.all_nninfo[nnNb]["site"]))
nnSite = myNNData.all_nninfo[nnNb]["site"] # creates a (Periodic)Site object
print('Distance to site ',siteNb,' : ',nnSite.distance(sites[siteNb],None), ' A')
print(myNNData.all_nninfo[nnNb]["image"])
print(myNNData.all_nninfo[nnNb]["weight"])
print(type(myNNData))

print("*************************")
for i in range(0,2):
    print(i)

"""
