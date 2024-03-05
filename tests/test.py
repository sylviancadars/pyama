import pymatgen
from pymatgen.core.structure import Structure
from importlib.metadata import version
import inspect

print(version('pymatgen'))
print(inspect.getsourcefile(Structure))


