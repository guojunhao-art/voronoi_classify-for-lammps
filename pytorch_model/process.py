# Import OVITO modules.
from ovito.io import *
from ovito.modifiers import *

# Import NumPy module.
import numpy

# Load a simulation snapshot
pipeline = import_file("dump.1.cho")
print(pipeline.source)
pipeline.modifiers.append(VoronoiAnalysisModifier())
with open('voro.txt','w') as f:
    for i in range(0,10001):
        data = pipeline.compute(i)
        Coordination = data.particles['Coordination']
        Atomic_Volume = data.particles['Atomic Volume']
        for j in range(len(Coordination)):
            aa=data.particles.identifiers[j].astype(str)
            x=data.particles.positions[j][0].astype(str)
            y=data.particles.positions[j][1].astype(str)
            z=data.particles.positions[j][2].astype(str)
            a=Coordination[j].astype(str)
            b=Atomic_Volume[j].astype(str)
            if data.particles.particle_types[j]<=3:
                tp='1'
            else:
                tp='2'
            f.write(aa+' '+x+' '+y+' '+z+' '+a+' '+b+' '+tp+'\n')
print('down')
