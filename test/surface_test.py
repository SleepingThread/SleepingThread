import numpy as np
from SleepingThread.surface import Surface
surf = Surface.readSurface("1.points","1.meshidx")
mask = surf.getSubset(surf.points[0])

points,mesh_index = surf.getSubSurface(mask)

print Surface.calculateNormal(points,mesh_index,Surface.calculateSurfaceCenter(points))

print np.sum(mask)
