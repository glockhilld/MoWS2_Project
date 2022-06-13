import numpy as np
import math as ma
from numba import njit


def dist(a, b):
    return ma.dist(a,b)

class RDF_calculator:
    '''
    calculate input points with 2D coordinates inside, and plot radial distribution function
        
    Parameters:
    -----------
    points: 2D coordinates input
    cellParas: the 2D size of pixels
    pixelSize: determine real physical size
    resolution: the steps of g(r)

    Attribute:
    -----------
    points: input coordinates of obects like atoms
    cellParas: the boundary size of calculated frame
    pixelSize: the input image pixel size for facilitating real measurements
    resolution: the resolution of rdf for the density of calculation and plotting
    g: the resultant rdf
    '''
    def __init__(self, points, cellParas=(1024.0, 1024.0), pixelSize=(1,1), resolution=1024.0) -> None:
        self.points = points
        self.cellParas = cellParas
        self.pixelSize = pixelSize
        self.resolution = resolution

        self.realpoints = self.points * self.pixelSize
        self.realsize = np.array(self.cellParas) * self.pixelSize
        self.dr = self.realsize[0] / 2 / self.resolution
        r = np.ones(self.resolution) * self.dr * np.arange(1,self.resolution+1)
        y= np.zeros((self.resolution, 2))
        y[:, 0] = r
        self.g = y
    

    def cal_rdf(self):
        for i, coords in enumerate(self.realpoints):
            for j, coords2 in enumerate(self.realpoints[i+1:,:]):
                distance = dist(coords, coords2)
                index = int(distance / self.dr)
                if 0 < index < self.resolution:
                    self.g[index,1] += 2.0

        volume = np.zeros(self.resolution)
        pi = np.pi
        for k in range(self.resolution):
            r1 = k * self.dr
            r2 = r1 + self.dr
            s1 = 2 * pi * r1 **2
            s2 = 2 * pi * r2 **2
            volume[k] += s2 - s1
        self.g[:,1] /= volume
        return self.g
    
    def plot(self, ax=None, style=None, cutoff=None, filename="", **kwgs):
        import matplotlib.pyplot as plt
        if ax is None:
           fig, ax = plt.subplots()
        
        if style is None:
            with plt.style.use(['ieee',]):
                ax.plot(self.g[:,0], self.g[:,1], **kwgs)
                ax.set_xlabel('r (Å)')   
                ax.set_ylabel('g_r')
        else:
            with plt.style.use(list(style)):
                ax.plot(self.g[:,0], self.g[:,1], **kwgs)
                ax.set_xlabel('r (Å)')   
                ax.set_ylabel('g_r')
        
        if not self.g.any():
            print('compute the radial distribution function first\n')
            
        
        if cutoff:
            ax.set_xlim([0, cutoff])

        if filename:
            plt.savefig(filename, dpi=300, bbox='tight')
        return ax


        