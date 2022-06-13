from sre_constants import SUCCESS
import numpy as np
from RDF import dist
#from numba import njit
from copy import deepcopy

class Atom:
    """
    define every single atoms objects with unique lable for later clustering

    Parameters:
    ------------
    points: means input coordinates for single atoms
    -----------
    """
    def __init__(self, coordinate, order, lable=None):
        self.coordinate = coordinate
        self.order = order
        self.lable = lable
        if self.coordinate is None:
            print('Empty coordinates')
        if self.order is None:
            print('Empty order, please input order.')
   
    def distance(self, b):
        return dist(self.coordinate, b.coordinate)


class Atoms:
    def __init__(self, coordinates, orders=None, lables=None) -> None:
        if orders is not None:
            self.atoms = [Atom(coordinate, order) for order, coordinate in zip(orders, coordinates)]
        elif lables is not None and orders is not None:
            self.atoms = [Atom(coordinate, order, lable) for order, coordinate, lable in zip(orders, coordinates, lables)]
        else:
            self.atoms = [Atom(coordinate, order) for order, coordinate in enumerate(coordinates)]
        self.number = len(self.atoms)

    def getAtoms(self):
        return self.atoms

    def sortout(self, series, return_coordinates=None):
        mask = np.zeros(self.number+1, dtype=int)
        for i in range(1, self.number+1):
            if i in series:
                mask[i] += 1
        #mask = mask.astype(np.int32)
        if return_coordinates is True:
            select_atoms = self.atoms[mask]
            return [i.coordinate for i in select_atoms]
        else:
            return list(np.array(self.atoms)[mask])


class clustering:
    """
    clusters generator and return a bunch of attributes about it.
    
    Parameters:
    -----------
    Atoms: generated from Class Atoms


    Return:
    -----------
    atoms: input atoms objects
    atomsNumber: the number of atoms input
    clusters: the clusters results after partitioning
    clustersNumber: the number of clusters
    clusters_coordinaters: the dict containning coordinates
                           in form of {clusterNumber:coordinates,...}
    centroids: the collection of the centroid of every individual clusters
    clusters_sizes: the collection of sizes of all individual clusters, a dict in 
                    form of {clusterNumber:size,...}
    """

    def __init__(self, Atoms) -> None:
        self.atoms = Atoms.atoms # input atoms to adress
        self.atomsNumber = Atoms.number

    def partition(self, cutoff):
        self.clusters = []
        _pool = deepcopy(self.atoms) # deepcopy atoms into the pool
        self.clusters.append([_pool[0]]) #append ever first atom in the pool to the empty list clusters
        del _pool[0]
        i = 0 # cluster number
        while _pool:
            j = 0 # atom number in cluster[i]
            while j < len(self.clusters[i]):
                k = 0 # atom number in the pool 
                while k < len(_pool):
                    if self.clusters[i][j].distance(_pool[k]) < cutoff:
                        self.clusters[i].append(_pool[k])
                        del _pool[k]
                        k += 1
                    else:k += 1
                if len(_pool) != 0: j += 1
                else:break
            if len(_pool) != 0:
                self.clusters.append([_pool[0]])
                del _pool[0]
                i += 1
                if len(_pool) == 0: break
            else: break
        self.clustersNumber = len(self.clusters)
        centroids = []
        self.clusters_coordinates = {}  # {clusterNumber:coordinates,...}
        for i, cluster in enumerate(self.clusters):
            coords = [k.coordinate for k in cluster]
            coor_arr = np.array(coords).reshape([len(cluster), 2])
            self.clusters_coordinates.setdefault(i, coor_arr)
            centroid = np.mean(coor_arr, axis=0)
            centroids.append(list(centroid))
        self.centroids = np.array(centroids)
        self.clusters_sizes = {}  # {clusterNumber:size,...}
        for j, cluster in self.clusters_coordinates.items():
            self.clusters_sizes.setdefault(j, len(cluster))

    def slenderness_calc(self, ellipse_paras=False):
        from scipy.spatial import ConvexHull
        from skimage.measure import EllipseModel
        #from voronoi import cart_to_pol, fit_ellipse
        coords = np.array(list(self.clusters_coordinates.values()), dtype=object)
        self.slenderness = {}
        self.slenderness_points = {}
        for coord in coords:
            if coord.shape[1] == 2:
                if len(coord) >=3:
                    '''hull = ConvexHull(coord)
                    vertices = coord[hull.vertices]
                    x0, y0, ap, bp, e, phi = cart_to_pol(
                        fit_ellipse(vertices[:, 0], vertices[:, 1]))
                    slenderness = ap / bp'''
                    #hull = ConvexHull(coord)
                    #vertices = coord[hull.vertices]
                    ellipse = EllipseModel()
                    success = ellipse.estimate(coord)
                    if success: 
                        coeffs = ellipse.params
                        slenderness = coeffs[2] / coeffs[3]
                    else: 
                        slenderness = 0
                else:
                    slenderness = 0
                centriod = np.round(np.mean(coord, axis=0), decimals=3)
            else:
                print('dimension wrong')
            
            if ellipse_paras == False:
                self.slenderness.setdefault(tuple(centriod), slenderness)
            else:
                pass
            self.slenderness_points.setdefault(tuple(centriod), coord)

        















class clusterabandon:
    '''
    Judge if all input data is belong to a cluster defined by first atom in Atoms object, 
    according to interspace constrained by cutoff,
    and will provide those not belonging to cluster as aliens.
    -------------
    Parameters:
    -------------
    Atoms: input data/atoms
    cutoff: the distance constraint as a judgement
    -------------
    Return nothing, but a bunch of attributes to call
    '''
    def __init__(self, Atoms, cutoff) -> None:
        self.batch = Atoms
        self.cutoff = cutoff
        i = 0
        aliens = []
        while i < len(self.batch):
            j = i + 1
            while j < len(self.batch):
                if self.batch[i].distance(self.batch[j]) < cutoff:
                    aliens.append(self.batch[j])
                    del self.batch[j]
                    j += 1
                else:
                    j += 1
            i += 1
        self.atoms = self.batch
        self.aliens = aliens
        coordinates = [self.batch[k].coordinate for k in range(len(self.batch))]
        self.centroid = np.mean(coordinates, axis=0)
        self.atomNumber = len(self.atoms)



class clusteringabandon:
    '''clustering all atoms providered based on distance
    
    Parameters:
    ------------
    atoms: input atom objects
    cutoff: distance cutoff of if atoms belong to same cluster


    '''

    def __init__(self, atoms, cutoff) -> None:
        self.atoms = atoms.atoms
        self.obj = atoms
        self.cutoff = cutoff
        self.pool = []
        for i, atom in enumerate(self.atoms):  # finish pair distance calculation
            for j in self.atoms[i+1:]:
                distance = dist(atom.coordinate, j.coordinate)
                lable_set = set()
                lable_set.update([atom.order, j.order])
                if distance <= self.cutoff:
                    self.pool.append(lable_set)
    
        
    def partition(self, return_atoms=None):  # do not return anything but store the final clusters or atoms in clusters into self.clusters
        _pool = deepcopy(self.pool)
        flag = 0
        while flag < len(_pool):
            j = flag + 1
            while j < (len(_pool)-1):
                if _pool[flag] & _pool[j] is not None:
                    _pool[flag].update(_pool[j])
                    del _pool[j]
                    j += 1
                    #print(len(_pool))
                else:
                    pass
            flag += 1
         
        self.clusterNumber = len(_pool)
        if return_atoms is None: # if return_atoms is True then give the atoms lables in clusters, 
            self.clusters = _pool
            print('Please check attribute clusters')
        else:
            series = [np.array(i) for i in _pool]
            self.clusters = [self.obj.sortout(j) for j in series]
            print('Please check attribute centroids')
            centroids = np.zeros([self.clusterNumber, 2])
            for k, cluster in enumerate(self.clusters):
                length = len(cluster)
                coords = np.zeros([length, 2])
                for j, a in enumerate(cluster):
                    coords[j,:] = a.coordinate 
                centroids[k,:] = np.mean(coords, axis=0)
            self.centroids = centroids




    



