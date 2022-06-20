from bdb import effective
from sre_parse import expand_template
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def PolyArea(coordinates):
    x = coordinates[:,0]
    y = coordinates[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def FrameDetect(vor, extent):
    """
    detect frame-touching cells and return the filtered results containing
    non-touching cell regions
   
    Parameters:
    -----------
    vor : Voronoi object from scipy.spatial.Voronoi
    
    Return:
    ------------
    new regions of non-touching cells 
    new vertices as a dictionary with index:coordinate
    """
    if extent is None:
        raise
    regions = vor.regions
    vertices = vor.vertices
    veridx = vor.ridge_vertices
    #filter all in-range vertices
    new_vertices = {}
    new_veridx = []
    for i, v in enumerate(vertices):
        if np.all(v <= max(extent)) and np.all(v >= min(extent)):
            new_vertices.setdefault(i,v)
            new_veridx.append(i)
    new_veridx = np.array(new_veridx)

    #filter non-touching regions
    new_regionos = {}
    for j, k in enumerate(regions[1:]):
        if np.all(np.isin(k,new_veridx)):
            new_regionos.setdefault(j, k)
    return new_regionos, new_vertices

def Convexity(regions, vertices):
    """
    Calculate convexity of regions based on its conprising vertices using 
    scipy.spatial.Convexhull and self-defined PolyArea

    Parameters:
    ----------- 
    regions: indices of comprising vertices in cells in a dict{ridx:[1,2,3,...],...}
    vertices: indices of vertices and their coordinates in a dict{idx:[coordinate],...}

    Return:
    -----------
    a list of convexity of input regions
    """
    from scipy.spatial import ConvexHull
    convexity = []
    for ridx, value in regions.items():
        coordinates = [vertices[i] for i in value]
        coord = np.array(coordinates)
        S1 = PolyArea(coord) # The real area of cell
        cvh = ConvexHull(coord)
        S2 = PolyArea(cvh.points[cvh.vertices])
        c = S1/S2
        convexity.append(c)
    return convexity


def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.

    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - 4*a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


def Slenderness(regions, vertices, params=None,):
    """
    
    """
    if params is not None:
        return params[2]/params[3]
    else:
        result = {}
        for cellidx, ridx in regions.items():
            cc= [vertices[i] for i in ridx]
            result.setdefault(cellidx, cc)
        out_ = []
        for i, value in result.items():
            r_ = np.concatenate(value, axis=0).reshape(len(value)/2, 2)
            contain1 = fit_ellipse(r_[:,0], r_[:,1])
            para = cart_to_pol(contain1)
            out_.append(para[2]/para[3])
        return out_

def ellipse_map(clusters, extent=None, points_on=None,
                slenderness_on=None, **kwargs):
    """
    draw a map of clusters using fitted ellipses

    Parameters:
    ------------
    clusters: input clusters.
    extent: limits for drawing proper size of map and facilitate 
            its latter imposing on image.
    """
    from matplotlib.patches import Ellipse
    number = len(clusters.clusters)
    centroid = clusters.centroids
    ellipse_para = []
    for i, coord in clusters.clusters_coordinates.items():
        x0, y0, ap, bp, e, phi = cart_to_pol(fit_ellipse(coord[:,0], 
                                 coord[:,1]))
        if slenderness_on == True: color = ap/bp
        else:color = len(coord)
        xy = np.array([x0, y0])
        width = ap * 2
        height = bp * 2 
        angle = np.degrees(phi)
        para = {'xy':xy, 'width':width, 'height':height,
               'angle': angle, 'c': color}
        ellipse_para.append(para)
    ellipses = [Ellipse(j['xy'], j['width'] , j['height'] , j['angle'],) 
               for j in ellipse_para]
    fig, ax = plt.subplots(dpi=300, **kwargs)
    ax.plot(centroid[:,0], centroid[:,1], '*', c='r')
    for k in ellipses:
        ax.add_patch(k)
    ax.set_axis_off()
    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])
    ax.set_aspect(1)


def slenderness_calc(cluster_coordinates, ellipse_paras=False):
    from scipy.spatial import ConvexHull
    coords = np.array(cluster_coordinates)
    if coords.shape[1] == 2:
        hull = ConvexHull(coords)
    else:print('dimension wrong')
    vertices = coords[hull.vertices]
    x0, y0, ap, bp, e, phi = cart_to_pol(fit_ellipse(vertices[:, 0], 
                             vertices[:, 1]))
    slenderness = ap / bp
    if ellipse_paras == False: return slenderness
    else:
        return x0, y0, ap, bp, e, phi, slenderness


def slender_voronoi_plot(image, clusters, regions, vertices,
                         centroids, cmap='gray', peaks=None, ax=None, alpha=.8):
    import matplotlib
    import matplotlib.cm as cm
    from skimage.measure import EllipseModel

    points_augment = clusters.slenderness_points
    slenderness = []
    effectivenum = 0
    for region in regions:
        ellipse = EllipseModel()
        polygon = vertices[region]
        center_v = np.mean(polygon, axis=0).reshape((1, 2))
        keys = np.array([i for i in points_augment.keys()])
        center = np.tile(center_v, (len(keys), 1))
        pos = np.argmin(np.sum((keys-center)**2, axis=1))
        key = tuple(np.round(keys[pos], decimals=3))
        t1 = points_augment[key]
        poly_augmented = np.concatenate((polygon, points_augment[key]), axis=0)
        xx = [x for x in poly_augmented.ravel()]
        if np.all(np.array(xx) > 0) and np.all(np.array(xx) < 1024):
            success = ellipse.estimate(poly_augmented)
            coeffs = ellipse.params
            a = coeffs[2] / coeffs[3]
            b = coeffs[3] / coeffs[2]
            if np.all(coeffs):
                if a > b:
                    slender = a
                    effectivenum += 1
                else:
                    slender = b
                    effectivenum += 1
            else:
                slender = 0
        else:
            slender = 0
        slenderness.append(slender)

    norm = matplotlib.colors.Normalize(
        vmin=min(slenderness), vmax=max(slenderness), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    if peaks is not None:
        centroids = peaks

    if ax == None:
        fig, ax = plt.subplots()
        for region, color in zip(regions, slenderness):
            polygon = vertices[region]
            ax.fill(*zip(*polygon), c=mapper.to_rgba(color),
                    alpha=alpha)
        ax.plot(centroids[:, 0], centroids[:, 1], 'o', ms=.5, mec='m', mfc='g')
        ax.imshow(image.T, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        ax.set_aspect(1)
        fig.colorbar(mapper, ax=ax)

    else:
        for region, color in zip(regions, slenderness):
            polygon = vertices[region]
            ax.fill(*zip(*polygon), c=mapper.to_rgba(color),
                    alpha=alpha)
        ax.plot(centroids[:, 0], centroids[:, 1], 'o', ms=.5, mec='m', mfc='g')
        ax.imshow(image.T, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 1024)
        ax.set_aspect(1)
        plt.colorbar(mapper, ax=ax)


def portion_slenderness(regions, vertices, slenderness, threshold=1.5):
    total_area = 0.
    for region in regions:
        polygon = vertices[region]
        total_area += PolyArea(polygon)

    effective_area = 0.
    for region, s_val in zip(regions, slenderness):
        polygon = vertices[region]
        if s_val > threshold:
            effective_area += PolyArea(polygon)
    
    return effective_area / total_area
        

def modified_slenderness_producer(regions, vertices, clusters):
    from skimage.measure import EllipseModel
    points_augment = clusters.slenderness_points
    slenderness = []
    effectivenum = 0
    for region in regions:
        ellipse = EllipseModel()
        polygon = vertices[region]
        center_v = np.mean(polygon, axis=0).reshape((1, 2))
        keys = np.array([i for i in points_augment.keys()])
        center = np.tile(center_v, (len(keys), 1))
        pos = np.argmin(np.sum((keys-center)**2, axis=1))
        key = tuple(np.round(keys[pos], decimals=3))
        poly_augmented = np.concatenate((polygon, points_augment[key]), axis=0)
        xx = [x for x in poly_augmented.ravel()]
        if np.all(np.array(xx) > 0) and np.all(np.array(xx) < 1024):
            success = ellipse.estimate(poly_augmented)
            coeffs = ellipse.params
            a = coeffs[2] / coeffs[3]
            b = coeffs[3] / coeffs[2]
            if np.all(coeffs):
                if a > b:
                    slender = a
                    effectivenum += 1
                else:
                    slender = b
                    effectivenum += 1
            else:
                slender = 0
        else:
            slender = 0
        slenderness.append(slender)
    return slenderness
    





    



        
        

    
    








