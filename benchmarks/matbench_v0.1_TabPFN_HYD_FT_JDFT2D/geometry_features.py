"""Original target-free geometric descriptors, unchanged."""
import numpy as np

def geometry_features(s):
    """Target-free cell, packing, plane gaps and periodic-neighbor descriptors."""
    n = len(s)
    lengths = np.sort(s.lattice.abc)
    angles = np.sort(s.lattice.angles)
    heights = 1 / np.linalg.norm(s.lattice.reciprocal_lattice_crystallographic.matrix, axis=1)
    gaps = []
    for axis in range(3):
        z = np.sort(s.frac_coords[:, axis] % 1)
        gaps.append(float(np.diff(np.r_[z, z[0]+1]).max()*heights[axis]))
    center, _, _, dist = s.get_neighbor_list(6.0, exclude_self=True)
    good = dist > 1e-6
    center, dist = center[good], np.round(dist[good], 10)
    bins = np.linspace(0,6,61)
    histogram = np.histogram(dist,bins=bins)[0] / n
    stats = []
    for cutoff in [2.,2.5,3.,3.5,4.,5.,6.]:
        degree = np.bincount(center[dist <= cutoff], minlength=n)
        stats.extend([degree.mean(),degree.std(),degree.min(),degree.max()])
    quantiles = np.quantile(dist,[0,.1,.25,.5,.75,.9,1]) if len(dist) else np.full(7,np.nan)
    return np.r_[np.log1p(n),s.volume/n,float(s.density),lengths,angles,
                 lengths[1:]/lengths[0],np.sort(gaps),heights.max()/heights.min(),
                 histogram,stats,quantiles]
