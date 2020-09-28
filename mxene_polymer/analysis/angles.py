import numpy as np
import MDAnalysis
import matplotlib.pyplot as plt

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    theta = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return theta 


def compute_angles(top_file, trj_file, tam_length):
    """ Compute angles between specific groups and normal of the MXene surface
        WARNING: Do not use Gromacs XTC file as lower precision will result in noise around 90 degrees

    top_file : str
        A topology file that contains atomtype information
    trj_file : str
        A trajectory file that is valid with MDAnalysis
    tam_length : int
        tetraalkylammonium length in box
    """
    universe = MDAnalysis.Universe(top_file, trj_file)

    #ring_carbon = universe.select_atoms('type kpl_012 kpl_011')
    
    box = universe.select_atoms('all').bbox()
    _compute_ring_angles(universe, tam_length)


def _compute_ring_angles(universe, tam_length):
    ring_groups = [universe.select_atoms('type kpl_012 kpl_011 and resid {}'.format(r.resid)) for r in universe.residues if r.resname == 'emim']

    ring_angles = np.nan * np.empty(shape=(universe.trajectory.n_frames, len(ring_groups)))
    """
    >>> np.where(np.isnan([4, 4, ]))[0].shape[0]
    0
    >>> np.where(np.isnan(x))[0].shape[0]
    4
    """
    for n_frame, frame in enumerate(universe.trajectory[::1]):
        for n_ring, ring_group in enumerate(ring_groups):
            # Consider first layer of ions
            #if ring_group.center_of_mass()[2] > 6.82 + 5.0:
            if tam_length == 12:
                z_length = 28.37
            elif tam_length == 16:
                z_length = 32.62
            pore1 = ring_group.center_of_mass()[2] > z_length or ring_group.center_of_mass()[2] < z_length + 5.0
            pore2 = ring_group.center_of_mass()[2] > 9.37 or ring_group.center_of_mass()[2] < 9.37 + 5.0
            if not pore1 or not pore2:
                continue
            # Only consider ions in the pore
            if ring_group.center_of_mass()[1] < 5 or ring_group.center_of_mass()[1] > 45:
                continue
            print(n_ring)
            xyz = ring_group.positions 
            #AB = xyz[0, :] - ring_group.center_of_mass()
            #AC = xyz[1, :] - ring_group.center_of_mass()
            AB = xyz[0, :] - xyz[2, :]
            AC = xyz[1, :] - xyz[2, :]
            plane_vector = np.cross(AB, AC) 
            ring_angles[n_frame, n_ring] = angle_between([plane_vector[0], plane_vector[1], plane_vector[2]], [0, 0, 1]) * (180 / np.pi)

    ring_angles = ring_angles.reshape((-1))
    ring_angles = ring_angles[np.logical_not(np.isnan(ring_angles))]
    histo = np.histogram(ring_angles, bins=181, density=True)
    new_bins = get_center(histo[1])
    np.savetxt('histo_ring_angles.txt', np.transpose(np.vstack([new_bins, histo[0]])),
        header='Count\tAngle')
    np.savetxt('ring_angles.txt', np.asarray(ring_angles))
    fig, ax = plt.subplots()
    ax.hist(ring_angles, bins=181, density=True)
    plt.xlim((0, 180))
    fig.savefig('ring_angles.pdf')


def _compute_tail_angles(universe, tam_length):
    tail_groups = [universe.select_atoms('type kpl_014 kpl_016 and resid {} and not bonded type lopes_016'.format(r.resid)) for r in universe.residues if r.resname == 'emim']

    tail_angles = np.nan * np.empty(shape=(universe.trajectory.n_frames, len(tail_groups)))
    """
    >>> np.where(np.isnan([4, 4, ]))[0].shape[0]
    0
    >>> np.where(np.isnan(x))[0].shape[0]
    4
    """

    for n_frame, frame in enumerate(universe.trajectory[::1]):
        for n_tail, tail_group in enumerate(tail_groups):
            # Consider first layer of ions
            #if ring_group.center_of_mass()[2] > 6.82 + 5.0:
            if tam_length == 12:
                z_length = 28.37
            elif tam_length == 16:
                z_length = 32.62
            elif tam_length == 'channel':
                z_length = 6.82
            pore1 = tail_group.center_of_mass()[2] > z_length and tail_group.center_of_mass()[2] < (z_length+7.0)
            if not pore1:
                continue
            #print(n_tail)
            xyz = tail_group.positions 
            plane_vector = xyz[1] - xyz[0]
            tail_angles[n_frame, n_tail] = angle_between([plane_vector[0], plane_vector[1], plane_vector[2]], [0, 0, 1]) * (180 / np.pi)

    tail_angles = tail_angles.reshape((-1))
    tail_angles = tail_angles[np.logical_not(np.isnan(tail_angles))]
    histo = np.histogram(tail_angles, bins=181, density=True)
    new_bins = get_center(histo[1])
    np.savetxt('histo_tail_angles.txt', np.transpose(np.vstack([histo[1][1:], histo[0]])),
        header='Count\tAngle')
    np.savetxt('tail_angles.txt', np.asarray(tail_angles))
    fig, ax = plt.subplots()
    ax.hist(tail_angles, density=True, bins=181)
    plt.xlim((0, 180))
    plt.ylim((0, 0.06))
    fig.savefig('tail_angles.pdf')

def get_center(bins):
    new_bins = list()
    for idx, bi in enumerate(bins):
        if (idx+1) >= len(bins):
            continue
        mid = (bins[idx] + bins[idx+1])/2
        new_bins.append(mid)

    return new_bins
