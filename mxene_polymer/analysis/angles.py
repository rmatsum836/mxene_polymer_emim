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


def compute_angles(top_file, trj_file, tam_length, angle_type='ring', cutoff=7.0):
    """ Compute angles between specific groups and normal of the MXene surface
        WARNING: Do not use Gromacs XTC file as lower precision will result in noise around 90 degrees

    top_file : str
        A topology file that contains atomtype information
    trj_file : str
        A trajectory file that is valid with MDAnalysis
    tam_length : int
        tetraalkylammonium length in box
    angle_type : str, default='ring'
        angle to calculate between normal of wall
    cutoff : float, units=angstroms, default=7.0
        distance from wall to consider
    """
    universe = MDAnalysis.Universe(top_file, trj_file)
    box = universe.select_atoms('all').bbox()
    if cutoff == None:
        if tam_length == 12:
            cutoff = 11.13
        elif tam_length == 16:
            cutoff = 14.0
    if angle_type == 'ring':
        _compute_ring_angles(universe, tam_length, cutoff)
    if angle_type == 'tail':
        _compute_tail_angles(universe, tam_length, cutoff)
    if angle_type == 'taa':
        _compute_taa_angles(universe, tam_length, cutoff)


def _compute_ring_angles(universe, tam_length, cutoff):
    ring_groups = [universe.select_atoms('type kpl_012 kpl_011 and resid {}'.format(r.resid)) for r in universe.residues if r.resname == 'emim']

    #ring_angles = np.nan * np.empty(shape=(universe.trajectory.n_frames-1000, len(ring_groups)))
    ring_angles = np.nan * np.empty(shape=(4000, len(ring_groups)))
    """
    >>> np.where(np.isnan([4, 4, ]))[0].shape[0]
    0
    >>> np.where(np.isnan(x))[0].shape[0]
    4
    """
    for n_frame, frame in enumerate(universe.trajectory[:4000:1]):
        for n_ring, ring_group in enumerate(ring_groups):
            # For some reason the zero frame exists twice
            if n_frame == 0:
                continue
            # Consider first layer of ions
            if tam_length == 12:
                z_length = 29.47
            elif tam_length == 16:
                z_length = 32.62
            pore1 = ring_group.center_of_mass()[2] > z_length or ring_group.center_of_mass()[2] < z_length + cutoff
            pore2 = ring_group.center_of_mass()[2] > 9.37 or ring_group.center_of_mass()[2] < 9.37 + cutoff
            if not pore1 or not pore2:
                continue
            # Only consider ions in the pore, max is hardcoded to edge of MXene sheet
            if ring_group.center_of_mass()[1] < 10 or ring_group.center_of_mass()[1] > (54.67-10):
                continue
            # Check which side of pore ion is on to determine normal vector of surface
            if pore1:
                if ring_group.positions[0][2] < (z_length + (z_length+cutoff)) / 2:
                    side_of_pore = 1
                else:
                    side_of_pore = -1
            if pore2:
                if ring_group.positions[0][2] < (9.37 + (9.37+cutoff)) / 2:
                    side_of_pore = 1
                else:
                    side_of_pore = -1
            xyz = ring_group.positions 
            AB = xyz[1, :] - xyz[0, :]
            AC = xyz[2, :] - xyz[0, :]
            plane_vector = np.cross(AB, AC) 
            #ring_angles[n_frame, n_ring] = angle_between([plane_vector[0], plane_vector[1], plane_vector[2]], [0, 0, 1]) * (180 / np.pi)
            ring_angles[n_frame, n_ring] = angle_between([plane_vector[0], plane_vector[1], plane_vector[2]], [0, 0, side_of_pore]) * (180 / np.pi)

    ring_angles = ring_angles.reshape((-1))
    ring_angles = ring_angles[np.logical_not(np.isnan(ring_angles))]
    histo = np.histogram(ring_angles, bins=90, range=(0.0, 180.0), density=True)
    new_bins = get_center(histo[1])
    np.savetxt(f'histo_ring_angles_{cutoff}.txt', np.transpose(np.vstack([new_bins, histo[0]])),
        header='Count\tAngle')
    np.savetxt(f'ring_angles_{cutoff}.txt', np.asarray(ring_angles))
    fig, ax = plt.subplots()
    counts, angle_bins, bars = ax.hist(ring_angles, bins=90, range=(0.0, 180.0), density=True)
    angle_bins_center = (angle_bins[:-1] + angle_bins[1:])/2
    plt.xlim((0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    fig.savefig(f'unnormalized_ring_angles_{cutoff}.pdf')
   
    normalized_counts = np.divide(counts,
        abs(np.sin((np.pi / 180)*angle_bins_center)))
    fig, ax = plt.subplots()
    arr = ax.hist(angle_bins_center, weights=normalized_counts, bins=90, range=(0.0, 180.0),
        density=False)
    plt.xlim((0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    fig.savefig(f'normalized_ring_angles_{cutoff}.pdf')
    np.savetxt(f'normalized_ring_angles_{cutoff}.txt', np.transpose(np.vstack([angle_bins_center, normalized_counts])), header="bins\tnormalized counts")

def _compute_tail_angles(universe, tam_length, cutoff):
    #tail_groups = [universe.select_atoms('type kpl_014 kpl_016 and resid {} and not bonded type kpl_016'.format(r.resid)) for r in universe.residues if r.resname == 'emim']
    tail_groups = [universe.select_atoms('(type kpl_010 and resid {} and around 3 type kpl_016) or (type kpl_016 and resid {})'.format(r.resid, r.resid)) for r in universe.residues if r.resname == 'emim']

    #tail_angles = np.nan * np.empty(shape=(universe.trajectory.n_frames-1000, len(tail_groups)))
    tail_angles = np.nan * np.empty(shape=(4000, len(tail_groups)))
    """
    >>> np.where(np.isnan([4, 4, ]))[0].shape[0]
    0
    >>> np.where(np.isnan(x))[0].shape[0]
    4
    """

    for n_frame, frame in enumerate(universe.trajectory[:4000:1]):
        for n_tail, tail_group in enumerate(tail_groups):
            # For some reason the zero frame exists twice
            if n_frame == 0:
                continue
            # Consider first layer of ions
            #if ring_group.center_of_mass()[2] > 6.82 + 5.0:
            if tam_length == 12:
                z_length = 29.47
            elif tam_length == 16:
                z_length = 32.62
            pore1 = tail_group.center_of_mass()[2] > z_length and tail_group.center_of_mass()[2] < (z_length+cutoff)
            pore2 = tail_group.center_of_mass()[2] > 9.37 or tail_group.center_of_mass()[2] < 9.37 + cutoff
            if not pore1 or not pore2:
                continue
            # Only consider ions in the pore, max is hardcoded to edge of MXene sheet
            if tail_group.center_of_mass()[1] < 10 or tail_group.center_of_mass()[1] > (54.67-10):
                continue
            # Check which side of pore ion is on to determine normal vector of surface
            if pore1:
                #if tail_group.positions[0][2] < (z_length + (z_length+cutoff)) / 2:
                if tail_group.center_of_mass()[2] < (z_length + (z_length+cutoff)) / 2:
                    side_of_pore = 1
                elif tail_group.center_of_mass()[2] > (z_length + (z_length+cutoff)) / 2:
                    side_of_pore = -1
            if pore2:
                if tail_group.center_of_mass()[2] < (9.37 + (9.37+cutoff)) / 2:
                    side_of_pore = 1
                elif tail_group.center_of_mass()[2] > (9.37 + (9.37+cutoff)) / 2:
                    side_of_pore = -1
            xyz = tail_group.positions 
            plane_vector = xyz[1] - xyz[0]
            tail_angles[n_frame, n_tail] = angle_between([plane_vector[0], plane_vector[1], plane_vector[2]], [0, 0, 1]) * (180 / np.pi)

    tail_angles = tail_angles.reshape((-1))
    tail_angles = tail_angles[np.logical_not(np.isnan(tail_angles))]
    histo = np.histogram(tail_angles, bins=90, range=(0.0, 180.0), density=True)
    new_bins = get_center(histo[1])
    np.savetxt(f'histo_tail_angles_{cutoff}.txt', np.transpose(np.vstack([new_bins, histo[0]])),
        header='Count\tAngle')
    np.savetxt(f'tail_angles_{cutoff}.txt', np.asarray(tail_angles))
    fig, ax = plt.subplots()
    counts, angle_bins, bars = ax.hist(tail_angles, bins=90, range=(0.0, 180.0), density=True)
    angle_bins_center = (angle_bins[:-1] + angle_bins[1:])/2
    plt.xlim((0, 180))
    plt.ylim((0, 0.06))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    fig.savefig(f'unnormalized_tail_angles_{cutoff}.pdf')

    normalized_counts = np.divide(counts,
        abs(np.sin((np.pi / 180)*angle_bins_center)))
    fig, ax = plt.subplots()
    arr = ax.hist(angle_bins_center, weights=normalized_counts, bins=90, range=(0.0, 180.0),
        density=False)
    plt.xlim((0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    fig.savefig(f'normalized_tail_angles_{cutoff}.pdf')
    np.savetxt(f'normalized_tail_angles_{cutoff}.txt', np.transpose(np.vstack([angle_bins_center, normalized_counts])), header="bins\tnormalized counts")

def _compute_taa_angles(universe, tam_length, cutoff):
    #taa_groups = [universe.select_atoms('(type seiji_007 and resid {} and bonded type seiji_008) or (type seiji_008 and resid {})'.format(r.resid, r.resid)) for r in universe.residues if r.resname == 'tam']
    taa_groups = [universe.select_atoms('(type seiji_007 and resid {} and bonded type seiji_008) or (type seiji_003 and resid {})'.format(r.resid, r.resid)) for r in universe.residues if r.resname == 'tam']

    #taa_angles = np.nan * np.empty(shape=(universe.trajectory.n_frames-1000, len(taa_groups)))
    taa_angles = np.nan * np.empty(shape=(4000, len(taa_groups)))
    """
    >>> np.where(np.isnan([4, 4, ]))[0].shape[0]
    0
    >>> np.where(np.isnan(x))[0].shape[0]
    4
    """

    for n_frame, frame in enumerate(universe.trajectory[:4000:1]):
        for n_taa, taa_group in enumerate(taa_groups):
            # For some reason the zero frame exists twice
            if n_frame == 0:
                continue
            # Consider first layer of ions
            #if ring_group.center_of_mass()[2] > 6.82 + 5.0:
            # These are coordinates of H atoms
            if tam_length == 12:
                z_length = 29.47
            elif tam_length == 16:
                z_length = 32.62
            pore1 = taa_group.center_of_mass()[2] > z_length and taa_group.center_of_mass()[2] < (z_length+cutoff)
            pore2 = taa_group.center_of_mass()[2] > 9.37 or taa_group.center_of_mass()[2] < 9.37 + cutoff
            if not pore1 or not pore2:
                continue
            # Only consider ions in the pore, max is hardcoded to edge of MXene sheet
            if taa_group.center_of_mass()[1] < 10 or taa_group.center_of_mass()[1] > (54.67-10):
                continue
            # Check which side of pore ion is on to determine normal vector of surface
            if pore1:
                if taa_group.positions[0][2] < (z_length + (z_length+cutoff)) / 2:
                    side_of_pore = 1
                else:
                    side_of_pore = -1
            if pore2:
                if taa_group.positions[0][2] < (9.37 + (9.37+cutoff)) / 2:
                    side_of_pore = 1
                else:
                    side_of_pore = -1
            xyz = taa_group.positions 
            plane_vector = xyz[1] - xyz[0]
            taa_angles[n_frame, n_taa] = angle_between([plane_vector[0], plane_vector[1], plane_vector[2]], [0, 0, 1]) * (180 / np.pi)

    taa_angles
    taa_angles = taa_angles.reshape((-1))
    taa_angles = taa_angles[np.logical_not(np.isnan(taa_angles))]
    histo = np.histogram(taa_angles, bins=90, range=(0.0, 180.0), density=True)
    new_bins = get_center(histo[1])
    np.savetxt(f'histo_taa_angles_{cutoff}.txt', np.transpose(np.vstack([new_bins, histo[0]])),
        header='Count\tAngle')
    np.savetxt(f'taa_angles_{cutoff}.txt', np.asarray(taa_angles))
    fig, ax = plt.subplots()
    counts, angle_bins, bars = ax.hist(taa_angles, bins=90, range=(0.0, 180.0), density=True)
    angle_bins_center = (angle_bins[:-1] + angle_bins[1:])/2
    plt.xlim((0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    fig.savefig(f'unnormalized_taa_angles_{cutoff}.pdf')

    normalized_counts = np.divide(counts,
        abs(np.sin((np.pi / 180)*angle_bins_center)))
    fig, ax = plt.subplots()
    arr = ax.hist(angle_bins_center, weights=normalized_counts, bins=190, range=(0.0, 180.0),
        density=False)
    plt.xlim((0, 180))
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Probability')
    fig.savefig(f'normalized_taa_angles_{cutoff}.pdf')
    np.savetxt(f'normalized_taa_angles_{cutoff}.txt', np.transpose(np.vstack([angle_bins_center, normalized_counts])), header="bins\tnormalized counts")

def get_center(bins):
    new_bins = list()
    for idx, bi in enumerate(bins):
        if (idx+1) >= len(bins):
            continue
        mid = (bins[idx] + bins[idx+1])/2
        new_bins.append(mid)

    return new_bins
