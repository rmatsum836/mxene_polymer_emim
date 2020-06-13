from mxenes.structures import build_structure
from mxenes.utils.utils import get_fn

import numpy as np
import mbuild as mb
from foyer import Forcefield
from ilforcefields.utils.utils import get_ff
from parmed.gromacs.gromacstop import GromacsTopologyFile
from mtools.gromacs.gromacs import parse_nonbond_params
from mbuild.formats.lammpsdata import write_lammpsdata

composition = {'O': 1}
dspacing = 1.1 # nm

periods = [12, 12, 1]
ti3c2 = build_structure(periods=periods,
        composition=[0, 1, 0], dspacing=0.7, lateral_shift=True,
        atomtype=True)

aa = mb.load('aa_12.mol2')
aa.name = 'alkylam'

lopes = get_ff('lopes')

region1 = mb.Box(mins=[0,
                       0,
                       (ti3c2.box[2] / 10 - 2 * 0.5) / 4 + 0.15],
                 maxs=[ti3c2.box[0] / 10,
                       ti3c2.box[1] / 10,
                       (ti3c2.box[2] / 10 -2 * 0.5) / 4 + 1.15])

region2 = mb.Box(mins=[0,
                       0,
                       (ti3c2.box[2] / 10 - 2 * 0.5) * 3 / 4 + 0.5 + 0.15],
                 maxs=[ti3c2.box[0] / 10,
                       ti3c2.box[1] / 10,
                       (ti3c2.box[2] / 10 - 2 * 0.5) * 3 / 4 + 1.65])

aa_1 = mb.fill_box(
    compound=aa,
    n_compounds=50,
    box=region1,
    fix_orientation=True)

water2 = mb.fill_box(
    compound=aa,
    n_compounds=50,
    box=region2,
    fix_orientation=True)

aaPM = lopes.apply(aa, residues=['alkylam'])
system = aaPM + ti3c2

write_lammpsdata(system, 'data.mxene')
