from mxenes.structures import build_structure, change_charge
from mxenes.utils.utils import get_fn

import numpy as np
import mbuild as mb
from foyer import Forcefield
from ilforcefields.utils.utils import get_ff
from parmed.gromacs.gromacstop import GromacsTopologyFile
from mtools.gromacs.gromacs import parse_nonbond_params
from mbuild.formats.lammpsdata import write_lammpsdata

def build_alkylammonium_mxene(n_compounds, composition, periods, chain_length=12,
        displacement=1.1):
    ti3c2 = build_structure(periods=periods,
            composition=composition,
            displacement=displacement,
            lateral_shift=True,
            atomtype=True)
   
    n_carbons = 'C' * chain_length
    aa = mb.load(f'{n_carbons}[N](C)(C)C', smiles=True)
    aa.name = 'alkylam'
    
    lopes = get_ff('lopes')
    
    region1 = mb.Box(mins=[0,
                           0,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement])
    
    aa_1 = mb.fill_box(
        compound=aa,
        n_compounds=n_compounds,
        box=region1,
        fix_orientation=True)
        
    aaPM = lopes.apply(aa_1, residues=['alkylam'],
            assert_dihedral_params=False)
    system = aaPM + ti3c2
    change_charge(system, new_charge=0)
   
    system.save('ti3c2.gro', combine='all', overwrite=True)
    system.save('ti3c2.top', combine='all', overwrite=True)
    write_lammpsdata(system, 'data.mxene')
