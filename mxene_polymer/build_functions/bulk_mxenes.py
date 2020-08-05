from mxenes.structures import build_structure, change_charge, _apply_nbfixes, collapse_atomtypes
from mxenes.utils.utils import get_fn

import numpy as np
import mbuild as mb
import parmed as pmd
from foyer import Forcefield
from ilforcefields.utils.utils import get_ff, get_il
from parmed.gromacs.gromacstop import GromacsTopologyFile
from mtools.gromacs.gromacs import parse_nonbond_params

def build_bulk_mxene(n_compounds, composition, periods, chain_length=12,
        displacement=1.1):
    displacement -= 0.08
    ti3c2 = build_structure(periods=periods,
            composition=composition,
            displacement=displacement,
            lateral_shift=True,
            atomtype=True)
   
    n_carbons = 'C' * chain_length
    aa = mb.load(f'{n_carbons}[N](C)(C)C', smiles=True)
    emim = mb.load(get_il('emim'))
    tf2n = mb.load(get_il('tf2n'))
    emim.name = 'emim'
    tf2n.name = 'tf2n'
    aa.name = 'alkylam'
    
    lopes = get_ff('lopes')
    kpl = get_ff('kpl')

    bulk_region = mb.Box(mins=[0,
                               ti3c2.box[1]/10,
                               0],
                         maxs=[ti3c2.box[0]/10,
                               ti3c2.box[1]/10 + 10,
                               ti3c2.box[2]/10])
    
    region1 = mb.Box(mins=[0,
                           0,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement])
    region2 = mb.Box(mins=[0,
                           0,
                           np.max(ti3c2.coordinates[:,2])/10],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           np.max(ti3c2.coordinates[:,2])/10 + displacement + 0.08])
    
    aa_1 = mb.fill_box(
        compound=aa,
        n_compounds=n_compounds,
        box=region1,
        )

    aa_2 = mb.fill_box(
        compound=aa,
        n_compounds=n_compounds,
        box=region2,
        )

    bulk = mb.fill_box(
        compound=[emim, tf2n],
        density=1420,
        compound_ratio=[0.5, 0.5],
        box=bulk_region,
        fix_orientation=True)

    cation = mb.Compound()
    anion = mb.Compound()

    for child in bulk.children:
        if child.name == 'emim':
            cation.add(mb.clone(child))
        else:
            anion.add(mb.clone(child))
        
    aa1PM = lopes.apply(aa_1, residues=['alkylam'],
            assert_dihedral_params=False)
    aa2PM = lopes.apply(aa_2, residues=['alkylam'],
            assert_dihedral_params=False)

    cationPM = kpl.apply(cation, residues='emim',
            assert_dihedral_params=False)
    anionPM = kpl.apply(anion, residues='tf2n',
            assert_dihedral_params=False)

    system = aa1PM + aa2PM + ti3c2 + cationPM + anionPM
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    system.box[1] = bulk.boundingbox.maxs[1] * 10
    system.box[2] = max_ti3c2[2] + (displacement*10) + .8
   
    system.save('ti3c2.gro', combine='all', overwrite=True)
    system.save('ti3c2.top', combine='all', overwrite=True)
