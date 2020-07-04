from mxenes.structures import build_structure, change_charge, _apply_nbfixes, collapse_atomtypes
from mxenes.utils.utils import get_fn

import numpy as np
import mbuild as mb
from foyer import Forcefield
from ilforcefields.utils.utils import get_ff, get_il
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


    # Rename some carbons
    # CE = end carbon
    # CM = middle carbon
    # CB = branch carbon
    #for idx, particle in enumerate(aa.particles()):
    #    if idx == 0:
    #        particle.name = 'C_E'
    #    if idx == chain_length / 2:
    #        particle.name = 'C_M'
    #    # TODO: Find way to get branch carbon index without
    #    # hardcoding
    #    if idx == 13:
    #        particle.name = 'C_B1'
    #    if idx == 14:
    #        particle.name = 'C_B2'

    aa.name = 'alkylam'
    
    lopes = get_ff('lopes')
    
    region1 = mb.Box(mins=[0,
                           0,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement])
    region2 = mb.Box(mins=[0,
                           0,
                           (ti3c2.box[2] / 10 - 2 * displacement) + displacement + 0.15],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           (ti3c2.box[2] / 10 - 2 * displacement) + 2 + displacement])
    
    aa_1 = mb.fill_box(
        compound=aa,
        n_compounds=n_compounds,
        box=region1,
        fix_orientation=True)

    aa_2 = mb.fill_box(
        compound=aa,
        n_compounds=n_compounds,
        box=region2,
        fix_orientation=True)
        
    aa1PM = lopes.apply(aa_1, residues=['alkylam'],
            assert_dihedral_params=False)
    aa2PM = lopes.apply(aa_2, residues=['alkylam'],
            assert_dihedral_params=False)

    system = aa1PM + aa2PM + ti3c2
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    system.box = ti3c2.box
    for atom in system.atoms:
        if atom.name == 'C_E':
            atom.name = 'CE'
        elif atom.name == 'C_M':
            atom.name = 'CM'
        elif atom.name == 'C_B1':
            atom.name = 'CB1'
        elif atom.name == 'C_B2':
            atom.name = 'CB2'
   
    system.save('ti3c2.gro', combine='all', overwrite=True)
    system.save('ti3c2.top', combine='all', overwrite=True)
    write_lammpsdata(system, 'data.mxene')


def build_tam_emim_mxene(n_compounds, composition, periods, chain_length=12,
        displacement=1.1):
    """Build a MXene with TAM and EMIM in the interlayer

    Parameters
    ----------
    n_compounds : list
        List of [TAM, EMIM] compounds
    composition : dict
        Composition of MXene surface groups
    periods : list
        Periods of MXene crystal
    chain_length : int
        Chain length of TAM
    displacement : float
        Interlayer spacing (nm)
    """
    ti3c2 = build_structure(periods=periods,
            composition=composition,
            displacement=displacement,
            lateral_shift=True,
            atomtype=True)
   
    n_carbons = 'C' * chain_length
    aa = mb.load(f'{n_carbons}[N](C)(C)C', smiles=True)

    emim = mb.load(get_il('emim'))
    emim.name = 'emim'

    aa.name = 'alkylam'
    
    lopes = get_ff('lopes')
    
    region1 = mb.Box(mins=[0,
                           0,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement])
    region2 = mb.Box(mins=[0,
                           0,
                           (ti3c2.box[2] / 10 - 2 * displacement) + displacement + 0.15],
                     maxs=[ti3c2.box[0] / 10,
                           ti3c2.box[1] / 10,
                           (ti3c2.box[2] / 10 - 2 * displacement) + 2 + displacement])
    
    aa_1 = mb.fill_box(
        compound=[aa, emim],
        n_compounds=n_compounds,
        box=region1,
        fix_orientation=True)

    aa_2 = mb.fill_box(
        compound=[aa, emim],
        n_compounds=n_compounds,
        box=region2,
        fix_orientation=True)
        
    aa1PM = lopes.apply(aa_1, residues=['alkylam', 'emim'],
            assert_dihedral_params=False)
    aa2PM = lopes.apply(aa_2, residues=['alkylam', 'emim'],
            assert_dihedral_params=False)

    system = aa1PM + aa2PM + ti3c2
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    system.box = ti3c2.box
   
    system.save('ti3c2.gro', combine='all', overwrite=True)
    system.save('ti3c2.top', combine='all', overwrite=True)
    write_lammpsdata(system, 'data.mxene')
