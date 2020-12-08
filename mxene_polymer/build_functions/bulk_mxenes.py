from mxenes.structures import (
    build_structure,
    change_charge,
    _apply_nbfixes,
    collapse_atomtypes,
)
from mxenes.utils.utils import get_fn
from mxenes.lattices import Ti3C2Functionalized
from copy import deepcopy

import numpy as np
import mbuild as mb
import parmed as pmd
from foyer import Forcefield
from ilforcefields.utils.utils import get_ff, get_il
from parmed.gromacs.gromacstop import GromacsTopologyFile
from mtools.gromacs.gromacs import parse_nonbond_params


def build_bulk_mxene(
    n_compounds,
    composition,
    periods,
    density=1420,
    chain_length=12,
    displacement=1.1,
    emim_ff="kpl",
):
    """Function to build MXene exposed to bulk EMIM-TFSI

    Parameters
    ----------
    n_compounds : int
        Number of alkylammonium cations to initiialize in pore
    periods : list of ints
        Lattice repeat units for MXene
    density : int or float, default=1420
        Density of EMIM-TFSI in bulk
    chain_length : int, default=12
        Chain length of alkylammonium cations
    displacement : int or float, default=1.1
        Displacement of MXene interlayer
    emim_ff : str, default="kpl"
        Forcefield to use for EMIM-TFSI atoms

    Returns
    -------
    pmd.Structure : Parameterized ParmEd structure
    """
    displacement -= 0.08
    ti3c2 = build_structure(
        periods=periods,
        composition=composition,
        displacement=displacement,
        lateral_shift=True,
        atomtype=True,
    )

    for bond in ti3c2.bonds:
        bond.delete()

    for angle in ti3c2.angles:
        angle.delete()
    ti3c2._prune_empty_bonds()
    ti3c2._prune_empty_angles()

    n_carbons = "C" * chain_length
    aa = mb.load(f"{n_carbons}[N](C)(C)C", smiles=True)
    emim = mb.load(get_il("emim"))
    tf2n = mb.load(get_il("tf2n"))
    emim.name = "emim"
    tf2n.name = "tf2n"
    aa.name = "alkylam"

    lopes = get_ff("lopes")
    if emim_ff == "kpl":
        kpl = get_ff("kpl")

    bulk_region = mb.Box(
        mins=[0, ti3c2.box[1] / 10, 0],
        maxs=[ti3c2.box[0] / 10, ti3c2.box[1] / 10 + 10, ti3c2.box[2] / 10],
    )

    region1 = mb.Box(
        mins=[0, 0, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )
    region2 = mb.Box(
        mins=[0, 0, np.max(ti3c2.coordinates[:, 2]) / 10],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            np.max(ti3c2.coordinates[:, 2]) / 10 + displacement + 0.08,
        ],
    )

    if n_compounds != 0:
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
        aa1PM = lopes.apply(aa_1, residues=["alkylam"], assert_dihedral_params=False)
        aa2PM = lopes.apply(aa_2, residues=["alkylam"], assert_dihedral_params=False)

    if density != 0:
        bulk = mb.fill_box(
            compound=[emim, tf2n],
            density=density,
            compound_ratio=[0.5, 0.5],
            box=bulk_region,
            fix_orientation=True,
        )

        cation = mb.Compound()
        anion = mb.Compound()

        for child in bulk.children:
            if child.name == "emim":
                cation.add(mb.clone(child))
            else:
                anion.add(mb.clone(child))

        if emim_ff == "kpl":
            cationPM = kpl.apply(cation, residues="emim", assert_dihedral_params=False)
            anionPM = kpl.apply(anion, residues="tf2n", assert_dihedral_params=False)
        else:
            cationPM = lopes.apply(
                cation, residues="emim", assert_dihedral_params=False
            )
            anionPM = lopes.apply(anion, residues="tf2n", assert_dihedral_params=False)

    if n_compounds != 0:
        ils = aa1PM + aa2PM + cationPM + anionPM
    elif density != 0:
        ils = cationPM + anionPM

    if emim_ff == "lopes":
        for atom in ils.atoms:
            atom.charge *= 0.8
    if n_compounds != 0 or density != 0:
        system = ils + ti3c2
    else:
        system = ti3c2
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    if density != 0:
        system.box[1] = bulk.boundingbox.maxs[1] * 10
    else:
        system.box[1] = 20 * 10
    system.box[2] = max_ti3c2[2] + (displacement * 10) + 0.8

    system.save("ti3c2.gro", combine="all", overwrite=True)
    system.save("ti3c2.top", combine="all", overwrite=True)


def build_awh_mxene(
    n_compounds, composition, periods, chain_length=12, displacement=1.1
):
    displacement -= 0.08
    ti3c2 = build_structure(
        periods=periods,
        composition=composition,
        displacement=displacement,
        lateral_shift=True,
        atomtype=True,
    )

    n_carbons = "C" * chain_length
    aa = mb.load(f"{n_carbons}[N](C)(C)C", smiles=True)
    emim = mb.load(get_il("emim"))
    tf2n = mb.load(get_il("tf2n"))
    emim.name = "emim"
    tf2n.name = "tf2n"
    aa.name = "alkylam"
    emim_awh = mb.clone(emim)
    emim_awh.name = "awh"

    lopes = get_ff("lopes")
    kpl = get_ff("kpl")

    bulk_region = mb.Box(
        mins=[0, ti3c2.box[1] / 10, 0],
        maxs=[ti3c2.box[0] / 10, ti3c2.box[1] / 10 + 10, ti3c2.box[2] / 10],
    )

    region1 = mb.Box(
        mins=[0, 0, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )

    ti3c2_max = np.max(ti3c2.coordinates, axis=0)
    region2 = mb.Box(
        mins=[0, 0, ti3c2_max[2] / 10],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            ti3c2_max[2] / 10 + displacement + 0.08,
        ],
    )

    aa_1 = mb.fill_box(
        compound=[aa, emim_awh],
        n_compounds=[n_compounds, 1],
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
        fix_orientation=True,
    )

    cation = mb.Compound()
    anion = mb.Compound()

    marker = 0
    for child in bulk.children:
        if child.name == "emim":
            if cation.n_particles == 1:
                marker = 1
            if marker == 1:
                cation.add(mb.clone(child))
        else:
            anion.add(mb.clone(child))

    tam1 = mb.Compound()
    new_emim_awh = mb.Compound()

    for child in aa_1.children:
        if child.name == "awh":
            new_emim_awh.add(mb.clone(child))
        else:
            tam1.add(mb.clone(child))

    new_emim_awh.translate_to(
        [(ti3c2_max[0] / 10) / 2, ti3c2_max[1] / 10, 0.937 + (displacement / 2) + 0.08]
    )

    aa1PM = lopes.apply(tam1, residues=["alkylam"], assert_dihedral_params=False)
    aa2PM = lopes.apply(aa_2, residues=["alkylam"], assert_dihedral_params=False)

    cationPM = kpl.apply(cation, residues="emim", assert_dihedral_params=False)
    anionPM = kpl.apply(anion, residues="tf2n", assert_dihedral_params=False)
    emim_awhPM = kpl.apply(new_emim_awh, residues="awh", assert_dihedral_params=False)

    system = aa1PM + aa2PM + ti3c2 + cationPM + anionPM + emim_awhPM
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    system.box[1] = bulk.boundingbox.maxs[1] * 10
    system.box[2] = max_ti3c2[2] + (displacement * 10) + 0.8

    system.save("init.gro", combine="all", overwrite=True)
    system.save("init.top", combine="all", overwrite=True)


def build_vacuum_mxene(
    n_compounds, composition, periods, chain_length=12, displacement=1.1
):
    """Function to build MXene exposed to vacuum space

    Parameters
    ----------
    n_compounds : int
        Number of alkylammonium cations to initiialize in pore
    composition : dict {"O": composition: "OH": composition, "F": composition}
        Composition of surface groups 
    periods : list of ints
        Lattice repeat units for MXene
    chain_length : int, default=12
        Chain length of alkylammonium cations
    displacement : int or float, default=1.1
        Displacement of MXene interlayer

    Returns
    -------
    pmd.Structure : Parameterized ParmEd structure
    """
    displacement -= 0.08
    ti3c2 = build_structure(
        periods=periods,
        composition=composition,
        displacement=displacement,
        lateral_shift=True,
        atomtype=True,
    )

    n_carbons = "C" * chain_length
    aa = mb.load(f"{n_carbons}[N](C)(C)C", smiles=True)
    emim = mb.load(get_il("emim"))
    emim.name = "pull"
    aa.name = "alkylam"

    lopes = get_ff("lopes")
    kpl = get_ff("kpl")

    bulk_region = mb.Box(
        mins=[0, ti3c2.box[1] / 10, 0],
        maxs=[ti3c2.box[0] / 10, ti3c2.box[1] / 10 + 10, ti3c2.box[2] / 10],
    )

    region1 = mb.Box(
        mins=[0, 0, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )

    ti3c2_max = np.max(ti3c2.coordinates, axis=0)
    region2 = mb.Box(
        mins=[0, 0, ti3c2_max[2] / 10],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            ti3c2_max[2] / 10 + displacement + 0.08,
        ],
    )

    aa_1 = mb.fill_box(
        compound=[aa, emim],
        n_compounds=[n_compounds, 1],
        box=region1,
    )

    aa_2 = mb.fill_box(
        compound=aa,
        n_compounds=n_compounds,
        box=region2,
    )

    tam1 = mb.Compound()
    new_emim_awh = mb.Compound()

    for child in aa_1.children:
        if child.name == "pull":
            new_emim_awh.add(mb.clone(child))
        else:
            tam1.add(mb.clone(child))

    new_emim_awh.translate_to(
        [
            (ti3c2_max[0] / 10) / 2,
            ti3c2_max[1] / 10 - 2,
            0.937 + (displacement / 2) + 0.08,
        ]
    )

    aa1PM = lopes.apply(tam1, residues=["alkylam"], assert_dihedral_params=False)
    aa2PM = lopes.apply(aa_2, residues=["alkylam"], assert_dihedral_params=False)

    emim_awhPM = kpl.apply(new_emim_awh, residues="pull", assert_dihedral_params=False)

    system = aa1PM + aa2PM + ti3c2 + emim_awhPM
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    system.box[1] = bulk_region.maxs[1] * 10
    system.box[2] = max_ti3c2[2] + (displacement * 10) + 0.8

    system.save("vacuum.gro", combine="all", overwrite=True)
    system.save("vacuum.top", combine="all", overwrite=True)


def build_awh_mxene(
    n_compounds, composition, periods, chain_length=12, displacement=1.1
):
    """Function to build MXene exposed to bulk EMIM-TFSI for awh simulations

    Parameters
    ----------
    n_compounds : int
        Number of alkylammonium cations to initiialize in pore
    composition : dict {"O": composition: "OH": composition, "F": composition}
        Composition of surface groups 
    periods : list of ints
        Lattice repeat units for MXene
    density : int or float, default=1420
        Density of EMIM-TFSI in bulk
    chain_length : int, default=12
        Chain length of alkylammonium cations
    displacement : int or float, default=1.1
        Displacement of MXene interlayer

    Returns
    -------
    pmd.Structure : Parameterized ParmEd structure
    """
    displacement -= 0.08
    ti3c2 = build_structure(
        periods=periods,
        composition=composition,
        displacement=displacement,
        lateral_shift=True,
        atomtype=True,
    )

    n_carbons = "C" * chain_length
    aa = mb.load(f"{n_carbons}[N](C)(C)C", smiles=True)
    emim = mb.load(get_il("emim"))
    tf2n = mb.load(get_il("tf2n"))
    emim.name = "emim"
    tf2n.name = "tf2n"
    aa.name = "alkylam"
    emim_awh = mb.clone(emim)
    emim_awh.name = "awh"

    lopes = get_ff("lopes")
    kpl = get_ff("kpl")

    bulk_region = mb.Box(
        mins=[0, ti3c2.box[1] / 10, 0],
        maxs=[ti3c2.box[0] / 10, ti3c2.box[1] / 10 + 10, ti3c2.box[2] / 10],
    )

    region1 = mb.Box(
        mins=[0, 0, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )

    ti3c2_max = np.max(ti3c2.coordinates, axis=0)
    region2 = mb.Box(
        mins=[0, 0, ti3c2_max[2] / 10],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            ti3c2_max[2] / 10 + displacement + 0.08,
        ],
    )

    aa_1 = mb.fill_box(
        compound=[aa, emim_awh],
        n_compounds=[n_compounds, 1],
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
        fix_orientation=True,
    )

    cation = mb.Compound()
    anion = mb.Compound()

    marker = 0
    for child in bulk.children:
        if child.name == "emim":
            if cation.n_particles == 1:
                marker = 1
            if marker == 1:
                cation.add(mb.clone(child))
        else:
            anion.add(mb.clone(child))

    tam1 = mb.Compound()
    new_emim_awh = mb.Compound()

    for child in aa_1.children:
        if child.name == "awh":
            new_emim_awh.add(mb.clone(child))
        else:
            tam1.add(mb.clone(child))

    new_emim_awh.translate_to(
        [(ti3c2_max[0] / 10) / 2, ti3c2_max[1] / 10, 0.937 + (displacement / 2) + 0.08]
    )

    aa1PM = lopes.apply(tam1, residues=["alkylam"], assert_dihedral_params=False)
    aa2PM = lopes.apply(aa_2, residues=["alkylam"], assert_dihedral_params=False)

    cationPM = kpl.apply(cation, residues="emim", assert_dihedral_params=False)
    anionPM = kpl.apply(anion, residues="tf2n", assert_dihedral_params=False)
    emim_awhPM = kpl.apply(new_emim_awh, residues="awh", assert_dihedral_params=False)

    system = aa1PM + aa2PM + ti3c2 + cationPM + anionPM + emim_awhPM
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    system.box[1] = bulk.boundingbox.maxs[1] * 10
    system.box[2] = max_ti3c2[2] + (displacement * 10) + 0.8

    system.save("init.gro", combine="all", overwrite=True)
    system.save("init.top", combine="all", overwrite=True)


def build_emim_in_pore(
    n_tam,
    composition,
    periods,
    n_pore_emim,
    lattice=Ti3C2Functionalized,
    bulk_density=None,
    bulk_n_ions=None,
    chain_length=12,
    displacement=1.1,
    emim_ff="kpl",
    taa_ff="seiji",
):
    """Function to build MXene exposed to bulk EMIM-TFSI with some EMIM-TFSI in pore

    Parameters
    ----------
    n_tam : int
        Number of alkylammonium cations to initiialize in pore
    composition : dict {"O": composition: "OH": composition, "F": composition}
        Composition of surface groups 
    periods : list of ints
        Lattice repeat units for MXene
    n_pore_emim : int
        Number of EMIM-TFSI ions to place in pore
    lattice : class, default=Ti3C2Functionalized
        MXene lattice class to use
    bulk_density : int or float, default=1420
        Density of EMIM-TFSI in bulk
    bulk_n_ions : int or float, default=1420
        Number of EMIM-TFSI in bulk
    chain_length : int, default=12
        Chain length of alkylammonium cations
    displacement : int or float, default=1.1
        Displacement of MXene interlayer
    emim_ff : str, default="kpl"
        Forcefield to use for EMIM-TFSI atoms
    taa_ff : str, default="seiji"
        Forcefield to use for TAM atoms

    Returns
    -------
    pmd.Structure : Parameterized ParmEd structure
    """

    if bulk_density is None and bulk_n_ions is None:
        raise TypeError("`bulk_density` and `bulk_n_ions` cannot both be of type None")
    if bulk_density and bulk_n_ions:
        raise TypeError("Either `bulk_density` or `bulk_n_ions` should be of type None")

    displacement -= 0.08
    ti3c2 = build_structure(
        lattice=lattice,
        periods=periods,
        composition=composition,
        displacement=displacement,
        lateral_shift=True,
        atomtype=True,
    )

    for bond in ti3c2.bonds:
        bond.delete()

    for angle in ti3c2.angles:
        angle.delete()
    ti3c2._prune_empty_bonds()
    ti3c2._prune_empty_angles()

    n_carbons = "C" * chain_length
    aa = mb.load(f"{n_carbons}[N](C)(C)C", smiles=True)
    emim = mb.load(get_il("emim"))
    tf2n = mb.load(get_il("tf2n"))
    emim.name = "emim"
    tf2n.name = "tf2n"
    aa.name = "tam"

    # Rename some carbons
    # CE = end carbon
    # CM = middle carbon
    # CB = branch carbon

    for idx, particle in enumerate(aa.particles()):
        if idx == 0:
            particle.name = "C_E"
        if idx == chain_length / 2:
            particle.name = "C_M"
        # TODO: Find way to get branch carbon index without
        # hardcoding
        if particle.name == "N":
            n_index = idx
        try:
            if idx == n_index + 1:
                particle.name = "C_B1"
            if idx == n_index + 2:
                particle.name = "C_B2"
        except:
            continue

    taaff = get_ff(taa_ff)
    if emim_ff == "kpl":
        kpl = get_ff("kpl")

    bulk_region = mb.Box(
        mins=[0, ti3c2.box[1] / 10, 0],
        maxs=[ti3c2.box[0] / 10, ti3c2.box[1] / 10 + 10, ti3c2.box[2] / 10 - 0.05],
    )

    region1 = mb.Box(
        mins=[0, 0, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )
    region2 = mb.Box(
        mins=[0, 0, np.max(ti3c2.coordinates[:, 2]) / 10],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            np.max(ti3c2.coordinates[:, 2]) / 10 + displacement + 0.08,
        ],
    )

    aa_1 = mb.fill_box(
        compound=[aa, emim, tf2n],
        n_compounds=[n_tam, n_pore_emim, n_pore_emim],
        box=region1,
    )

    aa_2 = mb.fill_box(
        compound=[aa, emim, tf2n],
        n_compounds=[n_tam, n_pore_emim, n_pore_emim],
        box=region2,
    )

    aa_compound = mb.Compound()
    for child in aa_1.children:
        if child.name == "tam":
            aa_compound.add(mb.clone(child))
    for child in aa_2.children:
        if child.name == "tam":
            aa_compound.add(mb.clone(child))

    print("Atomtyping TAM")
    aaPM = taaff.apply(aa_compound, residues="tam", assert_dihedral_params=False)

    if bulk_density:
        bulk = mb.fill_box(
            compound=[emim, tf2n],
            density=bulk_density,
            compound_ratio=[0.5, 0.5],
            box=bulk_region,
            fix_orientation=True,
        )
    elif bulk_n_ions:
        bulk = mb.fill_box(
            compound=[emim, tf2n],
            n_compounds=[bulk_n_ions, bulk_n_ions],
            box=bulk_region,
            fix_orientation=True,
        )

    cation = mb.Compound()
    anion = mb.Compound()

    for child in bulk.children:
        if child.name == "emim":
            cation.add(mb.clone(child))
        else:
            anion.add(mb.clone(child))
    for child in aa_1.children:
        if child.name == "emim":
            cation.add(mb.clone(child))
        elif child.name == "tf2n":
            anion.add(mb.clone(child))
    for child in aa_2.children:
        if child.name == "emim":
            cation.add(mb.clone(child))
        elif child.name == "tf2n":
            anion.add(mb.clone(child))

    print("Atomtyping emim-tf2n")
    if emim_ff == "kpl":
        cationPM = kpl.apply(cation, residues="emim", assert_dihedral_params=False)
        anionPM = kpl.apply(anion, residues="tf2n", assert_dihedral_params=False)
    else:
        cationPM = lopes.apply(cation, residues="emim", assert_dihedral_params=False)
        anionPM = lopes.apply(anion, residues="tf2n", assert_dihedral_params=False)

    if emim_ff == "lopes":
        print("Scaling EMIM-TF2N charges...")
        for atom in cationPM.atoms:
            atom.charge *= 0.8
        for atom in anionPM.atoms:
            atom.charge *= 0.8

    if taa_ff == "lopes":
        print("Scaling TAA charges...")
        for atom in aaPM.atoms:
            atom.charge *= 0.8
    if taa_ff == "seiji":
        for atom in aaPM.atoms:
            if atom.type == "seiji_004":
                for bond in atom.bonds:
                    if "seiji_006" in (bond.atom1.type, bond.atom2.type):
                        atom.charge = -0.16
                    continue
            if atom.type == "seiji_005":
                for bond in atom.bonds:
                    if "seiji_004" in (bond.atom1.type, bond.atom2.type):
                        if -0.16 in (bond.atom1.charge, bond.atom2.charge):
                            atom.charge = 0.18

    ils = aaPM + cationPM + anionPM

    system = ils + ti3c2
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    system.box[1] = bulk.boundingbox.maxs[1] * 10
    system.box[2] = max_ti3c2[2] + (displacement * 10) + 0.8
    for atom in system.atoms:
        if atom.name == "C_E":
            atom.name = "CE"
        elif atom.name == "C_M":
            atom.name = "CM"
        elif atom.name == "C_B1":
            atom.name = "CB1"
        elif atom.name == "C_B2":
            atom.name = "CB2"

    system.save("ti3c2.gro", combine="all", overwrite=True)
    system.save("ti3c2.top", combine="all", overwrite=True)


def build_two_mxene_pores(
    composition,
    periods,
    bulk_length,
    n_il,
    n_tam=20,
    chain_length=12,
    displacement=1.1,
    emim_ff="kpl",
    tam_ff="seiji",
):
    """Build system with two MXene pores in between a bulk region of EMIM-TF2N

    Parameters
    ----------
    composition : dict
        Composition of surface groups
    periods : list of length 3
        Periods in which to build MXene crystal
    bulk_legnth : float
        length of bulk region between MXene pores
    n_il : list, [n_emim, n_tf2n]
        Number of IL molecules to add into bulk region.
        Half of this value is added to side regions.
        If value is odd number, then the number of compounds in the side regions are rounded.
    n_tam : int, default=20
        Number of TAM molecules to add into pore
    chain_length : int, default=12
        Chain length of TAM
    displacement : float, default=1.1 nm
        Displacement of MXene pore
    emim_ff : str, default="kpl"
        ForceField to apply to EMIM-TF2N
    tam_ff : str , default="seiji"
        ForceField to apply to TAM
    """
    displacement -= 0.08
    ti3c2 = build_structure(
        periods=periods,
        composition=composition,
        displacement=displacement,
        lateral_shift=True,
        atomtype=True,
    )

    second_ti3c2 = deepcopy(ti3c2)
    for atom in second_ti3c2.atoms:
        atom.xy += (bulk_length * 10) + ti3c2.box[1]  # Convert nm to angstrom

    n_carbons = "C" * chain_length
    aa = mb.load(f"{n_carbons}[N](C)(C)C", smiles=True)
    emim = mb.load(get_il("emim"))
    tf2n = mb.load(get_il("tf2n"))
    emim.name = "emim"
    tf2n.name = "tf2n"
    aa.name = "tam"

    tam_ff_object = get_ff(tam_ff)
    emim_ff_object = get_ff(emim_ff)

    # Get mins and maxs for second pore
    mins = np.min([i.xy for i in second_ti3c2.atoms])
    maxs = np.max([i.xy for i in second_ti3c2.atoms])

    bulk_region = mb.Box(
        mins=[0, ti3c2.box[1] / 10, 0],
        maxs=[ti3c2.box[0] / 10, ti3c2.box[1] / 10 + bulk_length, ti3c2.box[2] / 10],
    )

    # Fill regions on side of MXene pores
    side_region_1 = mb.Box(
        mins=[0, -bulk_length / 2, 0], maxs=[ti3c2.box[0] / 10, 0, ti3c2.box[2] / 10]
    )

    side_region_2 = mb.Box(
        mins=[0, maxs / 10, 0],
        maxs=[ti3c2.box[0] / 10, maxs / 10 + bulk_length / 2, ti3c2.box[2] / 10],
    )

    pore1_region1 = mb.Box(
        mins=[0, 0, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )
    pore1_region2 = mb.Box(
        mins=[0, 0, np.max(ti3c2.coordinates[:, 2]) / 10],
        maxs=[
            ti3c2.box[0] / 10,
            ti3c2.box[1] / 10,
            np.max(ti3c2.coordinates[:, 2]) / 10 + displacement + 0.08,
        ],
    )

    pore2_region1 = mb.Box(
        mins=[0, mins / 10, (ti3c2.box[2] / 10 - 2 * displacement) / 2 + 0.15],
        maxs=[
            ti3c2.box[0] / 10,
            maxs / 10,
            (ti3c2.box[2] / 10 - 2 * displacement) / 2 + displacement,
        ],
    )
    pore2_region2 = mb.Box(
        mins=[0, mins / 10, np.max(ti3c2.coordinates[:, 2]) / 10],
        maxs=[
            ti3c2.box[0] / 10,
            maxs / 10,
            np.max(ti3c2.coordinates[:, 2]) / 10 + displacement + 0.08,
        ],
    )

    if n_tam != 0:
        pore1_aa_1 = mb.fill_box(
            compound=aa,
            n_compounds=n_tam,
            box=pore1_region1,
        )

        pore1_aa_2 = mb.fill_box(
            compound=aa,
            n_compounds=n_tam,
            box=pore1_region2,
        )
        pore2_aa_1 = mb.fill_box(
            compound=aa,
            n_compounds=n_tam,
            box=pore2_region1,
        )

        pore2_aa_2 = mb.fill_box(
            compound=aa,
            n_compounds=n_tam,
            box=pore2_region2,
        )

        tam_compounds = mb.Compound()
        for cmpd in [pore1_aa_1, pore1_aa_2, pore2_aa_1, pore2_aa_2]:
            tam_compounds.add(mb.clone(cmpd))

        # pore1_aa1PM = tam_ff_object.apply(pore1_aa_1, residues=['tam'],
        #        assert_dihedral_params=False)
        tamPM = tam_ff_object.apply(
            tam_compounds, residues=["tam"], assert_dihedral_params=False
        )
        # pore1_aa2PM = tam_ff_object.apply(pore1_aa_2, residues=['tam'],
        #        assert_dihedral_params=False)
        # pore2_aa1PM = tam_ff_object.apply(pore2_aa_1, residues=['tam'],
        #        assert_dihedral_params=False)
        # pore2_aa2PM = tam_ff_object.apply(pore2_aa_2, residues=['tam'],
        #        assert_dihedral_params=False)

    if n_il[0] != 0 or n_il[1] != 0:
        bulk = mb.fill_box(
            compound=[emim, tf2n],
            n_compounds=[n_il[0], n_il[1]],
            box=bulk_region,
            fix_orientation=True,
        )

        side1 = mb.fill_box(
            compound=[emim, tf2n],
            n_compounds=[round(n_il[0] / 2), round(n_il[1] / 2)],
            box=side_region_1,
            fix_orientation=True,
        )

        side2 = mb.fill_box(
            compound=[emim, tf2n],
            n_compounds=[round(n_il[0] / 2), round(n_il[1] / 2)],
            box=side_region_2,
            fix_orientation=True,
        )

        cation = mb.Compound()
        anion = mb.Compound()

        for child in bulk.children:
            if child.name == "emim":
                cation.add(mb.clone(child))
            else:
                anion.add(mb.clone(child))
        for child in side1.children:
            if child.name == "emim":
                cation.add(mb.clone(child))
            else:
                anion.add(mb.clone(child))
        for child in side2.children:
            if child.name == "emim":
                cation.add(mb.clone(child))
            else:
                anion.add(mb.clone(child))

        cationPM = emim_ff_object.apply(
            cation, residues="emim", assert_dihedral_params=False
        )
        anionPM = emim_ff_object.apply(
            anion, residues="tf2n", assert_dihedral_params=False
        )

    if tam_ff == "seiji":
        for atom in tamPM.atoms:
            if atom.type == "seiji_004":
                for bond in atom.bonds:
                    if "seiji_006" in (bond.atom1.type, bond.atom2.type):
                        atom.charge = -0.16
                    continue
            if atom.type == "seiji_005":
                for bond in atom.bonds:
                    if "seiji_004" in (bond.atom1.type, bond.atom2.type):
                        if -0.16 in (bond.atom1.charge, bond.atom2.charge):
                            atom.charge = 0.18

    if n_tam != 0:
        ils = tamPM + cationPM + anionPM
    elif n_il[0] != 0 or n_il[1] != 0:
        ils = cationPM + anionPM

    if emim_ff == "lopes":
        for atom in ils.atoms:
            atom.charge *= 0.8
    if n_tam != 0 or n_il[0] != 0 or n_il[1] != 0:
        system = ils + ti3c2 + second_ti3c2
    else:
        system = ti3c2 + second_ti3c2
    system = _apply_nbfixes(system)
    system = collapse_atomtypes(system)
    change_charge(system, new_charge=0)
    max_ti3c2 = np.max(ti3c2.coordinates, axis=0)
    system.box[0] = max_ti3c2[0]
    if n_il[0] != 0 or n_il[1] != 0:
        system.box[1] = ti3c2.box[1] * 2 + (bulk_length * 2) * 10
    else:
        system.box[1] = ti3c2.box[1] * 2 + (bulk_length * 2) * 10
    system.box[2] = max_ti3c2[2] + (displacement * 10) + 0.8

    # Shift all atoms into box
    for atom in system.atoms:
        atom.xy += (bulk_length / 2) * 10

    system.save("ti3c2.gro", combine="all", overwrite=True)
    system.save("ti3c2.top", combine="all", overwrite=True)
