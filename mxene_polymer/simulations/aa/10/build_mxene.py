from mxene_polymer.build_functions.aa_mxenes import build_alkylammonium_mxene#, build_tam_custom

composition = {'OH': 1}
chain_length = 10
periods = [26, 26, 1]
#displacement = 1.1
displacement = 2.0
n_compounds = 65

build_alkylammonium_mxene(chain_length=chain_length,
        displacement=displacement,
        n_compounds=n_compounds,
        composition=composition,
        periods=periods)

#build_tam_custom(periods, composition, 'npt.gro', 'bulk.top', displacement)
