from mxene_polymer.build_functions.aa_mxenes import build_alkylammonium_mxene

composition = {'OH': 1}
chain_length = 12
periods = [26, 26, 1]
#displacement = 1.1
displacement = 2.4
n_compounds = 120

build_alkylammonium_mxene(chain_length=chain_length,
        displacement=displacement,
        n_compounds=n_compounds,
        composition=composition,
        periods=periods)
