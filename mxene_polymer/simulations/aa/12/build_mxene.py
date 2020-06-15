from mxene_polymer.build_functions.aa_mxenes import build_alkylammonium_mxene

composition = {'OH': 1}
chain_length = 12
periods = [20, 20, 1]
#displacement = 1.1
displacement = 1.5
n_compounds = 50

build_alkylammonium_mxene(chain_length=chain_length,
        displacement=displacement,
        n_compounds=n_compounds,
        composition=composition,
        periods=periods)
