from mxene_polymer.build_functions.aa_mxenes import build_alkylammonium_mxene

composition = {'O': 1}
chain_length = 1
periods = [12, 12, 1]
displacement = 1.1
n_compounds = 20

build_alkylammonium_mxene(chain_length=chain_length,
        displacement=displacement,
        n_compounds=n_compounds,
        composition=composition,
        periods=periods)
