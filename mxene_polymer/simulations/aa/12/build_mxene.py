from build_functions.aa_mxenes import build_alkylammonium_mxene

composition = {'O': 1}
chain_length = 12
periods = [12, 12, 1]
dspacing = 1.1
n_compounds = 50

build_alkylammonium_mxene(chain_length=chain_length,
        dspacing=dspacing,
        n_compounds=n_compounds,
        composition=composition,
        periods=periods)
