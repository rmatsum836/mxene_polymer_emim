import mdtraj as md
from ramtools.calc_mxene_numden import calc_number_density, plot_mxene_numden

area = 41.84
box_range = [1.65, 5.31]

#resnames = {'water_o': '"8"', 'water_h': '"9"', 'OH': '"4" "5"', 'F': '"6"', 'O': '"7"'}
resnames = {'tam_N': 'N',
        'endC': 'CE',
        'midC': 'CM',
        'branchC': 'CB1 CB2'}

calc_number_density('ti3c2.gro', 'mxene_wrapped.dcd', bin_width=0.01,
        area=area, dim=2, box_range=box_range,
        data_path='numden',
        resnames=resnames)

plot_mxene_numden(['tam_N', 'endC', 'midC', 'branchC'], path='numden', ylim=(0,150))

