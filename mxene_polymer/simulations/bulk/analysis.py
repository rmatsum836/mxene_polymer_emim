import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from ramtools.structure.calc_number_density import calc_number_density, calc_atom_number_density


def number_density(path):
    """Calculate number density of ions in pore

    Parameters
    ----------
    path: str
        Path to directory to analyze

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots()
    trj = md.load(f'{path}/sample_res.trr', top=f'{path}/sample_2.gro')
    mxene_trj = trj.atom_slice(trj.topology.select('resname RES'))
    #area = np.max(mxene_trj.xyz[:,:,0]) * np.max(mxene_trj.xyz[:,:,1]-3)
    area = np.max(mxene_trj.xyz[:,:,0]) * np.max(mxene_trj.xyz[:,:,1])
    dim = 2
    box_range = [0, np.max(trj.xyz[:,:,2])]
    volume = area * (box_range[1] - box_range[0])

    maxs = np.array([np.max(mxene_trj.xyz[:,:,0]), np.max(mxene_trj.xyz[:,:,1]), np.max(trj.xyz[:,:,2])])
    mins = np.array([np.min(mxene_trj.xyz[:,:,0]), np.min(mxene_trj.xyz[:,:,1]), np.min(trj.xyz[:,:,2])])
    #mins = np.array([0, 0, 0])
    dims = maxs - mins

    rho, bins, res_list = calc_number_density(
                    trj=trj,
                    area=area,
                    dim=2,
                    shift=False,
                    box_range=box_range,
                    n_bins=200,
                    maxs=maxs,
                    mins=mins)
    for i in range(1, 4):
        plt.plot(bins, rho[i], label=f"{res_list[i]}")
    np.savetxt(f'{path}/number_density.txt', np.transpose(np.vstack([bins, rho[1], rho[2], rho[3]])),
               header=f"bins\t{res_list[1]}\t{res_list[2]}\t{res_list[3]}\t{dims}")
    plt.xlabel('z-position (nm)')
    plt.ylabel('number density (nm^-3)')
    plt.legend()
    plt.xlim((box_range[0], box_range[1]))
    plt.savefig(f'{path}/numden.pdf')


def atom_number_density(path):
    """Calculate number density of specific atoms for TAA in pore

    Parameters
    ----------
    path: str
        Path to directory to analyze

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots()
    trj = md.load(f'{path}/sample_res.trr', top=f'{path}/ti3c2.gro')
    mxene_trj = trj.atom_slice(trj.topology.select('resname RES'))
    area = np.max(mxene_trj.xyz[:,:,0]) * np.max(mxene_trj.xyz[:,:,1]-3)
    dim = 2
    box_range = [0, np.max(trj.xyz[:,:,2])]
    volume = area * (box_range[1] - box_range[0])

    maxs = [np.max(mxene_trj.xyz[:,:,0]), np.max(mxene_trj.xyz[:,:,1])-1.5, np.max(trj.xyz[:,:,2])]

    selections = {'tam_N': 'resname tam and name N',
            'endC': 'resname tam and name CE',
            'midC': 'resname tam and name CM',
            'branchC': 'resname tam and name CB1 CB2'}

    rho, bins, res_list = calc_atom_number_density(
                    trj=trj,
                    area=area,
                    dim=2,
                    box_range=box_range,
                    n_bins=200,
                    atom_selection=selections,
                    maxs=maxs,
                    shift=False,
                    mins=[0,1.5,0])

    for i in range(1, 4):
        rho_1 = sum(rho[i][:100]) / volume
        rho_2 = sum(rho[i][100:]) / volume
        plt.plot(bins, rho[i], label=f"{res_list[i]}")
    plt.xlabel('z-position (nm)')
    plt.ylabel('number density (nm^-3)')
    plt.legend()
    #plt.xlim((0.8, box_range[1]))
    plt.savefig(f'{path}/taa_atom_numden.pdf')
    np.savetxt(f'{path}/taa_atom_numden.txt',
               np.transpose(np.vstack([bins, rho])),
               header=f'bins\trho\t{res_list}'
    )


def calc_rg(path):
    trj = md.load(f'{path}/sample.trr', top=f'{path}/anneal.gro')
    tam = trj.atom_slice(trj.topology.select('resname tam'))
    rg_list = list()
    for i in range(40):
        single = tam.atom_slice(tam.topology.select(f'resid {i}'))
        rg = md.compute_rg(single)
        rg_list.append(rg)
    print(np.mean(rg_list))
    print(np.std(rg_list))

    return rg


def calc_bulk_density():
    """Calculate density of ions in the bulk region """
    trj = md.load('sample.trr', top='em.gro')
   
    mean = list() 
    # Loop through frames in the trajectory
    for i in range(1800, 2000):
        frame = trj[i]
        print(i)
        indices = np.intersect1d(np.where(frame.xyz[-1, :, 1] > 8),
                                 np.where(frame.xyz[-1, :, 1] < 12)
                  )
        
        sliced = frame.atom_slice(indices)
        sliced.unitcell_lengths[:,1] = 4
        density = md.density(sliced)
        mean.append(density)
        #print(np.mean(density))
    print(np.mean(mean))


def calc_pore_density():
    """Calculate density of ion in the pore """
    trj = md.load('sample_2.trr', top='nvt.gro')
    for resname in ['emim', 'tf2n']:
        mean = list()
        il = trj.atom_slice(trj.topology.select(f'resname {resname}'))
        for i in range(4800, 5000):
            frame = il[i]
        
            indices = np.intersect1d(np.where(frame.xyz[-1, :, 1] > 2),
                                     np.where(frame.xyz[-1, :, 1] < 5)
                      )
            
            sliced = il.atom_slice(indices)
            sliced.unitcell_lengths[:,1] = 3
            density = md.density(sliced)
            mean.append(density)

        print(np.mean(mean))

if __name__ == '__main__':
    #atom_number_density('12/kpl_seiji/1415_no_shift')
    #atom_number_density('16/kpl_seiji/1410_no_shift')
    number_density('12/kpl_seiji/1415_no_shift')
    number_density('16/kpl_seiji/1410_no_shift')
    #calc_rg('12/kpl_lopes/1515_density')
    #calc_rg('16/1514_density')
