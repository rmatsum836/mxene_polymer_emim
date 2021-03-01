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
    trj = md.load(f'{path}/sample_res.trr', top=f'{path}/nvt.gro')
    mxene_trj = trj.atom_slice(trj.topology.select('resname RES'))
    # Only consider center of MXene pore 1 nm away from each entrance
    area = trj.unitcell_lengths[0][0] * (np.max(mxene_trj.xyz[:,:,1])-2)
    dim = 2
    box_range = [0, trj.unitcell_lengths[0][2]]
    volume = area * (box_range[1] - box_range[0])

    maxs = np.array([np.max(mxene_trj.xyz[:,:,0]), np.max(mxene_trj.xyz[:,:,1])-1, np.max(trj.xyz[:,:,2])])
    mins = np.array([np.min(mxene_trj.xyz[:,:,0]), np.min(mxene_trj.xyz[:,:,1])+1, np.min(trj.xyz[:,:,2])])
    dims = maxs - mins
    n_bins = np.arange(box_range[0], box_range[1], 0.0203)

    rho, bins, res_list = calc_number_density(
                    trj=trj,
                    area=area,
                    dim=2,
                    shift=False,
                    box_range=box_range,
                    n_bins=n_bins,
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
    area = trj.unitcell_lengths[0][0] * (np.max(mxene_trj.xyz[:,:,1])-2)
    dim = 2
    box_range = [0, trj.unitcell_lengths[0][2]]
    volume = area * (box_range[1] - box_range[0])
    n_bins = np.arange(box_range[0], box_range[1], 0.0203)

    maxs = np.array([np.max(mxene_trj.xyz[:,:,0]), np.max(mxene_trj.xyz[:,:,1])-1, np.max(trj.xyz[:,:,2])])
    mins = np.array([np.min(mxene_trj.xyz[:,:,0]), np.min(mxene_trj.xyz[:,:,1])+1, np.min(trj.xyz[:,:,2])])

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
                    mins=mins,
                    shift=False,
                    )

    for i in range(1, 4):
        rho_1 = sum(rho[i][:100]) / volume
        rho_2 = sum(rho[i][100:]) / volume
        plt.plot(bins, rho[i], label=f"{res_list[i]}")
    plt.xlabel('z-position (nm)')
    plt.ylabel('number density (nm^-3)')
    plt.legend()
    plt.savefig(f'{path}/taa_atom_numden.pdf')
    np.savetxt(f'{path}/taa_atom_numden.txt',
               np.transpose(np.vstack([bins, rho])),
               header=f'bins\trho\t{res_list}'
    )


#def calc_pore_density_12(path):
#    """Calculate density of EMIM-TFSI in pore for TAM-12 when considering 
#       original definition of d-spacing.
#
#    Parameters
#    ----------
#    path: str
#        Path to directory to analyze
#
#    Returns
#    -------
#    None 
#    """
#    fig, ax = plt.subplots()
#    trj = md.load(f'{path}/com.trr', top=f'{path}/com.gro')
#    for resname in ['emim', 'tf2n']:
#        mean = list()
#        il = trj.atom_slice(trj.topology.select(f'resname {resname}'))
#        for i in range(0, 4000):
#            frame = il[i]
#        
#            pore1 = np.intersect1d(
#                        np.intersect1d(np.where(frame.xyz[-1, :, 1] > 1),
#                                     np.where(frame.xyz[-1, :, 1] < (5.467-1))
#                        ),
#                        np.intersect1d(np.where(frame.xyz[-1, :, 2] > 0.937),
#                                     np.where(frame.xyz[-1, :, 2] < 2.051)
#                        ),
#            )
#            pore2 = np.intersect1d(
#                        np.intersect1d(np.where(frame.xyz[-1, :, 1] > 1),
#                                     np.where(frame.xyz[-1, :, 1] < (5.467-1))
#                        ),
#                        np.intersect1d(np.where(frame.xyz[-1, :, 2] > 2.947),
#                                     np.where(frame.xyz[-1, :, 2] < 4.06017)
#                        ),
#            )
#
#            pore_avg = list()
#            for pore in (pore1, pore2):
#                sliced = frame.atom_slice(pore)
#                sliced.unitcell_lengths[:,1] = (5.467-1) - 1
#                sliced.unitcell_lengths[:,2] = 1.113
#                masses = list()
#                for i in sliced.topology.atoms:
#                    if i.name == 'emim':
#                        masses.append(111)
#                    if i.name == 'tf2n':
#                        masses.append(280)
#
#                density = md.density(sliced, masses=masses)
#                pore_avg.append(density)
#
#            avg_density = np.mean(pore_avg)
#            mean.append(avg_density)
#
#        if resname == 'emim':
#            label = 'EMI'
#        elif resname == 'tf2n':
#            label = 'TFSI'
#        plt.plot(range(0,4000), mean, label=label)
#        print(np.mean(mean))
#    plt.xlabel("MD Frame")
#    plt.ylabel("density (kg/m^3)")
#    plt.legend()
#    plt.savefig(f'{path}/sample_densities.pdf', dpi=400)
#    plt.savefig(f'{path}/sample_densities.png', dpi=400)
def calc_pore_density_12(path):
    """Calculate density of EMIM-TFSI in pore for TAM-12 when considering 
       original definition of d-spacing with 0.281 nm of spacing.

    Parameters
    ----------
    path: str
        Path to directory to analyze

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots()
    trj = md.load(f'{path}/com.trr', top=f'{path}/com.gro')
    for resname in ['emim', 'tf2n']:
        mean = list()
        il = trj.atom_slice(trj.topology.select(f'resname {resname}'))
        for i in range(0, 4000):
            frame = il[i]
        
            pore1 = np.intersect1d(
                        np.intersect1d(np.where(frame.xyz[-1, :, 1] > 1),
                                     np.where(frame.xyz[-1, :, 1] < (5.467-1))
                        ),
                        np.intersect1d(np.where(frame.xyz[-1, :, 2] > 0.937),
                                     np.where(frame.xyz[-1, :, 2] < 2.332)
                        ),
            )
            pore2 = np.intersect1d(
                        np.intersect1d(np.where(frame.xyz[-1, :, 1] > 1),
                                     np.where(frame.xyz[-1, :, 1] < (5.467-1))
                        ),
                        np.intersect1d(np.where(frame.xyz[-1, :, 2] > 3.228),
                                     np.where(frame.xyz[-1, :, 2] < 4.6222)
                        ),
            )

            pore_avg = list()
            for pore in (pore1, pore2):
                sliced = frame.atom_slice(pore)
                sliced.unitcell_lengths[:,1] = (5.467-1) - 1
                sliced.unitcell_lengths[:,2] = 1.395
                masses = list()
                for i in sliced.topology.atoms:
                    if i.name == 'emim':
                        masses.append(111)
                    if i.name == 'tf2n':
                        masses.append(280)

                density = md.density(sliced, masses=masses)
                pore_avg.append(density)

            avg_density = np.mean(pore_avg)
            mean.append(avg_density)

        if resname == 'emim':
            label = 'EMI'
        elif resname == 'tf2n':
            label = 'TFSI'
        plt.plot(range(0,4000), mean, label=label)
        print(np.mean(mean))
    plt.xlabel("MD Frame")
    plt.ylabel("density (kg/m^3)")
    plt.legend()
    plt.savefig(f'{path}/number_densities.pdf', dpi=400)
    plt.savefig(f'{path}/number_densities.png', dpi=400)


def calc_pore_density_16(path):
    """Calculate density of EMIM-TFSI in pore for TAM-16

    Parameters
    ----------
    path: str
        Path to directory to analyze

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots()
    trj = md.load(f'{path}/com.trr', top=f'{path}/com.gro')
    for resname in ['emim', 'tf2n']:
        mean = list()
        il = trj.atom_slice(trj.topology.select(f'resname {resname}'))
        for i in range(0, 4000):
            frame = il[i]
        
            pore1 = np.intersect1d(
                        np.intersect1d(np.where(frame.xyz[-1, :, 1] > 1),
                                     np.where(frame.xyz[-1, :, 1] < (5.467-1))
                        ),
                        np.intersect1d(np.where(frame.xyz[-1, :, 2] > 0.937),
                                     np.where(frame.xyz[-1, :, 2] < 2.366)
                        ),
            )
            pore2 = np.intersect1d(
                        np.intersect1d(np.where(frame.xyz[-1, :, 1] > 1),
                                     np.where(frame.xyz[-1, :, 1] < (5.467-1))
                        ),
                        np.intersect1d(np.where(frame.xyz[-1, :, 2] > 3.262),
                                     np.where(frame.xyz[-1, :, 2] < 4.69017)
                        ),
            )

            pore_avg = list()
            for pore in (pore1, pore2):
                sliced = frame.atom_slice(pore)
                sliced.unitcell_lengths[:,1] = (5.467-1) - 1
                sliced.unitcell_lengths[:,2] = 1.428
                masses = list()
                for i in sliced.topology.atoms:
                    if i.name == 'emim':
                        masses.append(111)
                    if i.name == 'tf2n':
                        masses.append(280)

                density = md.density(sliced, masses=masses)
                pore_avg.append(density)

            avg_density = np.mean(pore_avg)
            mean.append(avg_density)

        if resname == 'emim':
            label = 'EMI'
        elif resname == 'tf2n':
            label = 'TFSI'
        plt.plot(range(0,4000), mean, label=label)
        print(np.mean(mean))
    plt.xlabel("MD Frame")
    plt.ylabel("density (kg/m^3)")
    plt.legend()
    plt.savefig(f'{path}/sample_densities.pdf', dpi=400)
    plt.savefig(f'{path}/sample_densities.png', dpi=400)


def calc_bulk_com_density(path):
    """Calculate density of EMIM-TFSI in bulk

    Parameters
    ----------
    path: str
        Path to directory to analyze

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots()
    trj = md.load(f'{path}/com.trr', top=f'{path}/com.gro')
    ions = trj.atom_slice(trj.topology.select("resname emim tf2n"))
   
    mean = list() 
    # Loop through frames in the trajectory
    for i in range(0, 4000):
        frame = ions[i]
        indices = np.intersect1d(np.where(frame.xyz[-1, :, 1] >= 8.5),
                                 np.where(frame.xyz[-1, :, 1] <= 11.5)
                  )
        
        sliced = frame.atom_slice(indices)
        sliced.unitcell_lengths[:,1] = 3
        masses = list()
        for i in sliced.topology.atoms:
            if i.name == 'emim':
                masses.append(111)
            if i.name == 'tf2n':
                masses.append(280)

        masses = np.array(masses)
        density = md.density(sliced, masses = masses)
        mean.append(density)

    plt.plot(range(0, 4000), mean)
    plt.xlabel("MD Frame")
    plt.ylabel("density (kg/m^3)")
    plt.savefig(f"{path}/bulk_density.pdf", dpi=400) 
    plt.savefig(f"{path}/bulk_density.png", dpi=400) 

    print(np.mean(mean))



if __name__ == '__main__':
    #atom_number_density('12/kpl_seiji/568_bulk_ions/longer')
    #atom_number_density('16/kpl_seiji/1410_updated_params/longer')
    #number_density('12/kpl_seiji/568_bulk_ions/longer')
    #number_density('16/kpl_seiji/1410_updated_params/longer')
    number_density('12/kpl_seiji/1.31nm/625_updated_4')
    #number_density('16/kpl_seiji/1.626nm/737_updated')
    atom_number_density('12/kpl_seiji/1.31nm/625_updated_4')
    #atom_number_density('16/kpl_seiji/1.626nm/737_updated')
    #calc_pore_density_12('12/kpl_seiji/568_bulk_ions/longer')
    #calc_pore_density_16('16/kpl_seiji/1410_updated_params/longer')
    #calc_bulk_com_density('12/kpl_seiji/568_bulk_ions/longer')
    #calc_bulk_com_density('16/kpl_seiji/1410_updated_params/longer')
