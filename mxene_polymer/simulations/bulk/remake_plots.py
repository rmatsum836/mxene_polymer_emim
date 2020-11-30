import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.ticker import MultipleLocator

def color_dict(label):
    """['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
 
       ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66', '#77BEDB']
       ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']
        'EMI': '#D65F5F',
        'AA': '#6ACC65',
        'TFSI': '#4878CF',
    """
    colors = {
        'EMI ring': '#956CB4',
        'AA tail': '#6ACC64',
        'EMI tail': '#D5BB67',
        'EMI': '#D65F5F',
        'AA': '#6ACC64',
        'TFSI': '#4878D0',
        'N': 'black',
        #'N': '#797979',
        #'Terminal C': '#D65F5F',
        'Branch C': '#4878d0',
        'Middle C': '#797979',
        'Terminal C': '#EE854A',
    }

    return colors[label]

def plot_all():
    plt.style.use('seaborn-muted')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,12))
    cutoff_12 = 11.13
    path_12 = "12/kpl_seiji/568_bulk_ions/longer"
    cutoff_16 = 14.0
    path_16 = "16/kpl_seiji/1410_updated_params/longer"
    paths = (path_12, path_16)
    cutoffs = (cutoff_12, cutoff_16)
    for idx, (cutoff, path) in enumerate(zip(cutoffs, paths)):
        ax = axes[0, idx] 
        if idx == 0:
            str_label = '(a)'
        else:
            str_label = '(b)'

        ax.text(0.05, 0.90, str_label, transform=ax.transAxes,
            size=20, weight='bold')
        for angle_type in ['ring', 'tail', 'taa']:
            data = np.loadtxt(f"{path}/normalized_{angle_type}_angles_{cutoff}.txt")
            if angle_type == 'taa':
                label = 'AA tail'
            elif angle_type == 'ring':
                label = 'EMI ring'
            elif angle_type == 'tail':
                label = 'EMI tail'
            ax.plot(data[:,0], data[:,1], label=label, color=color_dict(label), linewidth=3)
    
        ax.set_xlabel(r"$\mathregular{Angle\ \theta, \deg}$", fontsize=20)
        ax.set_ylabel(r"$\mathregular{Normalized\ distribution}$", fontsize=20)
        ax.tick_params(labelsize=18)
        ax.set_ylim((0, 0.1))
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
    
        legend_ax = ax
        handles, labels = legend_ax.get_legend_handles_labels()
        lgd = ax.legend(handles, 
                labels,
                fontsize=14,
                frameon=False,
                ncol=3)

    # Plot number density
    for idx, (cutoff, path) in enumerate(zip(cutoffs, paths)):
        ax = axes[1, idx] 
        if idx == 0:
            str_label = '(c)'
        else:
            str_label = '(d)'

        ax.text(0.05, 0.90, str_label, transform=ax.transAxes,
            size=20, weight='bold')

        data = np.loadtxt(f'{path}/number_density.txt')
        bins = data[:,0]
        emim = data[:,1]
        tam = data[:,2]
        tf2n = data[:,3]
    
        for k, v in {'EMI': emim, 'TAM': tam, 'TFSI': tf2n}.items():
            if k == 'TAM':
                label = 'AA'
            else:
                label = k
            ax.plot(bins, v, label=label, color=color_dict(label), linewidth=3)
    
        ax.set_xlabel('z, nm', fontsize=20)
        ax.set_ylabel(r'$ \mathrm{\rho(z)}, \mathrm{nm}^{-3}$', fontsize=20)
        if idx == 0:
            ax.set_xlim((1, 4.0))
        else:
            ax.set_xlim((1, 4.65))
        ax.set_ylim((0, 17))
        ax.tick_params(labelsize=18)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))

        legend_ax = ax
        handles, labels = legend_ax.get_legend_handles_labels()
        lgd = ax.legend(handles, 
                labels,
                fontsize=14,
                frameon=False,
                ncol=3)

    for idx, (cutoff, path) in enumerate(zip(cutoffs, paths)):
        ax = axes[2, idx] 
        if idx == 0:
            str_label = '(e)'
        else:
            str_label = '(f)'

        ax.text(0.05, 0.90, str_label, transform=ax.transAxes,
            size=20, weight='bold')

        data = np.loadtxt(f'{path}/taa_atom_numden.txt')
        bins = data[:,0]
        taa_n = data[:,1]
        emim_end = data[:,2]
        emim_mid = data[:,3]
        emim_branch = data[:,4]
    
        for k, v in {'N': taa_n, 'Terminal C': emim_end, 'Branch C': emim_branch}.items():
            ax.plot(bins, v, label=k, color=color_dict(k), linewidth=3)
    
        ax.set_xlabel('z, nm', fontsize=20)
        ax.set_ylabel(r'$ \mathrm{\rho(z)}, \mathrm{nm}^{-3}$', fontsize=20)
        if idx == 0:
            ax.set_xlim((1, 4.0))
        else:
            ax.set_xlim((1, 4.65))
        ax.set_ylim((0, 4))
        ax.tick_params(labelsize=18)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))

        legend_ax = ax
        handles, labels = legend_ax.get_legend_handles_labels()
        #all_handles = handles + handles_tam
        #all_labels = labels + labels_tam

        #lgd = ax.legend([all_handles[0], all_handles[5], all_handles[1], all_handles[3], all_handles[2], all_handles[4]],
        #        [all_labels[0], all_labels[5], all_labels[1], all_labels[3], all_labels[2], all_labels[4]],
        #        bbox_to_anchor=(0.5, 1.06),
        #        fontsize=18,
        #        loc='upper center',
        #        ncol=3)
        lgd = ax.legend(handles, 
                labels,
                fontsize=14,
                frameon=False,
                ncol=3)
    fig.tight_layout()
    #fig.savefig(f'figures/taa_number_density.pdf', bbox_inches='tight')
    fig.savefig(f'figures/overall.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'figures/overall.png', bbox_inches='tight', dpi=300)


#plot_angles()
#plot_number_density()
#plot_tam_number_density()
plot_all()
