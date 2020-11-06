import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.ticker import MultipleLocator

def plot_angles():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    cutoff_12 = 11.13
    path_12 = "12/kpl_seiji/1415_no_shift"
    cutoff_16 = 14.0
    path_16 = "16/kpl_seiji/1410_no_shift"
    paths = (path_12, path_16)
    cutoffs = (cutoff_12, cutoff_16)
    for idx, (cutoff, path) in enumerate(zip(cutoffs, paths)):
        ax = axes[idx] 
        if idx == 0:
            str_label = 'a)'
        else:
            str_label = 'b)'

        ax.text(0.05, 0.90, str_label, transform=ax.transAxes,
            size=16, weight='bold')
        for angle_type in ['ring', 'tail', 'taa']:
            data = np.loadtxt(f"{path}/normalized_{angle_type}_angles_{cutoff}.txt")
            if angle_type == 'taa':
                label = 'alkylammonium'
            elif angle_type == 'ring':
                label = 'EMIM ring'
            elif angle_type == 'tail':
                label = 'EMIM tail'
            ax.plot(data[:,0], data[:,1], label=label)
    
        ax.set_xlabel(r"$\mathregular{Angle\ \theta, \deg}$", fontsize=16)
        ax.set_ylabel(r"$\mathregular{Normalized\ distribution}$", fontsize=16)
        ax.set_ylim((0, 0.07))
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
    
    legend_ax = ax
    handles, labels = legend_ax.get_legend_handles_labels()
    lgd = fig.legend(handles, 
            labels,
            bbox_to_anchor=(0.5, 1.08),
            fontsize=12,
            loc='upper center',
            ncol=3)
    fig.tight_layout()
    fig.savefig(f'figures/normalized_angles.pdf', bbox_inches='tight')


def plot_number_density():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    cutoff_12 = 11.13
    path_12 = "12/kpl_seiji/1415_no_shift"
    cutoff_16 = 14.0
    path_16 = "16/kpl_seiji/1410_no_shift"
    paths = (path_12, path_16)
    cutoffs = (cutoff_12, cutoff_16)
    for idx, (cutoff, path) in enumerate(zip(cutoffs, paths)):
        ax = axes[idx] 
        if idx == 0:
            str_label = 'a)'
        else:
            str_label = 'b)'

        ax.text(0.05, 0.90, str_label, transform=ax.transAxes,
            size=16, weight='bold')

        data = np.loadtxt(f'{path}/number_density.txt')
        bins = data[:,0]
        emim = data[:,1]
        tam = data[:,2]
        tf2n = data[:,3]
    
        for k, v in {'EMIM': emim, 'TAM': tam, 'TFSI': tf2n}.items():
            if k == 'TAM':
                label = 'alkylammonium'
            else:
                label = k
            ax.plot(bins, v, label=label)
    
        ax.set_xlabel('z, nm', fontsize=16)
        ax.set_ylabel(r'$ \mathrm{\rho(z)}, \mathrm{nm}^{-3}$', fontsize=16)
        if idx == 0:
            ax.set_xlim((1, 4.0))
        else:
            ax.set_xlim((1, 4.65))
        ax.set_ylim((0, 16))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))

    legend_ax = ax
    handles, labels = legend_ax.get_legend_handles_labels()
    lgd = fig.legend(handles, 
            labels,
            bbox_to_anchor=(0.5, 1.08),
            fontsize=12,
            loc='upper center',
            ncol=3)
    fig.tight_layout()
    fig.savefig(f'figures/number_density.pdf', bbox_inches='tight')

#plot_angles()
plot_number_density()
