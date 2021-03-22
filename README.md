# MXENE Polymer and EMIM-TFSI Simulations

A collection of scripts and files to simulate MXenes filled with
tetraalkylammonium polymers and EMIM cations.  The mBuild recipe to build the MXene lattice compounds is currently private.  However the GROMACS input files are contained to run the simulations

## Requirements
- mBuild
- Foyer
- MXenes GitHub Repository (Currently Private)
- ILForcefields
- MDTraj
- MDAnalysis

## Workflow for simulations
All simulations were run with GROMACS 2020.  The final systems are contained within the `bulk` subdirectory in the `simulations` directory.  The final system containing the 12-carbon tetraalkylammonium is named `625_updated_4` and the final system containing the 16-carbon tetraalkylammonium is named `737_updated`.  To begin running the simulations, the index files need to first be created by running:
``` 
gmx make_ndx -f ti3c2.gro -o index.ndx
```

From there, create a group for the ions by typing `2 | 3 | 4` and `q`.

Once the index file is created, a short steepest-descent energy minimization will be run:

```
gmx grompp -f em.mdp -c ti3c2.gro -p ti3c2.top -n index.ndx -o em.tpr
gmx mdrun -deffnm em
```

Next we will run an annealing/equilibration run in the canoncial (NVT) ensemble:

```
gmx grompp -f nvt_bussi.mdp -c em.gro -p ti3c2.top -n index.ndx -o nvt.tpr
gmx mdrun -deffnm nvt
```

Finally, the sampling simulation is run:

```
gmx grompp -f sample.mdp -c nvt.gro -p ti3c2.top -n index.ndx -o sample.tpr
gmx mdrun -deffnm sample
```

