[![Build Status](https://github.com/asamli/aiida-castep-addons/workflows/ci/badge.svg?branch=master)](https://github.com/asamli/aiida-castep-addons/actions)
[![Docs status](https://readthedocs.org/projects/aiida-castep-addons/badge)](http://aiida-castep-addons.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# aiida-castep-addons

Useful addons for the aiida-castep plugin (primarily workflows)

How to install
-------------
In a conda environment type:
```
git clone https://github.com/asamli/test-repo.git
cd test-repo
pip install -e .
```

Usage
-----
Create an AiiDA profile using `verdi quicksetup` followed by a computer and code using `verdi computer setup` and `verdi code setup` respectively.
The workflows can be accessed using `WorkflowFactory('entry point')` (see features for the workflow entry points).

Features
--------
Workflows:
* `castep_addons.converge`: Workflow to converge the plane-wave energy cutoff, k-point mesh density and/or the phonon supercell size.
* `castep_addons.band_plot`: Workflow to calculate the density of states and band structures which are then plotted using `Sumo`. `Galore` is also integrated to plot UPS, XPS and HAXPES spectra.
* `castep_addons.phonon`: Workflow to calculate the phonon band structure (plotted using `Sumo`), IR and Raman spectra, and thermodynamics (plotted using `matplotlib.pyplot`). The spectra are broadened with `Galore`.
* `castep_addons.magnetic`: Workflow to enumerate magnetic orderings for a structure using `Pymatgen`, relax each structure and analyse the results. Note: The [EnumLib](https://github.com/msg-byu/enumlib) package must be installed in order to use this workflow.
* `castep_addons.core_loss`: Workflow to do spectral core loss calculations and plot EELS/XANES spectra using `OptaDOS`.
* `castep_addons.alloy`: Workflow to generate alloy structures with `Bsym` calculate thermodynamic properties of mixing after relaxing each structure.

Parsers:
* `castep_addons.phonon`: Parser for CASTEP .phonon output files to extract the structure, vibrational spectrum data, q-points, phonon frequencies and eigenvectors.
