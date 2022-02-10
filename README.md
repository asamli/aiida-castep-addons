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
* `castep_addons.converge`: Workflow to converge the plane-wave energy cutoff, k-point mesh density and/or the phonon supercell size
* `castep_addons.band_plot`: Workflow to calculate the density of states and band structures which are then plotted using `sumo`
* `castep_addons.phonon`: Workflow to calculate the phonon band structure (plotted using `sumo`) and IR and Raman spectra (plotted using `matplotlib.pyplot`)

Parsers:
* `castep_addons.phonon`: Parser for CASTEP .phonon output files to extract the structure, vibrational spectrum data, q-points, phonon frequencies and eigenvectors
