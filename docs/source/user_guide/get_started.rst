===============
Getting started
===============

This package contains additional workflows and other addons for the ``aiida-castep`` plugin. 
The usage of these workflows is identical to those in ``aiida-castep`` and they can be accessed
through ``WorkflowFactory`` with their entry points (in the format ``castep_addons.xxxxxx``).

Installation
++++++++++++

Use the following commands to install the plugin::

    git clone https://github.com/asamli/aiida-castep-addons .
    cd aiida-castep-addons
    pip install -e .  # also installs all dependencies
    #pip install -e .[pre-commit,testing] # install extras for more features
    verdi quicksetup  # set up a new profile

Then use ``verdi computer setup`` and ``verdi code setup`` to add computer and code nodes
to the AiiDA database.

Usage
+++++

See the tutorial page for an example calculation using a workflow.
