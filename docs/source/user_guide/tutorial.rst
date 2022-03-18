========
Tutorial
========

This is a simple tutorial for the convergence workflow (``castep_addons.converge``). This tutorial can either be done in an interactive Python 
shell with ``verdi shell`` or all the code can be saved in a Python file (e.g. tutorial.py) and executed using ``verdi run tutorial.py``.

Si convergence testing
+++++++++++++++++++++++

Before starting make sure you follow the instructions in the 'Get Started' page to set up an AiiDA profile, computer and code.
Ensure the daemon is running and if not, start it with ``verdi daemon start`` because it will be needed to submit the workflow.

The first step is to import some necessary modules using::

    from ase.build import bulk
    from aiida.engine import submit

You can then import the workflow and get its input builder with::

    conv = WorkflowFactory("castep_addons.converge")
    bld = conv.get_builder()

The inputs for the builder can be seen using TAB autocompletion (simply type ``bld.`` and press TAB to see the options).
A good place to start is inputting the code as follows::

    bld.calc.code = Code.get_from_string("code@computer")

replacing ``code@computer`` with your code.

If you don't have any pseudopotentials installed yet, you can use these commands to install the C19 pseudopotential family::

    from aiida_castep.data.otfg import upload_otfg_family
    upload_otfg_family(["C19"], "C19", "C19 potential library")

This can then be added into the workflow inputs using::

    bld.pseudos_family = "C19"

The parameters for the .param and .cell files can be entered using a flat format like with ``aiida-castep`` workflows as follows::

    bld.calc.parameters = {
        "xc_functional": "lda",
        "cut_off_energy": 300,
        "symmetry_generate": True,
    }

You can choose to either set the k-point mesh using the ``kpoints_spacing`` input port::

    bld.kpoints_spacing = 0.1

or as an AiiDA ``KpointsData`` node::

    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh((4, 4, 4))
    bld.calc.kpoints = kpoints

For the structure you can simply use ``ase.bulk`` to create the primitive unit cell and provide it as StructureData::

    StructureData = DataFactory("structure")
    silicon = StructureData(ase=bulk("Si", "diamond", 5.43))
    bld.calc.structure = silicon

The final step is to set the computational resources for the calculations. If your computer uses a ``direct`` scheduler you can use::

    bld.calc_options = {
    "max_wallclock_seconds": 3600,
    "resources": {"num_machines": 1, "tot_num_mpiprocs": 4},
    }

For computers with different schedulers please refer to this `official AiiDA page. <https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/schedulers.html#topics-schedulers-job-resources-node>`_

Optionally you can make the workflow clean the remote directory with::

    bld.clean_workdir = True

Now you can submit this builder to the daemon::

    submit(bld)

and exit the interactive shell with ``exit`` if you used ``verdi shell``.

To monitor the status of the workflow you can use ``verdi process list`` for all active jobs or ``verdi process list -a`` for all jobs including completed and failed ones.
You can see the log for the workflow using ``verdi process report NODE`` and all information including input and output nodes using ``verdi node show NODE``.
Using ``verdi node graph generate NODE`` on a workflow node will generate a provenance graph of the workflow.
