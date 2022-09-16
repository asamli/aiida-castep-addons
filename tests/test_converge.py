from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from aiida_castep_addons.workflows.converge import plot_phonons, seekpath_analysis
from ase.build import bulk


def test_plot_phonons():
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    seekpath = seekpath_analysis(silicon, orm.Dict(dict={}))
    kpoints = seekpath["kpoints"]
    files = []
    matrices = [
        ["2 0 0", "0 2 0", "0 0 2"],
        ["3 0 0", "0 3 0", "0 0 3"],
        ["4 0 0", "0 4 0", "0 0 4"],
    ]
    folders = ["calc-016", "calc-017", "calc-018"]
    for folder in folders:
        with open(f"registry/Si_converge/{folder}/out/aiida.phonon") as dot_phonon:
            phonon_data = " ".join(dot_phonon.readlines())
            files.append(phonon_data)
    supercell_convergence_plot = plot_phonons(
        orm.List(list=files), kpoints, orm.List(list=matrices), orm.Str("Si2_pbesol")
    )


def test_converge_wc(mock_castep_code):
    conv = WorkflowFactory("castep_addons.converge")
    bld = conv.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 300,
        "symmetry_generate": True,
    }
    bld.kpoints_spacing = 0.1
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    bld.calc.structure = silicon
    bld.calc_options = {
        "max_wallclock_seconds": 3600,
        "resources": {"num_machines": 1, "tot_num_mpiprocs": 4},
    }
    bld.clean_workdir = True
    bld.converge_settings = {"converge_supercell": True}
    results, node = run_get_node(bld)

    assert node.is_finished_ok
    assert "converged_pwcutoff" in results
    assert "converged_kspacing" in results
    assert "converged_supercell" in results
    assert "supercell_plot" in results
