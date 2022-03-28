from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from aiida_castep_addons.workflows.converge import (
    add_metadata,
    plot_phonons,
    seekpath_analysis,
)
from ase.build import bulk


def test_seekpath_analysis():
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    seekpath = seekpath_analysis(silicon)

    assert "kpoints" in seekpath
    assert "prim_cell" in seekpath


def test_add_metadata():
    file = orm.SinglefileData(Path("registry/test.pdf").resolve())
    new_file = add_metadata(
        file,
        orm.Str("changed_test.pdf"),
        orm.Str("test_formula"),
        orm.Str("test_uuid"),
        orm.Str("test_label"),
        orm.Str("test_description"),
    )

    assert new_file.filename == "changed_test.pdf"


def test_plot_phonons():
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    seekpath = seekpath_analysis(silicon)
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
    bld.converge_supercell = True
    results, node = run_get_node(bld)

    assert node.exit_status == 0
    assert "converged_pwcutoff" in results
    assert "converged_kspacing" in results
    assert "converged_supercell" in results
    assert "supercell_plot" in results
