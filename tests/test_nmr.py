from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from aiida_castep_addons.workflows.nmr import nmr_analysis
from ase.build import bulk


def test_nmr_analysis():
    folder = orm.FolderData(tree=Path("registry/Si_NMR/out").resolve())
    results = nmr_analysis(folder)


def test_nmr_wc(mock_castep_code):
    phonon = WorkflowFactory("castep_addons.nmr")
    bld = phonon.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 350,
        "symmetry_generate": True,
    }
    bld.kpoints_spacing = 0.04
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    bld.calc.structure = silicon
    bld.calc_options = {
        "max_wallclock_seconds": 3600,
        "resources": {"num_machines": 1, "tot_num_mpiprocs": 4},
    }
    bld.clean_workdir = True
    _, node = run_get_node(bld)

    assert node.is_finished_ok
