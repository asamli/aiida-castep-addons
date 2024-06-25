from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from ase.build import bulk

from aiida_castep_addons.workflows.core_loss import plot_core_loss


def test_plot_core_loss():
    folder = orm.FolderData(tree=Path("registry/Si_core_loss/out").resolve())
    results = plot_core_loss(
        folder,
        orm.Str("test_uuid"),
        orm.Str("test_label"),
        orm.Str("test_description"),
        orm.Str("test_prefix"),
        orm.ArrayData(),
    )

    assert "optados_data" in results
    assert "core_loss_spectrum" in results


def test_core_loss_wc(mock_castep_code):
    phonon = WorkflowFactory("castep_addons.core_loss")
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
    bld.metadata.label = "Si_pbesol_core_loss"
    bld.metadata.description = "Core loss spectrum of Si with the PBEsol functional"
    bld.clean_workdir = True
    _, node = run_get_node(bld)

    assert node.is_finished_ok
