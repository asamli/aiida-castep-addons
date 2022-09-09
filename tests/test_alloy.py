from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from aiida_castep_addons.workflows.alloy import add_metadata, generate_structures
from ase.build import bulk


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


def test_generate_structures():
    rutile = orm.StructureData(
        cell=[
            [4.5941, 0, 0],
            [0, 4.5941, 0],
            [0, 0, 2.9589],
        ]
    )
    rutile.append_atom(position=(0, 0, 0), symbols="Ti")
    rutile.append_atom(position=(2.29705, 2.29705, 1.47945), symbols="Ti")
    rutile.append_atom(position=(1.40441637, 1.40441637, 0), symbols="O")
    rutile.append_atom(position=(3.70146637, 0.89263363, 1.47945), symbols="O")
    rutile.append_atom(position=(0.89263363, 3.70146637, 1.47945), symbols="O")
    rutile.append_atom(position=(3.18968363, 3.18968363, 0), symbols="O")
    structures = generate_structures(
        rutile, orm.Str("Ti"), orm.Str("Zr"), orm.List(list=[1, 1, 1])
    )

    assert "xs" in structures
    assert "lens" in structures
    assert "structure_0_0" in structures
    assert "structure_1_0" in structures
    assert "structure_2_0" in structures


def test_alloy_wc(mock_castep_code):
    alloy = WorkflowFactory("castep_addons.alloy")
    bld = alloy.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.base.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 1275,
        "symmetry_generate": True,
        "fix_occupancy": True,
        "geom_force_tol": 0.01,
        "max_scf_cycles": 200,
        "geom_max_iter": 200,
    }
    bld.base.kpoints_spacing = 0.08
    bld.to_substitute = "Ti"
    bld.substituent = "Zr"
    bld.temperatures = [298, 398, 498, 598]
    rutile = orm.StructureData(
        cell=[
            [4.5941, 0, 0],
            [0, 4.5941, 0],
            [0, 0, 2.9589],
        ]
    )
    rutile.append_atom(position=(0, 0, 0), symbols="Ti")
    rutile.append_atom(position=(2.29705, 2.29705, 1.47945), symbols="Ti")
    rutile.append_atom(position=(1.40441637, 1.40441637, 0), symbols="O")
    rutile.append_atom(position=(3.70146637, 0.89263363, 1.47945), symbols="O")
    rutile.append_atom(position=(0.89263363, 3.70146637, 1.47945), symbols="O")
    rutile.append_atom(position=(3.18968363, 3.18968363, 0), symbols="O")
    bld.structure = rutile
    bld.calc.metadata.options.max_wallclock_seconds = 7200
    bld.calc.metadata.options.resources = {"num_machines": 1, "tot_num_mpiprocs": 4}
    bld.clean_workdir = True
    _, alloy_node = run_get_node(bld)

    assert alloy_node.is_finished_ok
