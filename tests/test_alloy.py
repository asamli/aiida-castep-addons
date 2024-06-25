from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family

from aiida_castep_addons.workflows.alloy import (
    generate_bsym_structures,
    generate_sqs_structures,
)


def test_generate_bsym_structures():
    iro2 = orm.StructureData(
        cell=[
            [4.5051, 0, 0],
            [0, 4.5051, 0],
            [0, 0, 3.1586],
        ]
    )
    iro2.append_atom(position=(0, 0, 0), symbols="Ir")
    iro2.append_atom(position=(2.25255, 2.25255, 1.5793), symbols="Ir")
    iro2.append_atom(position=(1.38621927, 1.38621927, 0), symbols="O")
    iro2.append_atom(position=(3.63876927, 0.86633073, 1.5793), symbols="O")
    iro2.append_atom(position=(0.86633073, 3.63876927, 1.5793), symbols="O")
    iro2.append_atom(position=(3.1188073, 3.1188073, 0), symbols="O")
    structures = generate_bsym_structures(
        iro2, orm.Str("Ir"), orm.Str("Ti"), orm.List(list=[2, 2, 1]), orm.List()
    )


def test_generate_sqs_structures():
    iro2 = orm.StructureData(
        cell=[
            [4.5051, 0, 0],
            [0, 4.5051, 0],
            [0, 0, 3.1586],
        ]
    )
    iro2.append_atom(position=(0, 0, 0), symbols="Ir")
    iro2.append_atom(position=(2.25255, 2.25255, 1.5793), symbols="Ir")
    iro2.append_atom(position=(1.38621927, 1.38621927, 0), symbols="O")
    iro2.append_atom(position=(3.63876927, 0.86633073, 1.5793), symbols="O")
    iro2.append_atom(position=(0.86633073, 3.63876927, 1.5793), symbols="O")
    iro2.append_atom(position=(3.1188073, 3.1188073, 0), symbols="O")
    structures = generate_sqs_structures(
        iro2, orm.Str("Ir"), orm.Str("Ti"), orm.List(list=[2, 2, 1]), orm.List()
    )


def test_alloy_wc(mock_castep_code):
    alloy = WorkflowFactory("castep_addons.alloy")
    bld = alloy.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.base.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 850,
        "symmetry_generate": True,
        "geom_force_tol": 0.01,
        "elec_energy_tol": 1e-06,
        "mixing_scheme": "pulay",
        "max_scf_cycles": 1000,
        "geom_max_iter": 100,
        "write_checkpoint": "minimal",
    }
    bld.base.kpoints_spacing = 0.07
    bld.to_substitute = "Ir"
    bld.substituent = "Ti"
    bld.temperatures = [298, 398, 498, 598]
    bld.supercell_matrix = [2, 2, 1]
    iro2 = orm.StructureData(
        cell=[
            [4.5051, 0, 0],
            [0, 4.5051, 0],
            [0, 0, 3.1586],
        ]
    )
    iro2.append_atom(position=(0, 0, 0), symbols="Ir")
    iro2.append_atom(position=(2.25255, 2.25255, 1.5793), symbols="Ir")
    iro2.append_atom(position=(1.38621927, 1.38621927, 0), symbols="O")
    iro2.append_atom(position=(3.63876927, 0.86633073, 1.5793), symbols="O")
    iro2.append_atom(position=(0.86633073, 3.63876927, 1.5793), symbols="O")
    iro2.append_atom(position=(3.1188073, 3.1188073, 0), symbols="O")
    bld.structure = iro2
    bld.calc.metadata.options.max_wallclock_seconds = int(3600 * 48)
    bld.calc.metadata.options.resources = {"num_machines": 1, "tot_num_mpiprocs": 4}
    bld.clean_workdir = True
    _, enum_node = run_get_node(bld)

    bld.use_ce = True
    bld.substituent = "Ru"
    bld.ce_file = orm.SinglefileData(Path("registry/pgm_rutile_alloys.ce").resolve())
    _, ce_node = run_get_node(bld)
    assert ce_node.is_finished_ok

    assert enum_node.is_finished_ok
