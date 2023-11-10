import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family

from aiida_castep_addons.workflows.competing_phases import generate_competing_phases


def test_generate_competing_phases():
    intrinsic_entries = generate_competing_phases(
        orm.Str("TiO2"),
        orm.List(),
        orm.Dict(dict={"e_above_hull": 0}),
        orm.Dict(dict={"e_above_hull": 0}),
    )
    extrinsic_entries = generate_competing_phases(
        orm.Str("TiO2"),
        orm.List(list=["Zr"]),
        orm.Dict(dict={"e_above_hull": 0}),
        orm.Dict(dict={"e_above_hull": 0}),
    )


def test_competing_phases_wc(mock_castep_code):
    competing_phases = WorkflowFactory("castep_addons.competing_phases")
    bld = competing_phases.get_builder()
    bld.calc.code = mock_castep_code
    bld.converge.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.base.pseudos_family = "C19"
    bld.converge.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 850,
        "symmetry_generate": True,
        "geom_force_tol": 0.01,
        "max_scf_cycles": 200,
        "geom_max_iter": 200,
    }
    bld.converge.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 850,
        "symmetry_generate": True,
        "max_scf_cycles": 200,
    }
    bld.extrinsic_species = ["Zr"]
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
    bld.converge.calc.structure = rutile
    bld.calc.metadata.options.max_wallclock_seconds = 3600
    bld.calc.metadata.options.resources = {"num_machines": 1, "tot_num_mpiprocs": 4}
    bld.converge.calc_options = {
        "max_wallclock_seconds": 3600,
        "resources": {"num_machines": 1, "tot_num_mpiprocs": 4},
    }
    bld.clean_workdir = True
    _, competing_phases_node = run_get_node(bld)

    assert competing_phases_node.is_finished_ok
