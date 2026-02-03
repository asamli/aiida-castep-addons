import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from aiida_castep_addons.workflows.defects import generate_defects
from pathlib import Path


def test_generate_defects():
    ceo2 = orm.StructureData(
        cell=[
            [5.4178, 0, 0],
            [0, 5.4178, 0],
            [0, 0, 5.4178],
        ]
    )
    ceo2.append_atom(position=(0, 0, 0), symbols="Ce")
    ceo2.append_atom(position=(0, 2.7089, 2.7089), symbols="Ce")
    ceo2.append_atom(position=(2.7089, 0, 2.7089), symbols="Ce")
    ceo2.append_atom(position=(2.7089, 2.7089, 0), symbols="Ce")
    ceo2.append_atom(position=(1.35445, 1.35445, 4.06355), symbols="O")
    ceo2.append_atom(position=(1.35445, 4.06355, 1.35445), symbols="O")
    ceo2.append_atom(position=(4.06355, 1.35445, 1.35445), symbols="O")
    ceo2.append_atom(position=(4.06355, 4.06355, 4.06355), symbols="O")
    ceo2.append_atom(position=(4.06355, 4.06355, 1.35445), symbols="O")
    ceo2.append_atom(position=(4.06355, 1.35445, 4.06355), symbols="O")
    ceo2.append_atom(position=(1.35445, 4.06355, 4.06355), symbols="O")
    ceo2.append_atom(position=(1.35445, 1.35445, 1.35445), symbols="O")
    structures = generate_defects(
        ceo2, orm.Dict(dict={"extrinsic": ["Zr"]}), orm.Str("CeO2_Zr_pbesol")
    )


def test_defects_wc(mock_castep_code):
    defects = WorkflowFactory("castep_addons.defects")
    bld = defects.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.base.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 850,
        "symmetry_generate": True,
        "fix_occupancy": True,
        "geom_force_tol": 0.01,
        "max_scf_cycles": 200,
        "geom_max_iter": 200,
    }
    bld.base.kpoints_spacing = 0.08
    ceo2 = orm.StructureData(
        cell=[
            [5.4178, 0, 0],
            [0, 5.4178, 0],
            [0, 0, 5.4178],
        ]
    )
    ceo2.append_atom(position=(0, 0, 0), symbols="Ce")
    ceo2.append_atom(position=(0, 2.7089, 2.7089), symbols="Ce")
    ceo2.append_atom(position=(2.7089, 0, 2.7089), symbols="Ce")
    ceo2.append_atom(position=(2.7089, 2.7089, 0), symbols="Ce")
    ceo2.append_atom(position=(1.35445, 1.35445, 4.06355), symbols="O")
    ceo2.append_atom(position=(1.35445, 4.06355, 1.35445), symbols="O")
    ceo2.append_atom(position=(4.06355, 1.35445, 1.35445), symbols="O")
    ceo2.append_atom(position=(4.06355, 4.06355, 4.06355), symbols="O")
    ceo2.append_atom(position=(4.06355, 4.06355, 1.35445), symbols="O")
    ceo2.append_atom(position=(4.06355, 1.35445, 4.06355), symbols="O")
    ceo2.append_atom(position=(1.35445, 4.06355, 4.06355), symbols="O")
    ceo2.append_atom(position=(1.35445, 1.35445, 1.35445), symbols="O")
    bld.structure = ceo2
    bld.doped_settings = {"extrinsic": ["Zr"]}
    bld.defect_metadata = {
        "dielectric": 0,
        "cbm": 1.86,
        "vbm": 0,
        "gap": 1.86,
        "num_elec_cbm": 0,
        "num_hole_vbm": 0,
        "bandfilling_meta": {
            "num_elec_cbm": 0,
            "num_hole_vbm": 0,
            "potalign": 0,
            "bandfilling_correction": 0,
        },
        "is_compatible": True,
        "phasediagram_meta": {
            "vbm": 0,
            "gap": 1.86,
        },
    }
    bld.chempots = orm.SinglefileData(
        Path("registry/Ce4O8_pbesol_chempots.json").resolve()
    )
    bld.chempot_limit = "Ce-rich"
    bld.calc.metadata.options.max_wallclock_seconds = int(3600 * 24)
    bld.calc.metadata.options.resources = {"num_machines": 1, "tot_num_mpiprocs": 4}
    bld.clean_workdir = True
    _, defects_node = run_get_node(bld)

    assert defects_node.is_finished_ok
