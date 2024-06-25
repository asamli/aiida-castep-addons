import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from ase.build import bulk

from aiida_castep_addons.workflows.magnetic import enumerate_spins


def test_enumerate_spins():
    structure = orm.StructureData(ase=bulk("Fe", "bcc", 2.86))
    enum_structures = enumerate_spins(structure, orm.Dict(dict={}))


def test_magnetic_wc(mock_castep_code):
    magnetic = WorkflowFactory("castep_addons.magnetic")
    bld = magnetic.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.base.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 675,
        "max_scf_cycles": 200,
        "symmetry_generate": True,
    }
    bld.base.kpoints_spacing = 0.04
    structure = orm.StructureData(ase=bulk("Fe", "bcc", 2.86))
    bld.structure = structure
    bld.calc.metadata.options.max_wallclock_seconds = 3600
    bld.calc.metadata.options.resources = {"num_machines": 1, "tot_num_mpiprocs": 4}
    bld.clean_workdir = True
    _, node = run_get_node(bld)

    assert node.is_finished_ok
