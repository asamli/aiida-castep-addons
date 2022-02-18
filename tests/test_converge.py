from aiida.engine import run_get_node
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from ase.build import bulk

StructureData = DataFactory("structure")


def test_seekpath_analysis():
    silicon = StructureData(ase=bulk("Si", "diamond", 5.43))
    seekpath = seekpath_analysis(silicon)

    assert "kpoints" in seekpath
    assert "prim_cell" in seekpath


def test_converge_wc(mock_castep_code):
    conv = WorkflowFactory("castep_addons.converge")
    bld = conv.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "lda",
        "cut_off_energy": 300,
        "symmetry_generate": True,
    }
    bld.kpoints_spacing = 0.1
    silicon = StructureData(ase=bulk("Si", "diamond", 5.43))
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
