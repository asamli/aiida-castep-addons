from ase.build import bulk
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import run_get_node
from aiida_castep.data.otfg import upload_otfg_family


def test_band_plot_wc(mock_castep_code):
    bands = WorkflowFactory("castep_addons.band_plot")
    bld = bands.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["C19"], "C19", "C19 potential library")
    bld.pseudos_family = "C19"
    bld.calc.parameters = {
        "xc_functional": "pbesol",
        "cut_off_energy": 250,
        "symmetry_generate": True,
    }
    bld.kpoints_spacing = 0.09
    StructureData = DataFactory("structure")
    silicon = StructureData(ase=bulk("Si", "diamond", 5.43))
    bld.calc.structure = silicon
    bld.calc_options = {
        "max_wallclock_seconds": 3600,
        "resources": {"num_machines": 1, "tot_num_mpiprocs": 4},
    }
    bld.metadata.label = "Si_pbesol_dos+bands"
    bld.metadata.description = (
        "Density of states and band structure of silicon with PBEsol functional"
    )
    bld.clean_workdir = True
    _, node = run_get_node(bld)

    assert node.exit_status == 0
