from ase.build import bulk
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import run_get_node
from aiida_castep.data.otfg import upload_otfg_family

def test_converge_wc(mock_castep_code):
    conv = WorkflowFactory("castep_addons.converge")
    bld = conv.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(['C19'], 'C19', 'C19 potential library')
    bld.pseudos_family = "C19"
    bld.calc.parameters = {"xc_functional":"pbesol", "cut_off_energy":300, "symmetry_generate":True}
    bld.kpoints_spacing = 0.1
    StructureData = DataFactory("structure")
    silicon = StructureData(ase=bulk('Si', 'diamond', 5.43))
    bld.calc.structure = silicon
    bld.calc_options = {'max_wallclock_seconds':3600, 'resources':{"num_machines":1, "tot_num_mpiprocs":4}}
    bld.clean_workdir = True
    results, node = run_get_node(bld)

    assert node.exit_status == 0
    assert results['conv_pwcutoff'] == 250
    assert results['conv_kspacing'] == 0.09
