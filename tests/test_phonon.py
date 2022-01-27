from ase.build import bulk
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import run_get_node
from aiida_castep.data.otfg import upload_otfg_family

def test_phonon_wc(mock_castep_code):
    phonon = WorkflowFactory("castep_addons.phonon")
    bld = phonon.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(['NCP'], 'NCP', 'NCP potential library')
    bld.pseudos_family = "NCP"
    bld.calc.parameters = {"xc_functional":"pbesol", "fix_occupancy":True, "cut_off_energy":800, "phonon_max_cycles":100, "symmetry_generate":True}
    bld.kpoints_spacing = 0.09
    KpointsData = DataFactory("array.kpoints")
    phonon_kpoints = KpointsData()
    phonon_kpoints.set_kpoints_mesh((3, 3, 3))
    bld.calc.phonon_kpoints = phonon_kpoints
    StructureData = DataFactory("structure")
    silicon = StructureData(ase=bulk('Si', 'diamond', 5.43))
    bld.calc.structure = silicon
    bld.calc_options = {'max_wallclock_seconds':3600, 'resources':{"num_machines":1, "tot_num_mpiprocs":4}}
    bld.metadata.label = 'MgO_pbesol_phonon'
    bld.metadata.description = 'Phonon band structure and vibrational spectra of MgO with the PBEsol functional'
    bld.clean_workdir = True
    _, dfpt_node = run_get_node(bld)
    
    bld.use_supercell = True
    _, supercell_node = run_get_node(bld)

    assert dfpt_node.exit_status == 0
    assert supercell_node.exit_status == 0