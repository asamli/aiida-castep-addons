from pathlib import Path

import aiida.orm as orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_castep.data.otfg import upload_otfg_family
from aiida_castep_addons.workflows.phonon import (
    phonon_analysis,
    seekpath_analysis,
    thermo_analysis,
)
from ase.build import bulk


def test_phonon_analysis():
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    seekpath = seekpath_analysis(silicon, orm.Dict(dict={}))
    kpoints = seekpath["kpoints"]
    ir_folder = orm.FolderData(tree=Path("registry/Si_phonon/dfpt/out").resolve())
    raman_folder = orm.FolderData(tree=Path("registry/Si_phonon/raman/out").resolve())
    results = phonon_analysis(
        orm.Str("test_prefix"),
        ir_folder,
        kpoints,
        raman_folder,
        silicon,
        orm.ArrayData(),
    )

    assert "band_data" in results
    assert "band_plot" in results
    assert "vib_spectrum_data" in results
    assert "vib_spectra" in results


def test_thermo_analysis():
    thermo_folder = orm.FolderData(tree=Path("registry/Si_phonon/thermo/out").resolve())
    results = thermo_analysis(orm.Str("test_prefix"), thermo_folder)

    assert "thermo_data" in results
    assert "energy_plot" in results
    assert "entropy_plot" in results


def test_phonon_wc(mock_castep_code):
    phonon = WorkflowFactory("castep_addons.phonon")
    bld = phonon.get_builder()
    bld.calc.code = mock_castep_code
    upload_otfg_family(["NCP"], "NCP", "NCP potential library")
    bld.pseudos_family = "NCP"
    phonon_parameters = {
        "xc_functional": "pbesol",
        "phonon_fine_method": "interpolate",
        "fix_occupancy": True,
        "cut_off_energy": 400,
        "phonon_max_cycles": 100,
        "symmetry_generate": True,
    }
    bld.calc.parameters = phonon_parameters
    bld.kpoints_spacing = 0.04
    phonon_kpoints = orm.KpointsData()
    phonon_kpoints.set_kpoints_mesh((3, 3, 3))
    bld.calc.phonon_kpoints = phonon_kpoints
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    bld.calc.structure = silicon
    bld.calc_options = {
        "max_wallclock_seconds": 3600,
        "resources": {"num_machines": 1, "tot_num_mpiprocs": 4},
    }
    bld.metadata.label = "Si_pbesol_phonon"
    bld.metadata.description = (
        "Phonon band structure and vibrational spectra of Si with the PBEsol functional"
    )
    bld.clean_workdir = True
    _, dfpt_node = run_get_node(bld)

    bld.run_thermo = True
    bld.run_phonon = False
    _, thermo_node = run_get_node(bld)

    bld.run_thermo = False
    bld.run_phonon = True
    phonon_parameters["phonon_fine_method"] = "supercell"
    bld.calc.parameters = phonon_parameters
    _, supercell_node = run_get_node(bld)

    assert dfpt_node.is_finished_ok
    assert thermo_node.is_finished_ok
    assert supercell_node.is_finished_ok
