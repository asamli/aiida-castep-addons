from pathlib import Path

import aiida.orm as orm
from ase.build import bulk

from aiida_castep_addons.utils import add_metadata, seekpath_analysis


def test_seekpath_analysis():
    silicon = orm.StructureData(ase=bulk("Si", "diamond", 5.43))
    seekpath = seekpath_analysis(silicon, orm.Dict(dict={}))

    assert "kpoints" in seekpath
    assert "prim_cell" in seekpath


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
