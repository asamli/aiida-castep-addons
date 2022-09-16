"""
Useful functions for workflows
"""
from tempfile import TemporaryDirectory

import aiida.orm as orm
from aiida.engine import calcfunction
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from PyPDF2 import PdfFileReader, PdfFileWriter


@calcfunction
def seekpath_analysis(structure, parameters):
    """
    Use seekpath for automatic k-point path generation.
    The k-point path is only valid for the generated primitive cell which may or may not be the same as the input structure.
    """
    seekpath = get_explicit_kpoints_path(structure, **parameters.get_dict())
    return {
        "kpoints": seekpath["explicit_kpoints"],
        "prim_cell": seekpath["primitive_structure"],
    }


@calcfunction
def add_metadata(file, fname, formula, uuid, label, description):
    """Add workflow metadata to a PDF file with PyPDF2"""
    with TemporaryDirectory() as temp:
        with file.open(mode="rb") as fin:
            reader = PdfFileReader(fin)
            writer = PdfFileWriter()
            writer.appendPagesFromReader(reader)
            metadata = reader.getDocumentInfo()
            writer.addMetadata(metadata)
            writer.addMetadata(
                {
                    "/Formula": formula.value,
                    "/WorkchainUUID": uuid.value,
                    "/WorkchainLabel": label.value,
                    "/WorkchainDescription": description.value,
                }
            )
            with open(f"{temp}/{fname.value}", "ab") as fout:
                writer.write(fout)
        output_file = orm.SinglefileData(f"{temp}/{fname.value}")
    return output_file
