"""Module for Density of States and Band Structure WorkChain"""
from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.plugins import DataFactory
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from aiida_castep.utils.dos import DOSProcessor
from aiida_castep.workflows.base import CastepBaseWorkChain
from aiida_castep_addons.utils.sumo_plotter import get_sumo_bands_plotter
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import Dos
from PyPDF2 import PdfFileReader, PdfFileWriter
from sumo.plotting.dos_plotter import SDOSPlotter

SinglefileData = DataFactory("singlefile")

__version__ = "0.0.1"


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
        output_file = SinglefileData(f"{temp}/{fname.value}")
    return output_file


@calcfunction
def analysis(dos_data, band_data, band_kpoints, prefix):
    """Plot the density of states and band structure with sumo"""
    # Plotting DOS from parsed BandsData
    dos_processor = DOSProcessor(
        dos_data.get_bands(), dos_data.get_array("weights"), smearing=0.2
    )
    dos = dos_processor.get_dos()
    dos_energies = dos[0]
    dos_densities = dos[1]
    if dos_data.get_attribute("nspins") > 1:
        dos_efermi = dos_data.get_attribute("efermi")[0]
        pmg_dos = Dos(
            dos_efermi,
            dos_energies,
            {Spin.up: dos_densities[0], Spin.down: dos_densities[1]},
        )
    else:
        dos_efermi = dos_data.get_attribute("efermi")
        pmg_dos = Dos(dos_efermi, dos_energies, {Spin.up: dos_densities[0]})
    dos_plotter = SDOSPlotter(pmg_dos, {}).get_plot()
    dos_plotter.plot()

    with TemporaryDirectory() as temp:
        dos_plotter.savefig(fname=f"{temp}/{prefix.value}_dos.pdf", bbox_inches="tight")
        dos_plot = SinglefileData(f"{temp}/{prefix.value}_dos.pdf")

        # Plotting band structure
        labelled_bands = deepcopy(band_data)
        labelled_bands.labels = band_kpoints.labels
        if labelled_bands.get_attribute("nspins") > 1:
            bands_efermi = labelled_bands.get_attribute("efermi")[0]
        else:
            bands_efermi = labelled_bands.get_attribute("efermi")
        band_plotter = get_sumo_bands_plotter(labelled_bands, bands_efermi).get_plot()
        band_plotter.plot()
        band_plotter.savefig(
            fname=f"{temp}/{prefix.value}_bands.pdf", bbox_inches="tight"
        )
        band_plot = SinglefileData(f"{temp}/{prefix.value}_bands.pdf")

    return {
        "dos_plot": dos_plot,
        "labelled_bands": labelled_bands,
        "band_plot": band_plot,
    }


class CastepBandPlotWorkChain(WorkChain):
    """
    WorkChain to calculate and plot the density of states and band structure
    of a material
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepBandPlotWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input(
            "file_prefix",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The prefix for the names of output files",
            required=False,
        )
        spec.input(
            "seekpath_parameters",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Parameters to use with seekpath for the k-point path generation",
            required=False,
        )

        # The outputs
        spec.output(
            "dos_data",
            valid_type=orm.BandsData,
            help="The parsed BandsData for the density of states calculation",
            required=True,
        )
        spec.output(
            "dos_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the density of states as a PDF file",
            required=True,
        )
        spec.output(
            "labelled_band_data",
            valid_type=orm.BandsData,
            help="The labelled BandsData for the band structure calculation",
            required=True,
        )
        spec.output(
            "band_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the band structure as a PDF file",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup, cls.run_dos, cls.run_bands, cls.analyse_calculations, cls.results
        )

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        prefix = self.inputs.get("file_prefix", None)
        if prefix is None:
            self.ctx.prefix = f'{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters["xc_functional"]}'
        else:
            self.ctx.prefix = prefix
        self.ctx.seekpath_parameters = self.inputs.get("seekpath_parameters", {})

    def run_dos(self):
        """Run the spectral density of states calculation"""
        inputs = self.ctx.inputs
        dos_parameters = deepcopy(self.ctx.parameters)
        if ("spectral_kpoints" not in inputs.calc) and (
            "spectral_kpoint_mp_spacing" not in dos_parameters
        ):
            if "kpoints_spacing" not in inputs:
                dos_parameters.update({"spectral_kpoint_mp_spacing": 0.03})
                self.report(
                    "No spectral k-point spacing or mesh provided. Setting spectral k-point spacing to 0.03 A-1(this may not be converged)."
                )
            else:
                dos_parameters.update(
                    {"spectral_kpoint_mp_spacing": inputs.kpoints_spacing / 3}
                )
        dos_parameters.update(
            {
                "task": "spectral",
                "spectral_task": "dos",
                "spectral_perc_extra_bands": 50,
            }
        )
        inputs.calc.parameters = dos_parameters
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running spectral density of states calculation")
        return ToContext(dos=running)

    def run_bands(self):
        """Run the spectral band structure calculation with k-point path from seekpath"""
        inputs = self.ctx.inputs
        band_parameters = deepcopy(self.ctx.parameters)
        band_parameters.update(
            {
                "task": "spectral",
                "spectral_task": "bandstructure",
                "spectral_perc_extra_bands": 50,
            }
        )
        inputs.calc.parameters = band_parameters
        current_structure = inputs.calc.structure
        seekpath_data = seekpath_analysis(
            current_structure, orm.Dict(dict=self.ctx.seekpath_parameters)
        )
        self.ctx.band_kpoints = seekpath_data["kpoints"]
        inputs.calc.spectral_kpoints = self.ctx.band_kpoints
        inputs.calc.structure = seekpath_data["prim_cell"]
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running spectral band structure calculation")
        return ToContext(bands=running)

    def analyse_calculations(self):
        """Analyse the two calculations to plot the density of states and band structure"""
        outputs = analysis(
            self.ctx.dos.outputs.output_bands,
            self.ctx.bands.outputs.output_bands,
            self.ctx.band_kpoints,
            orm.Str(self.ctx.prefix),
        )
        self.ctx.dos_plot = add_metadata(
            outputs["dos_plot"],
            orm.Str(f"{self.ctx.prefix}_dos.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.label),
            orm.Str(self.inputs.metadata.description),
        )
        self.ctx.labelled_bands = outputs["labelled_bands"]
        self.ctx.band_plot = add_metadata(
            outputs["band_plot"],
            orm.Str(f"{self.ctx.prefix}_bands.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.label),
            orm.Str(self.inputs.metadata.description),
        )
        self.report("Density of states and band structure plotted")

    def results(self):
        """Add the density of states and band structure plots to WorkChain outputs along with their raw data"""
        self.out("dos_data", self.ctx.dos.outputs.output_bands)
        self.out("dos_plot", self.ctx.dos_plot)
        self.out("labelled_band_data", self.ctx.labelled_bands)
        self.out("band_plot", self.ctx.band_plot)
