"""Module for Phonon WorkChain"""

from __future__ import absolute_import
from aiida.engine import WorkChain, ToContext, calcfunction
import aiida.orm as orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from aiida_castep.workflows.base import CastepBaseWorkChain
from aiida.plugins import DataFactory
from aiida_castep_addons.parsers.phonon import PhononParser

from copy import deepcopy
from tempfile import TemporaryDirectory
from PyPDF2 import PdfFileReader, PdfFileWriter
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter

SinglefileData = DataFactory("singlefile")
KpointsData = DataFactory("array.kpoints")
XyData = DataFactory("array.xy")
BandsData = DataFactory("array.bands")

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
def phonon_analysis(prefix, ir_folder, kpoints, raman_folder, structure):
    """Parse and plot the phonon band structure, IR spectrum and Raman spectrum"""
    ir_dot_phonon = ir_folder.get_object_content("aiida.phonon")
    raman_dot_phonon = raman_folder.get_object_content("aiida.phonon")
    with TemporaryDirectory() as temp:
        with open(f"{temp}/ir.phonon", "x") as phonon_file:
            phonon_file.write(ir_dot_phonon)
        with open(f"{temp}/raman.phonon", "x") as raman_file:
            raman_file.write(raman_dot_phonon)

        # Parsing the .phonon files from the two calculations
        ir_phonon_data = PhononParser(open(f"{temp}/ir.phonon"))
        raman_phonon_data = PhononParser(open(f"{temp}/raman.phonon"))

        # Plotting the phonon band structure with sumo and pymatgen
        qpoints = np.array(ir_phonon_data.qpoints)
        frequencies = np.array(ir_phonon_data.frequencies) / 33.36
        bands = np.transpose(frequencies)
        lattice = Lattice(ir_phonon_data.cell)
        rec_lattice = lattice.reciprocal_lattice
        pmg_structure = ir_phonon_data.structure
        labels = kpoints.labels
        label_dict = {}
        for index, label in labels:
            qpoint = qpoints[index]
            if label == "GAMMA":
                label_dict[r"\Gamma"] = qpoint
            else:
                label_dict[label] = qpoint
        pmg_bands = PhononBandStructureSymmLine(
            qpoints, bands, rec_lattice, labels_dict=label_dict, structure=pmg_structure
        )
        phonon_plotter = SPhononBSPlotter(pmg_bands).get_plot(ymin=-2)
        phonon_plotter.plot()
        phonon_plotter.savefig(
            fname=f"{temp}/{prefix.value}_phonon_bands.pdf", bbox_inches="tight"
        )
        phonon_plotter.close()
        band_plot = SinglefileData(f"{temp}/{prefix.value}_phonon_bands.pdf")

        # Create BandsData for the phonon band structure
        band_data = BandsData()
        band_data.set_cell_from_structure(structure)
        band_data.set_kpoints(qpoints)
        band_data.set_bands(frequencies, units="THz")
        band_data.labels = labels

        # Plotting IR spectrum with matplotlib and saving IR data as XyData
        ir_frequencies = ir_phonon_data.vib_frequencies
        ir_intensities = ir_phonon_data.ir_intensities
        ir_frequency_unit = ir_phonon_data.frequency_unit
        ir_intensity_unit = ir_phonon_data.ir_unit
        ir_data = XyData()
        ir_data.set_x(np.array(ir_frequencies), "Wavenumber", ir_frequency_unit)
        ir_data.set_y(np.array(ir_intensities), "Intensity", ir_intensity_unit)
        plt.style.use("default")
        plt.bar(ir_frequencies, ir_intensities, color="red")
        plt.xlabel(f"Wavenumber ({ir_frequency_unit})")
        plt.ylabel(f"Intensity ({ir_intensity_unit})")
        plt.savefig(fname=f"{temp}/{prefix.value}_ir.pdf", bbox_inches="tight")
        plt.close()
        ir_spectrum = SinglefileData(f"{temp}/{prefix.value}_ir.pdf")

        # Plotting Raman spectrum with matplotlib and saving Raman data as XyData
        raman_frequencies = raman_phonon_data.vib_frequencies
        raman_activities = raman_phonon_data.raman_activities
        raman_frequency_unit = raman_phonon_data.frequency_unit
        raman_activity_unit = raman_phonon_data.raman_unit
        raman_data = XyData()
        raman_data.set_x(
            np.array(raman_frequencies), "Raman shift", raman_frequency_unit
        )
        ir_data.set_y(np.array(raman_activities), "Intensity", raman_activity_unit)
        plt.bar(raman_frequencies, raman_activities, color="red")
        plt.xlabel(f"Raman shift ({raman_frequency_unit})")
        plt.ylabel(f"Intensity ({raman_activity_unit})")
        plt.savefig(fname=f"{temp}/{prefix.value}_raman.pdf", bbox_inches="tight")
        plt.close()
        raman_spectrum = SinglefileData(f"{temp}/{prefix.value}_raman.pdf")

    return {
        "band_data": band_data,
        "band_plot": band_plot,
        "ir_data": ir_data,
        "ir_spectrum": ir_spectrum,
        "raman_data": raman_data,
        "raman_spectrum": raman_spectrum,
    }


class CastepPhononWorkChain(WorkChain):
    """
    WorkChain to calculate and plot the phonon band structure,
    the IR spectrum and Raman spectrum for materials.
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepPhononWorkChain, cls).define(spec)

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
        spec.input(
            "use_supercell",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Use the finite displacement (supercell) method or not. Default is False (linear response/DFPT method).",
            required=False,
            default=lambda: orm.Bool(False),
        )

        # The outputs
        spec.output(
            "phonon_bands",
            valid_type=orm.BandsData,
            help="The parsed BandsData for the phonon bands.",
            required=True,
        )
        spec.output(
            "phonon_band_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the phonon band structure as a PDF file",
            required=True,
        )
        spec.output(
            "ir_data",
            valid_type=orm.XyData,
            help="IR spectrum data for vibrational modes as XyData",
            required=True,
        )
        spec.output(
            "ir_spectrum",
            valid_type=orm.SinglefileData,
            help="IR spectrum of the material as a PDF file",
            required=True,
        )
        spec.output(
            "raman_data",
            valid_type=orm.XyData,
            help="Raman spectrum data for vibrational modes as XyData",
            required=True,
        )
        spec.output(
            "raman_spectrum",
            valid_type=orm.SinglefileData,
            help="Raman spectrum of the material as a PDF file",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup, cls.run_phonon, cls.run_raman, cls.analyse_phonons, cls.results
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

    def run_phonon(self):
        """Run the phonon calculation using a phonon fine k-point path from seekpath"""
        inputs = self.exposed_inputs(CastepBaseWorkChain)
        phonon_parameters = deepcopy(self.ctx.parameters)
        phonon_parameters.update({"task": "phonon+efield"})
        if self.inputs.use_supercell:
            phonon_parameters.update({"phonon_fine_method": "supercell"})
        else:
            phonon_parameters.update(
                {"phonon_fine_method": "interpolate", "fix_occupancy": True}
            )
        inputs.calc.parameters = phonon_parameters
        current_structure = inputs.calc.structure
        seekpath_data = seekpath_analysis(
            current_structure, orm.Dict(dict=self.ctx.seekpath_parameters)
        )
        self.ctx.band_kpoints = seekpath_data["kpoints"]
        inputs.calc.phonon_fine_kpoints = self.ctx.band_kpoints
        inputs.calc.structure = seekpath_data["prim_cell"]
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running phonon band structure calculation")
        return ToContext(phonon=running)

    def run_raman(self):
        """Run the gamma-only Raman spectrum calculation"""
        inputs = self.ctx.inputs
        raman_parameters = deepcopy(self.ctx.parameters)
        phonon_kpoints = KpointsData()
        phonon_kpoints.set_kpoints_mesh((1, 1, 1))
        inputs.calc.phonon_kpoints = phonon_kpoints
        raman_parameters.update({"task": "phonon", "calculate_raman": True})
        raman_parameters.pop("phonon_fine_method", None)
        inputs.calc.parameters = raman_parameters
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running gamma-only Raman spectrum calculation")
        return ToContext(raman=running)

    def analyse_phonons(self):
        """Analyse the output .phonon file from the calculation to plot the phonon band structure and extract IR and Raman spectrum data"""
        outputs = phonon_analysis(
            orm.Str(self.ctx.prefix),
            self.ctx.phonon.called[-1].outputs.retrieved,
            self.ctx.band_kpoints,
            self.ctx.raman.called[-1].outputs.retrieved,
            self.ctx.inputs.calc.structure,
        )
        self.ctx.phonon_bands = outputs["band_data"]
        self.ctx.phonon_band_plot = add_metadata(
            outputs["band_plot"],
            orm.Str(f"{self.ctx.prefix}_phonon_bands.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.ir_data = outputs["ir_data"]
        self.ctx.ir_spectrum = add_metadata(
            outputs["ir_spectrum"],
            orm.Str(f"{self.ctx.prefix}_ir.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.raman_data = outputs["raman_data"]
        self.ctx.raman_spectrum = add_metadata(
            outputs["raman_spectrum"],
            orm.Str(f"{self.ctx.prefix}_raman.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.report("Phonon analysis complete")

    def results(self):
        """Add the phonon band structure plot, IR spectrum data and Raman spectrum data to the WorkChain outputs"""
        self.out("phonon_bands", self.ctx.phonon_bands)
        self.out("phonon_band_plot", self.ctx.phonon_band_plot)
        self.out("ir_data", self.ctx.ir_data)
        self.out("ir_spectrum", self.ctx.ir_spectrum)
        self.out("raman_data", self.ctx.raman_data)
        self.out("raman_spectrum", self.ctx.raman_spectrum)
