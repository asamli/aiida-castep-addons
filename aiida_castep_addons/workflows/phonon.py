"""Module for Phonon WorkChain"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import galore
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from aiida_castep.workflows.base import CastepBaseWorkChain
from aiida_castep_addons.parsers.phonon import PhononParser
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from PyPDF2 import PdfFileReader, PdfFileWriter
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter

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
        output_file = orm.SinglefileData(f"{temp}/{fname.value}")
    return output_file


@calcfunction
def phonon_analysis(
    prefix, ir_folder, kpoints, raman_folder, structure, experimental_spectra
):
    """Parse and plot the phonon band structure, IR spectrum and Raman spectrum"""
    ir_dot_phonon = ir_folder.get_object_content("aiida.phonon")
    ir_lines = ir_dot_phonon.split("\n")
    raman_dot_phonon = raman_folder.get_object_content("aiida.phonon")
    raman_lines = raman_dot_phonon.split("\n")

    # Parsing the .phonon files from the two calculations
    ir_phonon_data = PhononParser(ir_lines)
    raman_phonon_data = PhononParser(raman_lines)

    with TemporaryDirectory() as temp:
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
        band_plot = orm.SinglefileData(f"{temp}/{prefix.value}_phonon_bands.pdf")

        # Create BandsData for the phonon band structure
        band_data = orm.BandsData()
        band_data.set_cell_from_structure(structure)
        band_data.set_kpoints(qpoints)
        band_data.set_bands(frequencies, units="THz")
        band_data.labels = labels

        # Preparing IR and Raman spectra for plotting and saving the data as ArrayData
        ir_raw_frequencies = ir_phonon_data.vib_frequencies
        ir_raw_intensities = ir_phonon_data.ir_intensities
        ir_frequencies = np.arange(0, max(ir_raw_frequencies) * 1.2, 0.01)
        ir_xy = np.array(list(zip(ir_raw_frequencies, ir_raw_intensities)))
        ir_intensities = galore.xy_to_1d(ir_xy, ir_frequencies, spikes=True)
        ir_intensities = galore.broaden(ir_intensities, dist="lorentzian", d=0.01)
        ir_intensities = galore.broaden(ir_intensities, dist="gaussian", d=0.01)
        ir_intensities = ir_intensities.astype("float64")
        ir_intensities = [
            (
                (intensity - min(ir_intensities))
                / (max(ir_intensities) - min(ir_intensities))
            )
            for intensity in ir_intensities
        ]
        ir_frequency_unit = ir_phonon_data.frequency_unit

        raman_raw_frequencies = raman_phonon_data.vib_frequencies
        raman_raw_intensities = raman_phonon_data.raman_intensities
        raman_frequencies = np.arange(0, max(raman_raw_frequencies) * 1.2, 0.01)
        raman_xy = np.array(list(zip(raman_raw_frequencies, raman_raw_intensities)))
        raman_intensities = galore.xy_to_1d(raman_xy, raman_frequencies, spikes=True)
        raman_intensities = galore.broaden(raman_intensities, dist="lorentzian", d=0.01)
        raman_intensities = galore.broaden(raman_intensities, dist="gaussian", d=0.01)
        raman_intensities = raman_intensities.astype("float64")
        raman_intensities = [
            (
                (intensity - min(raman_intensities))
                / (max(raman_intensities) - min(raman_intensities))
            )
            for intensity in raman_intensities
        ]
        vib_spectrum_data = orm.ArrayData()
        vib_spectrum_data.set_array("ir", np.array([ir_frequencies, ir_intensities]))
        vib_spectrum_data.set_array(
            "raman", np.array([raman_frequencies, raman_intensities])
        )

        # Plotting IR and Raman spectra with matplotlib
        plt.style.use("default")
        plt.plot(ir_frequencies, ir_intensities, label="IR")
        plt.plot(raman_frequencies, raman_intensities, label="Raman")
        try:
            experimental_ir = experimental_spectra.get_array("ir")
            experimental_ir[1] = experimental_ir[1].astype("float64")
            experimental_ir[1] = [
                (
                    (intensity - min(experimental_ir[1]))
                    / (max(experimental_ir[1]) - min(experimental_ir[1]))
                )
                for intensity in experimental_ir[1]
            ]
            plt.plot(experimental_ir[0], experimental_ir[1], label="Experimental IR")
        except:
            pass
        try:
            experimental_raman = experimental_spectra.get_array("raman")
            experimental_raman[1] = experimental_raman[1].astype("float64")
            experimental_raman[1] = [
                (
                    (intensity - min(experimental_raman[1]))
                    / (max(experimental_raman[1]) - min(experimental_raman[1]))
                )
                for intensity in experimental_raman[1]
            ]
            plt.plot(
                experimental_raman[0], experimental_raman[1], label="Experimental Raman"
            )
        except:
            pass
        plt.xlabel(f"Frequency ({ir_frequency_unit})")
        plt.xlim(left=0)
        plt.ylabel(f"Normalised intensity")
        plt.ylim(bottom=0)
        plt.legend(loc="best")
        plt.savefig(fname=f"{temp}/{prefix.value}_vib_spectra.pdf", bbox_inches="tight")
        plt.close()
        vib_spectra = orm.SinglefileData(f"{temp}/{prefix.value}_vib_spectra.pdf")

    return {
        "band_data": band_data,
        "band_plot": band_plot,
        "vib_spectrum_data": vib_spectrum_data,
        "vib_spectra": vib_spectra,
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
            default=lambda: orm.Dict(dict={}),
        )
        spec.input(
            "use_supercell",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Use the finite displacement (supercell) method or not. Default is False (linear response/DFPT method).",
            required=False,
            default=lambda: orm.Bool(False),
        )
        spec.input(
            "experimental_spectra",
            valid_type=orm.ArrayData,
            help="Experimental IR and/or Raman spectra as 2D arrays. Use 'ir' and 'raman' as the array names.",
            required=False,
            default=lambda: orm.ArrayData(),
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
            "vib_spectrum_data",
            valid_type=orm.ArrayData,
            help="IR and Raman spectrum data as ArrayData",
            required=True,
        )
        spec.output(
            "vib_spectra",
            valid_type=orm.SinglefileData,
            help="IR and Raman spectra of the material as a PDF file",
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
        if prefix:
            self.ctx.prefix = prefix
        else:
            self.ctx.prefix = f'{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters["xc_functional"]}'

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
            current_structure, self.inputs.seekpath_parameters
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
        phonon_kpoints = orm.KpointsData()
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
            self.inputs.experimental_spectra,
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
        self.ctx.vib_spectrum_data = outputs["vib_spectrum_data"]
        self.ctx.vib_spectra = add_metadata(
            outputs["vib_spectra"],
            orm.Str(f"{self.ctx.prefix}_vib_spectra.pdf"),
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
        self.out("vib_spectrum_data", self.ctx.vib_spectrum_data)
        self.out("vib_spectra", self.ctx.vib_spectra)
