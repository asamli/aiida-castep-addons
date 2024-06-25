"""
Module for Phonon WorkChain
"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import galore
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import ToContext, WorkChain, calcfunction, if_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.base import CastepBaseWorkChain
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter

from aiida_castep_addons.parsers.phonon import PhononParser
from aiida_castep_addons.utils import add_metadata, seekpath_analysis


@calcfunction
def phonon_analysis(prefix, ir_folder, kpoints, raman_folder, experimental_spectra):
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
        qpoints = kpoints.get_kpoints()
        parsed_qpoints = np.array(ir_phonon_data.qpoints)
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
            parsed_qpoints,
            bands,
            rec_lattice,
            labels_dict=label_dict,
            structure=pmg_structure,
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
        aiida_structure = orm.StructureData(pymatgen=pmg_structure)
        band_data.set_cell_from_structure(aiida_structure)
        band_data.set_kpoints(parsed_qpoints)
        band_data.set_bands(frequencies, units="THz")
        band_data.labels = labels

        # Plotting IR and Raman spectra with matplotlib and saving the data as ArrayData
        ir_raw_frequencies = ir_phonon_data.vib_frequencies
        ir_raw_intensities = ir_phonon_data.ir_intensities
        ir_frequencies = np.arange(0, max(ir_raw_frequencies) * 1.2, 0.1)
        ir_xy = np.array(list(zip(ir_raw_frequencies, ir_raw_intensities)))
        ir_intensities = galore.xy_to_1d(ir_xy, ir_frequencies, spikes=True)
        ir_intensities = galore.broaden(ir_intensities, dist="lorentzian", d=0.1)
        ir_intensities = galore.broaden(ir_intensities, dist="gaussian", d=0.1)
        ir_intensities = ir_intensities.astype("float64")
        ir_intensities = [
            (
                (intensity - min(ir_intensities))
                / (max(ir_intensities) - min(ir_intensities))
            )
            for intensity in ir_intensities
        ]
        ir_frequency_unit = ir_phonon_data.frequency_unit
        vib_spectrum_data = orm.ArrayData()
        vib_spectrum_data.set_array("ir", np.array([ir_frequencies, ir_intensities]))
        plt.style.use("default")
        plt.plot(ir_frequencies, ir_intensities, label="IR")
        try:
            raman_raw_frequencies = raman_phonon_data.vib_frequencies
            raman_raw_intensities = raman_phonon_data.raman_intensities
            raman_frequencies = np.arange(0, max(raman_raw_frequencies) * 1.2, 0.1)
            raman_xy = np.array(list(zip(raman_raw_frequencies, raman_raw_intensities)))
            raman_intensities = galore.xy_to_1d(
                raman_xy, raman_frequencies, spikes=True
            )
            raman_intensities = galore.broaden(
                raman_intensities, dist="lorentzian", d=0.1
            )
            raman_intensities = galore.broaden(
                raman_intensities, dist="gaussian", d=0.1
            )
            raman_intensities = raman_intensities.astype("float64")
            raman_intensities = [
                (
                    (intensity - min(raman_intensities))
                    / (max(raman_intensities) - min(raman_intensities))
                )
                for intensity in raman_intensities
            ]
            vib_spectrum_data.set_array(
                "raman", np.array([raman_frequencies, raman_intensities])
            )
            plt.plot(raman_frequencies, raman_intensities, label="Raman")
        except:
            pass

        # Plotting experimental IR and Raman spectra with matplotlib
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
        plt.close("all")
        vib_spectra = orm.SinglefileData(f"{temp}/{prefix.value}_vib_spectra.pdf")

    return {
        "band_data": band_data,
        "band_plot": band_plot,
        "vib_spectrum_data": vib_spectrum_data,
        "vib_spectra": vib_spectra,
    }


@calcfunction
def thermo_analysis(prefix, thermo_folder):
    """Parse the thermodynamics data from the .castep file and plot each quantity against temperature"""
    thermo_dot_castep = thermo_folder.get_object_content("aiida.castep")
    thermo_lines = thermo_dot_castep.split("\n")
    temperatures = []
    enthalpies = []
    free_energies = []
    entropies = []
    heat_capacities = []
    read_thermo = False
    zero_point_energy = 0
    for line in thermo_lines:
        line = line.strip()
        if "Zero-point" in line:
            read_thermo = True
            line = line.split()
            zero_point_energy = float(line[-2])
            continue
        elif read_thermo and line == "":
            break

        if read_thermo:
            if line[0] == "-" or line[0] == "T":
                continue
            else:
                line = line.split()
                temperatures.append(float(line[0]))
                enthalpies.append(float(line[1]) - zero_point_energy)
                free_energies.append(float(line[2]) - zero_point_energy)
                entropies.append(float(line[3]))
                heat_capacities.append(float(line[4]))

    thermo_data = orm.Dict(
        dict={
            "temperatures": temperatures,
            "enthalpies": enthalpies,
            "free_energies": free_energies,
            "entropies": entropies,
            "heat_capacities": heat_capacities,
        }
    )

    with TemporaryDirectory() as temp:
        # Plotting enthalpy and Helmholtz free energy against temperature
        plt.plot(temperatures, enthalpies, label="Enthalpy")
        plt.plot(temperatures, free_energies, label="Helmholtz free energy")
        plt.xlabel(f"Temperature (K)")
        plt.xlim(left=min(temperatures))
        plt.ylabel(f"eV")
        plt.ylim(bottom=min(free_energies))
        plt.legend(loc="best")
        plt.savefig(
            fname=f"{temp}/{prefix.value}_thermo_energies.pdf", bbox_inches="tight"
        )
        plt.close("all")
        energy_plot = orm.SinglefileData(f"{temp}/{prefix.value}_thermo_energies.pdf")

        # Plotting entropy and heat capacity against temperature
        plt.plot(temperatures, entropies, label="Entropy")
        plt.plot(temperatures, heat_capacities, label="Heat capacity (Cv)")
        plt.xlabel(f"Temperature (K)")
        plt.xlim(left=min(temperatures))
        plt.ylabel(f"J/mol/K")
        plt.ylim(bottom=min(entropies + heat_capacities))
        plt.legend(loc="best")
        plt.savefig(
            fname=f"{temp}/{prefix.value}_thermo_entropies.pdf", bbox_inches="tight"
        )
        plt.close("all")
        entropy_plot = orm.SinglefileData(f"{temp}/{prefix.value}_thermo_entropies.pdf")

    return {
        "thermo_data": thermo_data,
        "energy_plot": energy_plot,
        "entropy_plot": entropy_plot,
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
            default=lambda: orm.Dict(),
        )
        spec.input(
            "run_phonon",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Run the phonon band structure and Raman calculations or not (True by default)",
            required=False,
            default=lambda: orm.Bool(True),
        )
        spec.input(
            "run_thermo",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Run a thermodynamics calculation or not (False by default)",
            required=False,
            default=lambda: orm.Bool(),
        )
        spec.input(
            "thermo_parameters",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Additional CASTEP parameters for the thermodynamics calculation as a dictionary",
            required=False,
            default=lambda: orm.Dict(),
        )
        spec.input(
            "continuation_folder",
            valid_type=orm.RemoteData,
            help="The folder to use for a continuation calculation. Disables all other calculations if provided.",
            required=False,
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
            required=False,
        )
        spec.output(
            "phonon_band_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the phonon band structure as a PDF file",
            required=False,
        )
        spec.output(
            "vib_spectrum_data",
            valid_type=orm.ArrayData,
            help="IR and Raman spectrum data as ArrayData",
            required=False,
        )
        spec.output(
            "vib_spectra",
            valid_type=orm.SinglefileData,
            help="IR and Raman spectra of the material as a PDF file",
            required=False,
        )
        spec.output(
            "thermo_data",
            valid_type=orm.Dict,
            help="Parsed thermodynamics data as a dictionary",
            required=False,
        )
        spec.output(
            "thermo_energy_plot",
            valid_type=orm.SinglefileData,
            help="A plot of enthalpy and Helmholtz free energy against temperature as a PDF file",
            required=False,
        )
        spec.output(
            "thermo_entropy_plot",
            valid_type=orm.SinglefileData,
            help="A plot of entropy and heat capacity against temperature as a PDF file",
            required=False,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            if_(cls.should_run_continuation)(
                cls.run_continuation, cls.analyse_continuation
            ),
            if_(cls.should_run_phonon)(
                cls.run_phonon, cls.run_raman, cls.analyse_phonons
            ),
            if_(cls.should_run_thermo)(cls.run_thermo, cls.analyse_thermo),
            cls.results,
        )

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        phonon_kpoints = orm.KpointsData()
        phonon_kpoints.set_kpoints_mesh([3, 3, 3])
        self.ctx.phonon_kpoints = self.ctx.inputs.calc.get(
            "phonon_kpoints", phonon_kpoints
        )
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.prefix = self.inputs.get(
            "file_prefix",
            f"{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters['xc_functional']}",
        )
        self.ctx.phonon_continuation = False
        self.ctx.thermo_continuation = False

    def should_run_continuation(self):
        """Whether a continuation calculation should be run or not"""
        if self.inputs.get("continuation_folder", None):
            return True
        else:
            return False

    def should_run_phonon(self):
        """Whether the phonon band structure and Raman calculations should be run or not"""
        if self.inputs.get("continuation_folder", None):
            return False
        else:
            return self.inputs.run_phonon

    def should_run_thermo(self):
        """Whether a thermodynamics calculation should be run or not"""
        if self.inputs.get("continuation_folder", None):
            return False
        else:
            return self.inputs.run_thermo

    def run_continuation(self):
        """Run the continuation calculation"""
        inputs = self.ctx.inputs
        phonon_parameters = deepcopy(self.ctx.parameters)
        phonon_parameters.update({"continuation": "default"})
        if "task" not in phonon_parameters:
            phonon_parameters.update({"task": "phonon"})
        if "phonon_fine_kpoints" not in inputs.calc:
            seekpath_data = seekpath_analysis(
                inputs.calc.structure, self.inputs.seekpath_parameters
            )
            inputs.calc.structure = seekpath_data["prim_cell"]
            inputs.calc.phonon_fine_kpoints = seekpath_data["kpoints"]
        inputs.calc.parameters = phonon_parameters
        inputs.calc.parent_calc_folder = self.inputs.continuation_folder
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running continuation calculation")
        return ToContext(continuation=running)

    def run_phonon(self):
        """Run the phonon calculation using a phonon fine k-point path from seekpath"""
        inputs = self.ctx.inputs
        phonon_parameters = deepcopy(self.ctx.parameters)
        phonon_parameters.update({"task": "phonon+efield"})
        if "phonon_fine_method" not in phonon_parameters:
            phonon_parameters.update({"phonon_fine_method": "supercell"})
        inputs.calc.parameters = phonon_parameters
        seekpath_data = seekpath_analysis(
            inputs.calc.structure, self.inputs.seekpath_parameters
        )
        self.ctx.band_kpoints = seekpath_data["kpoints"]
        inputs.calc.structure = seekpath_data["prim_cell"]
        inputs.calc.phonon_fine_kpoints = self.ctx.band_kpoints
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

    def run_thermo(self):
        """Run the thermodynamics calculation"""
        inputs = self.ctx.inputs
        thermo_parameters = deepcopy(self.ctx.parameters)
        thermo_parameters.update({"task": "thermodynamics"})
        thermo_parameters.update(self.inputs.thermo_parameters)
        if "phonon_fine_method" not in thermo_parameters:
            thermo_parameters.update({"phonon_fine_method": "supercell"})
        inputs.calc.parameters = thermo_parameters
        inputs.calc.phonon_kpoints = self.ctx.phonon_kpoints
        seekpath_data = seekpath_analysis(
            inputs.calc.structure, self.inputs.seekpath_parameters
        )
        inputs.calc.structure = seekpath_data["prim_cell"]
        inputs.calc.phonon_fine_kpoints = seekpath_data["kpoints"]
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running thermodynamics calculation")
        return ToContext(thermo=running)

    def analyse_continuation(self):
        """Analyse the outputs of the continuation calculation"""
        continuation_folder = self.ctx.continuation.called[-1].outputs.retrieved
        if "aiida.phonon" in continuation_folder.list_object_names():
            self.ctx.phonon_continuation = True
            outputs = phonon_analysis(
                orm.Str(self.ctx.prefix),
                continuation_folder,
                self.ctx.continuation.inputs.calc.phonon_fine_kpoints,
                continuation_folder,
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
        if "Thermodynamics" in continuation_folder.get_object_content("aiida.castep"):
            self.ctx.thermo_continuation = True
            outputs = thermo_analysis(
                orm.Str(self.ctx.prefix),
                continuation_folder,
            )
            self.ctx.thermo_data = outputs["thermo_data"]
            self.ctx.thermo_energy_plot = add_metadata(
                outputs["energy_plot"],
                orm.Str(f"{self.ctx.prefix}_thermo_energies.pdf"),
                orm.Str(self.ctx.inputs.calc.structure.get_formula()),
                orm.Str(self.uuid),
                orm.Str(self.inputs.metadata.get("label", "")),
                orm.Str(self.inputs.metadata.get("description", "")),
            )
            self.ctx.thermo_entropy_plot = add_metadata(
                outputs["entropy_plot"],
                orm.Str(f"{self.ctx.prefix}_thermo_entropies.pdf"),
                orm.Str(self.ctx.inputs.calc.structure.get_formula()),
                orm.Str(self.uuid),
                orm.Str(self.inputs.metadata.get("label", "")),
                orm.Str(self.inputs.metadata.get("description", "")),
            )
        self.report("Continuation analysis complete")

    def analyse_phonons(self):
        """Analyse the output .phonon file from the calculations to plot the phonon band structure and extract IR and Raman spectrum data"""
        outputs = phonon_analysis(
            orm.Str(self.ctx.prefix),
            self.ctx.phonon.called[-1].outputs.retrieved,
            self.ctx.band_kpoints,
            self.ctx.raman.called[-1].outputs.retrieved,
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

    def analyse_thermo(self):
        """Analyse the output .castep file from the thermodynamics calculation"""
        outputs = thermo_analysis(
            orm.Str(self.ctx.prefix),
            self.ctx.thermo.called[-1].outputs.retrieved,
        )
        self.ctx.thermo_data = outputs["thermo_data"]
        self.ctx.thermo_energy_plot = add_metadata(
            outputs["energy_plot"],
            orm.Str(f"{self.ctx.prefix}_thermo_energies.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.thermo_entropy_plot = add_metadata(
            outputs["entropy_plot"],
            orm.Str(f"{self.ctx.prefix}_thermo_entropies.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.report("Thermodynamics analysis complete")

    def results(self):
        """Add the phonon band structure plot, IR spectrum data and Raman spectrum data to the WorkChain outputs"""
        if self.inputs.run_phonon or self.ctx.phonon_continuation:
            self.out("phonon_bands", self.ctx.phonon_bands)
            self.out("phonon_band_plot", self.ctx.phonon_band_plot)
            self.out("vib_spectrum_data", self.ctx.vib_spectrum_data)
            self.out("vib_spectra", self.ctx.vib_spectra)
        if self.inputs.run_thermo or self.ctx.thermo_continuation:
            self.out("thermo_data", self.ctx.thermo_data)
            self.out("thermo_energy_plot", self.ctx.thermo_energy_plot)
            self.out("thermo_entropy_plot", self.ctx.thermo_entropy_plot)
