"""Module for Convergence WorkChain"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import WorkChain, append_, calcfunction, while_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from aiida_castep.workflows.base import CastepBaseWorkChain
from aiida_castep_addons.parsers.phonon import PhononParser
from matplotlib.lines import Line2D
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from PyPDF2 import PdfFileReader, PdfFileWriter
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter

__version__ = "0.0.1"


@calcfunction
def seekpath_analysis(structure):
    """
    Use seekpath for automatic k-point path generation.
    The k-point path is only valid for the generated primitive cell which may or may not be the same as the input structure.
    """
    seekpath = get_explicit_kpoints_path(structure)
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
def plot_phonons(files, kpoints, matrices, prefix):
    supercell_labels = [
        f"{matrices[i][0][0]}x{matrices[i][1][2]}x{matrices[i][2][4]}"
        for i in range(len(matrices))
    ]
    with TemporaryDirectory() as temp:
        for i, file in enumerate(files):
            with open(f"{temp}/{i}.phonon", "x") as phonon_file:
                phonon_file.write(file)

            # Parsing the .phonon files from the calculations
            phonon_data = PhononParser(open(f"{temp}/{i}.phonon"))

            # Plotting the phonon band structure with sumo and pymatgen
            qpoints = kpoints.get_kpoints()
            parsed_qpoints = np.array(phonon_data.qpoints)
            frequencies = np.array(phonon_data.frequencies) / 33.36
            bands = np.transpose(frequencies)
            lattice = Lattice(phonon_data.cell)
            rec_lattice = lattice.reciprocal_lattice
            pmg_structure = phonon_data.structure
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
            SPhononBSPlotter(pmg_bands).get_plot(
                ymin=-2,
                plt=plt,
                color=f"C{i}",
            )
        plt.legend(
            [Line2D([0], [0], color=f"C{i}") for i in range(len(supercell_labels))],
            supercell_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        plt.savefig(
            fname=f"{temp}/{prefix.value}_supercell_convergence.pdf",
            bbox_inches="tight",
        )
        return orm.SinglefileData(f"{temp}/{prefix.value}_supercell_convergence.pdf")


class CastepConvergeWorkChain(WorkChain):
    """
    WorkChain to converge the plane-wave energy cutoff and/or k-point grid spacing with respect to
    the ground state energy. The supercell size for phonon dispersion calculations can also be
    converged if needed.
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepConvergeWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input(
            "converge_pwcutoff",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Whether to converge the plane-wave cutoff or not (True by default)",
            required=False,
            default=lambda: orm.Bool(True),
        )
        spec.input(
            "converge_kspacing",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Whether to converge the k-point spacing or not (True by default)",
            required=False,
            default=lambda: orm.Bool(True),
        )
        spec.input(
            "converge_supercell",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Whether to converge the supercell for phonon calculations or not (False by default)",
            required=False,
            default=lambda: orm.Bool(False),
        )
        spec.input(
            "initial_pwcutoff",
            valid_type=orm.Int,
            serializer=to_aiida_type,
            help="Initial plane-wave cutoff value in electron volts (eV)",
            required=False,
            default=lambda: orm.Int(200),
        )
        spec.input(
            "final_pwcutoff",
            valid_type=orm.Int,
            serializer=to_aiida_type,
            help="""Final plane-wave cutoff value in electron volts (eV). 
                    It is considered the converged value if cutoff convergence is disabled.""",
            required=False,
            default=lambda: orm.Int(500),
        )
        spec.input(
            "pwcutoff_step",
            valid_type=orm.Int,
            serializer=to_aiida_type,
            help="Plane-wave cutoff step (increment) in electron volts (eV)",
            required=False,
            default=lambda: orm.Int(50),
        )
        spec.input(
            "coarse_kspacing",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="The Monkhorst-Pack k-point spacing for the coarsest grid in inverse Angstroms",
            required=False,
            default=lambda: orm.Float(0.1),
        )
        spec.input(
            "fine_kspacing",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="""The Monkhorst-Pack k-point spacing for the finest grid in inverse Angstroms. 
                    It is considered the converged value if k-spacing convergence is disabled.""",
            required=False,
            default=lambda: orm.Float(0.05),
        )
        spec.input(
            "kspacing_step",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="The Monkhorst-Pack k-point spacing step (reduction) in inverse Angstroms",
            required=False,
            default=lambda: orm.Float(0.01),
        )
        spec.input(
            "energy_tolerance",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="""The tolerance for the ground state energy in electron volts (eV). When the energy difference per atom between two
                    convergence calculations goes below this value, the former is considered converged.""",
            required=False,
            default=lambda: orm.Float(0.001),
        )
        spec.input(
            "initial_supercell_length",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="Initial supercell length in Angstroms",
            required=False,
            default=lambda: orm.Float(5.0),
        )
        spec.input(
            "final_supercell_length",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="Final supercell length in Angstroms",
            required=False,
            default=lambda: orm.Float(15.0),
        )
        spec.input(
            "supercell_step",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="Supercell length step (increment) in Angstroms",
            required=False,
            default=lambda: orm.Float(5.0),
        )
        spec.input(
            "frequency_tolerance",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="""The acceptable mean percentage error (MPE) for the phonon band frequencies. When the MPE 
                   between two convergence calculations goes below this value, the former is considered converged.""",
            required=False,
            default=lambda: orm.Float(5.0),
        )
        spec.input(
            "file_prefix",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The prefix for the name of the supercell convergence plot",
            required=False,
        )

        # The outputs
        spec.output(
            "converged_pwcutoff",
            valid_type=orm.Int,
            help="Converged plane-wave cutoff value in electron volts (eV)",
            required=False,
        )
        spec.output(
            "converged_kspacing",
            valid_type=orm.Float,
            help="The converged Monkhorst-Pack k-point spacing in inverse Angstroms",
            required=False,
        )
        spec.output(
            "converged_supercell",
            valid_type=orm.ArrayData,
            help="The transformation matrix for the converged supercell",
            required=False,
        )
        spec.output(
            "supercell_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the phonon band structures from supercell convergence as a PDF file",
            required=False,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            while_(cls.should_conv_pwcutoff)(
                cls.run_pwcutoff_conv, cls.analyse_pwcutoff_conv
            ),
            while_(cls.should_conv_kspacing)(
                cls.run_kspacing_conv, cls.analyse_kspacing_conv
            ),
            while_(cls.should_conv_supercell)(
                cls.run_supercell_conv, cls.analyse_supercell_conv
            ),
            cls.results,
        )

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.initial_pwcutoff = self.inputs.initial_pwcutoff.value
        self.ctx.final_pwcutoff = self.inputs.final_pwcutoff.value
        self.ctx.coarse_kspacing = self.inputs.coarse_kspacing.value
        self.ctx.fine_kspacing = self.inputs.fine_kspacing.value
        if self.inputs.converge_pwcutoff:
            self.ctx.pwcutoff_converged = False
        else:
            self.ctx.pwcutoff_converged = True
        if self.inputs.converge_kspacing:
            self.ctx.kspacing_converged = False
        else:
            self.ctx.kspacing_converged = True
        if self.inputs.converge_supercell:
            self.ctx.initial_supercell_length = (
                self.inputs.initial_supercell_length.value
            )
            self.ctx.final_supercell_length = self.inputs.final_supercell_length.value
            self.ctx.supercell_converged = False
            prefix = self.inputs.get("file_prefix", None)
            if prefix:
                self.ctx.prefix = prefix
            else:
                self.ctx.prefix = f'{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters["xc_functional"]}'
        else:
            self.ctx.supercell_converged = True

    def should_conv_pwcutoff(self):
        """Decide if next plane-wave energy cutoff convergence calculation should run"""
        return not self.ctx.pwcutoff_converged

    def should_conv_kspacing(self):
        """Decide if next k-point spacing convergence calculations should run"""
        return not self.ctx.kspacing_converged

    def should_conv_supercell(self):
        """Decide if next supercell convergence calculations should run"""
        return not self.ctx.supercell_converged

    def run_pwcutoff_conv(self):
        """Run parallel plane-wave energy cutoff convergence calculations with the energy cutoff range and increment provided"""
        inputs = self.ctx.inputs
        inputs.kpoints_spacing = self.ctx.coarse_kspacing
        for pwcutoff in range(
            self.ctx.initial_pwcutoff,
            self.ctx.final_pwcutoff + 1,
            self.inputs.pwcutoff_step.value,
        ):
            parameters = deepcopy(self.ctx.parameters)
            parameters["cut_off_energy"] = pwcutoff
            inputs.calc.parameters = parameters
            running = self.submit(CastepBaseWorkChain, **inputs)
            self.to_context(pwcutoff_calcs=append_(running))
        self.report("Running plane-wave cutoff convergence calculations")

    def analyse_pwcutoff_conv(self):
        """Analyse the plane-wave energy cutoff convergence calculations"""
        for i, wc in enumerate(self.ctx.pwcutoff_calcs):
            if i == 0:
                continue
            last_energy = wc.outputs.output_parameters["total_energy"]
            second_last_energy = self.ctx.pwcutoff_calcs[
                i - 1
            ].outputs.output_parameters["total_energy"]
            energy_diff_per_atom = (
                abs(last_energy - second_last_energy)
                / wc.outputs.output_parameters["num_ions"]
            )
            if energy_diff_per_atom < self.inputs.energy_tolerance:
                conv_pwcutoff = self.ctx.pwcutoff_calcs[i - 1].inputs.calc.parameters[
                    "cut_off_energy"
                ]
                self.report(f"Plane-wave energy cutoff converged at {conv_pwcutoff} eV")
                self.ctx.converged_pwcutoff = orm.Int(conv_pwcutoff)
                self.ctx.final_pwcutoff = self.ctx.converged_pwcutoff
                self.ctx.pwcutoff_converged = True
                return
        self.ctx.initial_pwcutoff = self.ctx.final_pwcutoff
        self.ctx.final_pwcutoff += 200
        self.report(
            "Plane-wave energy cutoff not converged. Increasing upper limit by 200 eV."
        )

    def run_kspacing_conv(self):
        """Run parallel k-point spacing convergence calculations with the k-point spacing range and step provided"""
        inputs = self.ctx.inputs
        inputs.kpoints_spacing = self.ctx.coarse_kspacing
        parameters = deepcopy(self.ctx.parameters)
        parameters["cut_off_energy"] = self.ctx.final_pwcutoff
        inputs.calc.parameters = parameters
        for kspacing in np.arange(
            self.ctx.coarse_kspacing,
            self.ctx.fine_kspacing - 0.01,
            -self.inputs.kspacing_step.value,
        ):
            inputs.kpoints_spacing = kspacing
            running = self.submit(CastepBaseWorkChain, **inputs)
            self.to_context(kspacing_calcs=append_(running))
        self.report("Running k-point grid spacing convergence calculations.")

    def analyse_kspacing_conv(self):
        """Analyse previous k-point spacing convergence calculations"""
        for i, wc in enumerate(self.ctx.kspacing_calcs):
            if i == 0:
                continue
            last_mesh = wc.called[-1].inputs.kpoints.get_kpoints_mesh()
            second_last_mesh = (
                self.ctx.kspacing_calcs[i - 1]
                .called[-1]
                .inputs.kpoints.get_kpoints_mesh()
            )
            if last_mesh == second_last_mesh:
                continue
            last_energy = wc.outputs.output_parameters["total_energy"]
            second_last_energy = self.ctx.kspacing_calcs[
                i - 1
            ].outputs.output_parameters["total_energy"]
            energy_diff_per_atom = (
                abs(last_energy - second_last_energy)
                / wc.outputs.output_parameters["num_ions"]
            )
            if energy_diff_per_atom < self.inputs.energy_tolerance:
                conv_kspacing = self.ctx.kspacing_calcs[
                    i - 1
                ].inputs.kpoints_spacing.value
                self.report(f"K-point spacing converged at {conv_kspacing} A-1")
                self.ctx.converged_kspacing = orm.Float(conv_kspacing)
                self.ctx.fine_kspacing = self.ctx.converged_kspacing
                self.ctx.kspacing_converged = True
                return
        self.ctx.coarse_kspacing = self.ctx.fine_kspacing
        if self.ctx.fine_kspacing >= 0.03:
            self.ctx.fine_kspacing -= 0.02
            self.report(
                "K-point spacing not converged. Decreasing lower limit by 0.02 A-1."
            )
        elif self.ctx.fine_kspacing >= 0.015:
            self.ctx.fine_kspacing -= 0.01
            self.report(
                "K-point spacing not converged but very low. Decreasing lower limit by 0.01 A-1."
            )
        else:
            self.ctx.converged_kspacing = orm.Float(self.ctx.fine_kspacing)
            self.ctx.kspacing_converged = True
            self.report(
                "K-point spacing not converged but too low to decrease further. Taking the lower limit as the converged value."
            )

    def run_supercell_conv(self):
        """Run parallel supercell convergence calculations with the supercell length range and step provided"""
        inputs = self.ctx.inputs
        parameters = deepcopy(self.ctx.parameters)
        parameters.update(
            {
                "task": "phonon",
                "phonon_fine_method": "supercell",
                "cut_off_energy": self.ctx.final_pwcutoff,
            }
        )
        inputs.calc.parameters = parameters
        seekpath_data = seekpath_analysis(inputs.calc.structure)
        self.ctx.kpoints = seekpath_data["kpoints"]
        inputs.calc.phonon_fine_kpoints = self.ctx.kpoints
        inputs.calc.structure = seekpath_data["prim_cell"]
        inputs.kpoints_spacing = self.ctx.fine_kspacing
        pmg_lattice = inputs.calc.structure.get_pymatgen().lattice
        self.ctx.supercell_lengths = np.arange(
            self.ctx.initial_supercell_length,
            self.ctx.final_supercell_length + 0.1,
            self.inputs.supercell_step.value,
        )
        for length in self.ctx.supercell_lengths:
            matrix_a = int(np.ceil(length / pmg_lattice.a))
            matrix_b = int(np.ceil(length / pmg_lattice.b))
            matrix_c = int(np.ceil(length / pmg_lattice.c))
            supercell_matrix = [f"{matrix_a} 0 0", f"0 {matrix_b} 0", f"0 0 {matrix_c}"]
            old_matrix = parameters.get("phonon_supercell_matrix", None)
            if old_matrix == supercell_matrix:
                continue
            parameters.update({"phonon_supercell_matrix": supercell_matrix})
            inputs.calc.parameters = parameters
            running = self.submit(CastepBaseWorkChain, **inputs)
            self.to_context(supercell_calcs=append_(running))
        self.report("Running supercell convergence calculations")

    def analyse_supercell_conv(self):
        """Analyse previous supercell convergence calculations"""
        files = []
        matrices = []
        for i, wc in enumerate(self.ctx.supercell_calcs):
            last_phonon_file = wc.called[-1].outputs.retrieved.get_object_content(
                "aiida.phonon"
            )
            last_matrix = wc.inputs.calc.parameters["phonon_supercell_matrix"]
            files.append(last_phonon_file)
            matrices.append(last_matrix)
            if i == 0:
                continue
            second_last_phonon_file = files[-2]
            with TemporaryDirectory() as temp:
                with open(f"{temp}/last.phonon", "x") as file:
                    file.write(last_phonon_file)
                last_phonon_data = PhononParser(open(f"{temp}/last.phonon"))
                last_freqs = np.array(last_phonon_data.frequencies)
                with open(f"{temp}/second_last.phonon", "x") as file:
                    file.write(second_last_phonon_file)
                second_last_phonon_data = PhononParser(
                    open(f"{temp}/second_last.phonon")
                )
                second_last_freqs = np.array(second_last_phonon_data.frequencies)
            percentage_errors = (
                np.absolute((second_last_freqs - last_freqs)) / last_freqs
            )
            mean_percentage_error = np.mean(percentage_errors)
            if (
                mean_percentage_error < (self.inputs.frequency_tolerance / 100)
                and not self.ctx.supercell_converged
            ):
                self.report(
                    f"Supercell converged at {self.ctx.supercell_lengths[i-1]} Angstroms"
                )
                self.ctx.supercell_converged = True
                self.ctx.converged_supercell = orm.ArrayData()
                self.ctx.converged_supercell.set_array(
                    "matrix",
                    np.array(
                        self.ctx.supercell_calcs[-1].inputs.calc.parameters[
                            "phonon_supercell_matrix"
                        ]
                    ),
                )
        if not self.ctx.converged_supercell:
            self.ctx.supercell_converged = True
            self.ctx.converged_supercell.set_array(
                "matrix",
                np.array(wc.inputs.calc.parameters["phonon_supercell_matrix"]),
            )
            self.report(
                f"Supercell not converged but very large. Taking {self.ctx.supercell_lengths[-1]} Angstroms as converged supercell."
            )
        supercell_plot = plot_phonons(
            orm.List(list=files),
            self.ctx.kpoints,
            orm.List(list=matrices),
            orm.Str(self.ctx.prefix),
        )
        self.ctx.supercell_plot = add_metadata(
            supercell_plot,
            orm.Str(f"{self.ctx.prefix}_supercell_convergence.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )

    def results(self):
        """Add converged plane-wave cutoff, k-point spacing, and supercell matrix to WorkChain outputs"""
        if self.inputs.converge_pwcutoff:
            self.out("converged_pwcutoff", self.ctx.converged_pwcutoff)
        if self.inputs.converge_kspacing:
            self.out("converged_kspacing", self.ctx.converged_kspacing)
        if self.inputs.converge_supercell:
            self.out("converged_supercell", self.ctx.converged_supercell)
            self.out("supercell_plot", self.ctx.supercell_plot)
