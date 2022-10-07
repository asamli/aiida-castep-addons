"""Module for Convergence WorkChain"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import WorkChain, append_, calcfunction, while_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.base import CastepBaseWorkChain
from aiida_castep_addons.parsers.phonon import PhononParser
from aiida_castep_addons.utils import add_metadata, seekpath_analysis
from matplotlib.lines import Line2D
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter


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
            "converge_settings",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="""Settings to modify the convergence calculations. Accepted keys are 'converge_pwcutoff' (True or False), 'converge_kspacing' (True or False), 'converge_supercell' (True or False), 'pwcutoff_start' (value in eV), 'pwcutoff_end' (value in eV), 'pwcutoff_step' (value in eV),
             'kspacing_start (value in inverse Angstroms)', 'kspacing_end (value in inverse Angstroms)', 'kspacing_step (value in inverse Angstroms)', 'supercell_start (length in Angstroms)', 'supercell_end (length in Angstroms)', 'supercell_step (length in Angstroms)', 
             'energy_tolerance (value in eV)' and 'frequency_tolerance (value as a percentage)'.""",
            required=False,
            default=lambda: orm.Dict(
                dict={
                    "converge_pwcutoff": True,
                    "converge_kspacing": True,
                    "converge_supercell": False,
                    "pwcutoff_start": 200,
                    "pwcutoff_end": 500,
                    "pwcutoff_step": 50,
                    "kspacing_start": 0.1,
                    "kspacing_end": 0.05,
                    "kspacing_step": 0.01,
                    "supercell_start": 5.0,
                    "supercell_end": 15.0,
                    "supercell_step": 5.0,
                    "energy_tolerance": 0.001,
                    "frequency_tolerance": 5.0,
                }
            ),
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
        converge_settings = self.inputs.converge_settings.get_dict()
        if converge_settings.get("converge_pwcutoff", True):
            self.ctx.pwcutoff_start = converge_settings.get("pwcutoff_start", 200)
            self.ctx.pwcutoff_end = converge_settings.get("pwcutoff_end", 500)
            self.ctx.pwcutoff_step = converge_settings.get("pwcutoff_step", 50)
            self.ctx.kspacing_start = converge_settings.get("kspacing_start", 0.1)
            self.ctx.energy_tolerance = converge_settings.get("energy_tolerance", 0.001)
            self.ctx.pwcutoff_converged = False
        else:
            self.ctx.pwcutoff_converged = True

        if converge_settings.get("converge_kspacing", True):
            self.ctx.kspacing_start = converge_settings.get("kspacing_start", 0.1)
            self.ctx.kspacing_end = converge_settings.get("kspacing_end", 0.05)
            self.ctx.kspacing_step = converge_settings.get("kspacing_step", 0.01)
            self.ctx.pwcutoff_end = converge_settings.get("pwcutoff_end", 500)
            self.ctx.energy_tolerance = converge_settings.get("energy_tolerance", 0.001)
            self.ctx.kspacing_converged = False
        else:
            self.ctx.kspacing_converged = True

        if converge_settings.get("converge_supercell", False):
            self.ctx.supercell_start = converge_settings.get("supercell_start", 5.0)
            self.ctx.supercell_end = converge_settings.get("supercell_end", 15.0)
            self.ctx.supercell_step = converge_settings.get("supercell_step", 5.0)
            self.ctx.pwcutoff_end = converge_settings.get("pwcutoff_end", 500)
            self.ctx.kspacing_end = converge_settings.get("kspacing_end", 0.05)
            self.ctx.frequency_tolerance = converge_settings.get(
                "frequency_tolerance", 5.0
            )
            self.ctx.supercell_converged = False
            self.ctx.prefix = self.inputs.get(
                "file_prefix",
                f"{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters['xc_functional']}",
            )
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
        inputs.kpoints_spacing = self.ctx.kspacing_start
        for pwcutoff in range(
            self.ctx.pwcutoff_start,
            self.ctx.pwcutoff_end + 1,
            self.ctx.pwcutoff_step,
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
            if energy_diff_per_atom < self.ctx.energy_tolerance:
                conv_pwcutoff = self.ctx.pwcutoff_calcs[i - 1].inputs.calc.parameters[
                    "cut_off_energy"
                ]
                self.report(f"Plane-wave energy cutoff converged at {conv_pwcutoff} eV")
                self.ctx.converged_pwcutoff = orm.Int(conv_pwcutoff)
                self.ctx.pwcutoff_end = self.ctx.converged_pwcutoff
                self.ctx.pwcutoff_converged = True
                return
        self.ctx.pwcutoff_start = self.ctx.pwcutoff_end
        self.ctx.pwcutoff_end += 200
        self.report(
            "Plane-wave energy cutoff not converged. Increasing upper limit by 200 eV."
        )

    def run_kspacing_conv(self):
        """Run parallel k-point spacing convergence calculations with the k-point spacing range and step provided"""
        inputs = self.ctx.inputs
        inputs.kpoints_spacing = self.ctx.kspacing_start
        parameters = deepcopy(self.ctx.parameters)
        parameters["cut_off_energy"] = self.ctx.pwcutoff_end
        inputs.calc.parameters = parameters
        for kspacing in np.arange(
            self.ctx.kspacing_start,
            self.ctx.kspacing_end - 0.01,
            -self.ctx.kspacing_step,
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
            if energy_diff_per_atom < self.ctx.energy_tolerance:
                conv_kspacing = self.ctx.kspacing_calcs[
                    i - 1
                ].inputs.kpoints_spacing.value
                self.report(f"K-point spacing converged at {conv_kspacing} A-1")
                self.ctx.converged_kspacing = orm.Float(conv_kspacing)
                self.ctx.kspacing_end = self.ctx.converged_kspacing
                self.ctx.kspacing_converged = True
                return
        self.ctx.kspacing_start = self.ctx.kspacing_end
        if self.ctx.kspacing_end >= 0.03:
            self.ctx.kspacing_end -= 0.02
            self.report(
                "K-point spacing not converged. Decreasing lower limit by 0.02 A-1."
            )
        elif self.ctx.kspacing_end >= 0.015:
            self.ctx.kspacing_end -= 0.01
            self.report(
                "K-point spacing not converged but very low. Decreasing lower limit by 0.01 A-1."
            )
        else:
            self.ctx.converged_kspacing = orm.Float(self.ctx.kspacing_end)
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
                "cut_off_energy": self.ctx.pwcutoff_end,
            }
        )
        inputs.calc.parameters = parameters
        seekpath_data = seekpath_analysis(inputs.calc.structure, orm.Dict(dict={}))
        self.ctx.kpoints = seekpath_data["kpoints"]
        inputs.calc.phonon_fine_kpoints = self.ctx.kpoints
        inputs.calc.structure = seekpath_data["prim_cell"]
        inputs.kpoints_spacing = self.ctx.kspacing_end
        pmg_lattice = inputs.calc.structure.get_pymatgen().lattice
        self.ctx.supercell_lengths = np.arange(
            self.ctx.supercell_start,
            self.ctx.supercell_end + 0.1,
            self.ctx.supercell_step,
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
                mean_percentage_error < (self.ctx.frequency_tolerance / 100)
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
        if not self.ctx.supercell_converged:
            self.ctx.supercell_converged = True
            self.ctx.converged_supercell = orm.ArrayData()
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
        if self.ctx.get("converged_pwcutoff", None):
            self.out("converged_pwcutoff", self.ctx.converged_pwcutoff)
        if self.ctx.get("converged_kspacing", None):
            self.out("converged_kspacing", self.ctx.converged_kspacing)
        if self.ctx.get("converged_supercell", None):
            self.out("converged_supercell", self.ctx.converged_supercell)
            self.out("supercell_plot", self.ctx.supercell_plot)
