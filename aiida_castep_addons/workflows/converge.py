"""Module for Convergence WorkChain"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import numpy as np
from aiida.engine import WorkChain, calcfunction, while_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from aiida_castep.workflows.base import CastepBaseWorkChain
from aiida_castep_addons.parsers.phonon import PhononParser

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
            help="Final plane-wave cutoff value in electron volts (eV)",
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
            help="The Monkhorst-Pack k-point spacing for the finest grid in inverse Angstroms",
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
            "energy_error",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="""The acceptable error for the ground state energy in electron volts (eV). When the energy difference per atom between two
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
            "frequency_error",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            help="""The acceptable mean percentage error (MPE) for the phonon band frequencies. When the MPE 
                   between two convergence calculations goes below this value, the former is considered converged.""",
            required=False,
            default=lambda: orm.Float(5.0),
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
        if self.inputs.converge_pwcutoff:
            self.ctx.initial_pwcutoff = self.inputs.initial_pwcutoff.value
            self.ctx.final_pwcutoff = self.inputs.final_pwcutoff.value
            self.ctx.pwcutoff_converged = False
        else:
            self.ctx.pwcutoff_converged = True
        if self.inputs.converge_kspacing:
            self.ctx.coarse_kspacing = self.inputs.coarse_kspacing.value
            self.ctx.fine_kspacing = self.inputs.fine_kspacing.value
            self.ctx.kspacing_converged = False
        else:
            self.ctx.kspacing_converged = True
        if self.inputs.converge_supercell:
            self.ctx.initial_supercell_length = (
                self.inputs.initial_supercell_length.value
            )
            self.ctx.final_supercell_length = self.inputs.final_supercell_length.value
            self.ctx.supercell_converged = False
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
        inputs.kpoints_spacing = self.inputs.coarse_kspacing.value
        for pwcutoff in range(
            self.ctx.initial_pwcutoff,
            self.ctx.final_pwcutoff + 1,
            self.inputs.pwcutoff_step.value,
        ):
            parameters = deepcopy(self.ctx.parameters)
            parameters["cut_off_energy"] = pwcutoff
            inputs.calc.parameters = parameters
            running = self.submit(CastepBaseWorkChain, **inputs)
            key = f"pwcutoff_{pwcutoff}"
            self.to_context(**{key: running})
        self.report("Running plane-wave cutoff convergence calculations")

    def analyse_pwcutoff_conv(self):
        """Analyse the plane-wave energy cutoff convergence calculations"""
        keys = []
        for pwcutoff in range(
            self.ctx.initial_pwcutoff,
            self.ctx.final_pwcutoff + 1,
            self.inputs.pwcutoff_step.value,
        ):
            key = f"pwcutoff_{pwcutoff}"
            keys.append(key)
            if len(keys) == 1:
                continue
            last_energy = self.ctx[key].outputs.output_parameters["total_energy"]
            second_last_energy = self.ctx[keys[-2]].outputs.output_parameters[
                "total_energy"
            ]
            energy_diff_per_atom = (
                abs(last_energy - second_last_energy)
                / self.ctx[key].outputs.output_parameters["num_ions"]
            )
            if energy_diff_per_atom < self.inputs.energy_error:
                self.report(
                    f"Plane-wave energy cutoff converged at {pwcutoff - self.inputs.pwcutoff_step.value} eV"
                )
                self.ctx.converged_pwcutoff = orm.Int(
                    pwcutoff - self.inputs.pwcutoff_step.value
                )
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
        parameters["cut_off_energy"] = self.inputs.initial_pwcutoff.value
        inputs.calc.parameters = parameters
        self.ctx.kspacings = np.arange(
            self.ctx.coarse_kspacing,
            self.ctx.fine_kspacing,
            -self.inputs.kspacing_step.value,
        )
        for kspacing in self.ctx.kspacings:
            inputs.kpoints_spacing = kspacing
            running = self.submit(CastepBaseWorkChain, **inputs)
            key = f"kspacing_{kspacing}"
            self.to_context(**{key: running})
        self.report("Running k-point grid spacing convergence calculations.")

    def analyse_kspacing_conv(self):
        """Analyse previous k-point spacing convergence calculations"""
        keys = []
        for i, kspacing in enumerate(self.ctx.kspacings):
            key = f"kspacing_{kspacing}"
            keys.append(key)
            if len(keys) == 1:
                continue
            last_energy = self.ctx[key].outputs.output_parameters["total_energy"]
            second_last_energy = self.ctx[keys[-2]].outputs.output_parameters[
                "total_energy"
            ]
            energy_diff_per_atom = (
                abs(last_energy - second_last_energy)
                / self.ctx[key].outputs.output_parameters["num_ions"]
            )
            if energy_diff_per_atom < self.inputs.energy_error:
                self.report(
                    f"K-point spacing converged at {self.ctx.kspacings[i - 1]} A-1"
                )
                self.ctx.converged_kspacing = orm.Float(self.ctx.kspacings[i - 1])
                self.ctx.kspacing_converged = True
                return
        self.ctx.coarse_kspacing = self.ctx.fine_kspacing
        if self.ctx.coarse_kspacing <= 0.02:
            self.report(
                "K-point spacing too low to decrease further. Ending convergence test."
            )
            self.ctx.converged_kspacing = kspacing
            self.ctx.kspacing_converged = True
        else:
            self.ctx.fine_kspacing -= 0.02
        self.report(
            "K-point spacing not converged. Decreasing lower limit by 0.02 A-1."
        )

    def run_supercell_conv(self):
        """Run parallel supercell convergence calculations with the supercell length range and step provided"""
        inputs = self.ctx.inputs
        parameters = deepcopy(self.ctx.parameters)
        parameters.update(
            {
                "task": "phonon",
                "phonon_fine_method": "supercell",
                "cut_off_energy": self.inputs.initial_pwcutoff.value,
            }
        )
        inputs.calc.parameters = parameters
        current_structure = inputs.calc.structure
        seekpath_data = seekpath_analysis(current_structure)
        inputs.calc.phonon_fine_kpoints = seekpath_data["kpoints"]
        inputs.calc.structure = seekpath_data["prim_cell"]
        inputs.kpoints_spacing = self.inputs.coarse_kspacing.value
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
            key = f"supercell_{length}"
            self.to_context(**{key: running})
        self.report("Running supercell convergence calculations")

    def analyse_supercell_conv(self):
        """Analyse previous supercell convergence calculations"""
        keys = []
        for i, length in enumerate(self.ctx.supercell_lengths):
            key = f"supercell_{length}"
            if key in self.ctx:
                keys.append(key)
                if len(keys) == 1:
                    continue
                last_phonon_file = (
                    self.ctx[key]
                    .called[-1]
                    .outputs.retrieved.get_object_content("aiida.phonon")
                )
                with TemporaryDirectory() as temp:
                    with open(f"{temp}/last.phonon", "x") as file:
                        file.write(last_phonon_file)
                    last_phonon_data = PhononParser(open(f"{temp}/last.phonon"))
                    last_freqs = np.array(last_phonon_data.frequencies)
                    with open(f"{temp}/second_last.phonon", "x") as file:
                        file.write(last_phonon_file)
                    second_last_phonon_data = PhononParser(
                        open(f"{temp}/second_last.phonon")
                    )
                    second_last_freqs = np.array(second_last_phonon_data.frequencies)
                percentage_errors = (
                    np.absolute((second_last_freqs - last_freqs)) / last_freqs
                )
                mean_percentage_error = np.mean(percentage_errors)
                if mean_percentage_error < (self.inputs.frequency_error / 100):
                    self.report(
                        f"Supercell converged at {self.ctx.supercell_lengths[i - 1]} Angstroms"
                    )
                    self.ctx.supercell_converged = True
                    self.ctx.converged_supercell = orm.ArrayData()
                    self.ctx.converged_supercell.set_array(
                        "matrix",
                        np.array(
                            self.ctx[keys[-2]].inputs.calc.parameters[
                                "phonon_supercell_matrix"
                            ]
                        ),
                    )
                    return
        self.ctx.initial_supercell_length = self.ctx.final_supercell
        self.ctx.final_supercell_length += 5
        self.report(
            "Supercell not converged. Increasing upper length limit by 5 Angstroms"
        )

    def results(self):
        """Add converged plane-wave cutoff, k-point spacing, and supercell matrix to WorkChain outputs"""
        if self.inputs.converge_pwcutoff:
            self.out("converged_pwcutoff", self.ctx.converged_pwcutoff)
        if self.inputs.converge_kspacing:
            self.out("converged_kspacing", self.ctx.converged_kspacing)
        if self.inputs.converge_supercell:
            self.out("converged_supercell", self.ctx.converged_supercell)
