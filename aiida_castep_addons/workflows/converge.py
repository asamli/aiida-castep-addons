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
def check_pwcutoff_conv(pwcutoffs, energy_tolerance, **kwargs):
    """Check if the total energy per atom is converged with respect to the plane-wave energy cutoff"""
    is_converged = orm.Bool()
    converged_pwcutoff = orm.Int()
    for i in range(len(pwcutoffs)):
        if i == 0:
            continue
        last_pwcutoff = pwcutoffs[i]
        second_last_pwcutoff = pwcutoffs[i - 1]
        if last_pwcutoff == second_last_pwcutoff:
            continue
        last_energy = kwargs[f"out_params_{i}"]["total_energy"]
        second_last_energy = kwargs[f"out_params_{i-1}"]["total_energy"]
        energy_diff_per_atom = (
            abs(last_energy - second_last_energy)
            / kwargs[f"out_params_{i}"]["num_ions"]
        )
        if energy_diff_per_atom < energy_tolerance:
            is_converged = orm.Bool(True)
            converged_pwcutoff = orm.Int(pwcutoffs[i - 1])
            break

    return {"is_converged": is_converged, "converged_pwcutoff": converged_pwcutoff}


@calcfunction
def check_kspacing_conv(kspacings, kgrids, energy_tolerance, **kwargs):
    """Check if the total energy per atom is converged with respect to the k-point grid spacing"""
    is_converged = orm.Bool()
    converged_kspacing = orm.Float()
    for i in range(len(kspacings)):
        if i == 0:
            continue
        last_mesh = kgrids[i]
        second_last_mesh = kgrids[i - 1]
        if last_mesh == second_last_mesh:
            continue
        last_energy = kwargs[f"out_params_{i}"]["total_energy"]
        second_last_energy = kwargs[f"out_params_{i-1}"]["total_energy"]
        energy_diff_per_atom = (
            abs(last_energy - second_last_energy)
            / kwargs[f"out_params_{i}"]["num_ions"]
        )
        if energy_diff_per_atom < energy_tolerance:
            is_converged = orm.Bool(True)
            converged_kspacing = orm.Float(kspacings[i - 1])
            break

    return {"is_converged": is_converged, "converged_kspacing": converged_kspacing}


@calcfunction
def check_supercell_conv(matrices, frequency_tolerance, kpoints, prefix, **kwargs):
    """Check if the phonon frequencies are converged with respect to the supercell size and
    plot phonon dispersions for different supercell sizes with sumo and pymatgen"""
    is_converged = orm.Bool()
    converged_supercell = orm.ArrayData()
    converged_supercell_label = orm.Str()
    supercell_labels = [
        f"{matrices[i][0][0]}x{matrices[i][1][2]}x{matrices[i][2][4]}"
        for i in range(len(matrices))
    ]
    with TemporaryDirectory() as temp:
        for i in range(len(matrices)):
            # Checking for convergence
            file = kwargs[f"retrieved_{i}"].get_object_content("aiida.phonon")
            with open(f"{temp}/{i}.phonon", "x") as phonon_file:
                phonon_file.write(file)
            if i == 0:
                continue
            last_phonon_data = PhononParser(open(f"{temp}/{i}.phonon"))
            last_freqs = np.array(last_phonon_data.frequencies)
            second_last_phonon_data = PhononParser(open(f"{temp}/{i-1}.phonon"))
            second_last_freqs = np.array(second_last_phonon_data.frequencies)

            percentage_errors = (
                np.absolute((second_last_freqs - last_freqs)) / last_freqs
            )
            mean_percentage_error = np.mean(percentage_errors)
            if mean_percentage_error < (
                frequency_tolerance / 100
            ) and is_converged == orm.Bool(False):
                is_converged = orm.Bool(True)
                converged_supercell.set_array("matrix", np.array(matrices[i - 1]))
                converged_supercell_label = orm.Str(supercell_labels[i - 1])

            # Plotting the phonon band structure with sumo and pymatgen
            qpoints = kpoints.get_kpoints()
            parsed_qpoints = np.array(last_phonon_data.qpoints)
            frequencies = np.array(last_phonon_data.frequencies) / 33.36
            bands = np.transpose(frequencies)
            lattice = Lattice(last_phonon_data.cell)
            rec_lattice = lattice.reciprocal_lattice
            pmg_structure = last_phonon_data.structure
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
        supercell_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_supercell_convergence.pdf"
        )
        plt.close("all")
        return {
            "is_converged": is_converged,
            "converged_supercell": converged_supercell,
            "converged_supercell_label": converged_supercell_label,
            "supercell_plot": supercell_plot,
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
        kwargs = {}
        pwcutoffs = []
        for i, wc in enumerate(self.ctx.pwcutoff_calcs):
            pwcutoffs.append(wc.inputs.calc.parameters["cut_off_energy"])
            kwargs[f"out_params_{i}"] = wc.outputs.output_parameters
        pwcutoff_conv = check_pwcutoff_conv(
            orm.List(list=pwcutoffs), orm.Float(self.ctx.energy_tolerance), **kwargs
        )
        if pwcutoff_conv["is_converged"]:
            self.ctx.converged_pwcutoff = pwcutoff_conv["converged_pwcutoff"]
            self.report(
                f"Plane-wave energy cutoff converged at {self.ctx.converged_pwcutoff.value} eV"
            )
            self.ctx.pwcutoff_end = self.ctx.converged_pwcutoff.value
            self.ctx.pwcutoff_converged = True
        else:
            self.ctx.pwcutoff_start = self.ctx.pwcutoff_end + self.ctx.pwcutoff_step
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
        kwargs = {}
        kspacings = []
        kgrids = []
        for i, wc in enumerate(self.ctx.kspacing_calcs):
            kspacings.append(wc.inputs.kpoints_spacing)
            kgrids.append(wc.called[-1].inputs.kpoints.get_kpoints_mesh())
            kwargs[f"out_params_{i}"] = wc.outputs.output_parameters
        kspacing_conv = check_kspacing_conv(
            orm.List(list=kspacings),
            orm.List(list=kgrids),
            orm.Float(self.ctx.energy_tolerance),
            **kwargs,
        )
        if kspacing_conv["is_converged"]:
            self.ctx.converged_kspacing = kspacing_conv["converged_kspacing"]
            self.report(
                f"K-point spacing converged at {self.ctx.converged_kspacing.value} A-1"
            )
            self.ctx.kspacing_end = self.ctx.converged_kspacing
            self.ctx.kspacing_converged = True
        else:
            self.ctx.kspacing_start = self.ctx.kspacing_end - self.ctx.kspacing_step
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
        kwargs = {}
        matrices = []
        for i, wc in enumerate(self.ctx.supercell_calcs):
            matrices.append(wc.inputs.calc.parameters["phonon_supercell_matrix"])
            kwargs[f"retrieved_{i}"] = wc.called[-1].outputs.retrieved
        supercell_conv = check_supercell_conv(
            orm.List(list=matrices),
            orm.Float(self.ctx.frequency_tolerance),
            self.ctx.kpoints,
            orm.Str(self.ctx.prefix),
            **kwargs,
        )
        if supercell_conv["is_converged"]:
            self.report(
                f"Supercell size converged at {supercell_conv['converged_supercell_label'].value}"
            )
            self.ctx.supercell_converged = True
            self.ctx.converged_supercell = supercell_conv["converged_supercell"]
            supercell_plot = supercell_conv["supercell_plot"]
            self.ctx.supercell_plot = add_metadata(
                supercell_plot,
                orm.Str(f"{self.ctx.prefix}_supercell_convergence.pdf"),
                orm.Str(self.ctx.inputs.calc.structure.get_formula()),
                orm.Str(self.uuid),
                orm.Str(self.inputs.metadata.get("label", "")),
                orm.Str(self.inputs.metadata.get("description", "")),
            )
        else:
            self.ctx.supercell_start = self.ctx.supercell_end
            self.ctx.supercell_end += 10
            self.report(
                "Supercell size not converged. Increasing upper length limit by 10 Angstroms."
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
