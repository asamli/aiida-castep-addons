"""
Module for Competing Phases WorkChain
"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
from aiida.engine import WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.relax import CastepRelaxWorkChain
from aiida_castep_addons.workflows.converge import CastepConvergeWorkChain
from aiida_castep_addons.utils import add_metadata
from pymatgen.core.composition import Composition
from doped.chemical_potentials import (
    CompetingPhases,
    ExtrinsicCompetingPhases,
    CompetingPhasesAnalyzer,
    _calculate_formation_energies,
    combine_extrinsic,
)
from monty.serialization import dumpfn
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram


@calcfunction
def generate_competing_phases(
    chem_formula, extrinsic_species, doped_settings, extrinsic_settings
):
    """Use Doped to generate defect structures"""
    competing_phases = CompetingPhases(chem_formula.value, **doped_settings)
    entries = competing_phases.entries
    if extrinsic_species:
        extrinsic_phases = ExtrinsicCompetingPhases(
            chem_formula.value, extrinsic_species.get_list(), **extrinsic_settings
        )
        entries += extrinsic_phases.entries
    phase_entries = {}
    names = orm.List()
    molecules = orm.List()
    for i, entry in enumerate(entries):
        names.append(entry.name)
        pmg_structure = entry.structure
        pmg_structure.add_oxidation_state_by_guess()
        phase_entries[f"{entry.name}_{i}_structure"] = orm.StructureData(
            pymatgen=pmg_structure
        )
        molecules.append(entry.data["molecule"])
    phase_entries["names"] = names
    phase_entries["molecules"] = molecules
    return phase_entries


@calcfunction
def competing_phases_analysis(
    entry_names, chem_formula, extrinsic_species, prefix, **kwargs
):
    """Use Doped and competing phase relaxation output data to calculate chemical potential limits and plot a phase diagram"""
    # Save data in a csv file
    elemental_energies = {}
    data = []
    extrinsic_species = []
    for i, name in enumerate(entry_names):
        structure = kwargs[f"{name}_{i}_relaxed_structure"].get_pymatgen()
        out_params = kwargs[f"{name}_{i}_out_params"]
        total_energy = out_params["total_energy"]
        energy_per_atom = total_energy / out_params["num_ions"]
        red_formula_and_factor = structure.composition.get_reduced_formula_and_factor()
        phase_formula = red_formula_and_factor[0]
        num_formula_units = red_formula_and_factor[1]
        energy_per_fu = total_energy / num_formula_units
        elements = structure.elements
        if len(elements) == 1:
            element = elements[0].name
            if element not in elemental_energies:
                elemental_energies[element] = energy_per_atom
                if element not in Composition(chem_formula.value):
                    extrinsic_species.append(element)
                elif energy_per_atom < elemental_energies[element]:
                    elemental_energies[element] = energy_per_atom

        d = {
            "formula": phase_formula,
            "energy": total_energy,
            "energy_per_atom": energy_per_atom,
            "energy_per_fu": energy_per_fu,
        }
        data.append(d)
    formation_energy_df = _calculate_formation_energies(data, elemental_energies)
    with TemporaryDirectory() as temp:
        formation_energy_df.to_csv(
            f"{temp}/{prefix.value}_competing_phase_energies.csv", index=False
        )
        formation_energies = orm.SinglefileData(
            f"{temp}/{prefix.value}_competing_phase_energies.csv"
        )

        # Read formation energies from csv, calculate potential limits
        if extrinsic_species:
            all_chempots = []
            for i, _ in enumerate(extrinsic_species):
                cpa = CompetingPhasesAnalyzer(
                    chem_formula.value, extrinsic_species=extrinsic_species[i]
                )
                cpa.from_csv(f"{temp}/{prefix.value}_competing_phase_energies.csv")
                all_chempots.append(cpa.chem_limits)
            if len(all_chempots) == 1:
                dumpfn(cpa.chem_limits, f"{temp}/{prefix.value}_chempots.json")
                chempots = orm.SinglefileData(f"{temp}/{prefix.value}_chempots.json")
                cpd = ChemicalPotentialDiagram(cpa.intrinsic_phase_diagram.entries)
                plot = cpd.get_plot()
                plot.write_image(f"{temp}/{prefix.value}_phase_diagram.pdf")
                phase_diagram_plot = orm.SinglefileData(
                    f"{temp}/{prefix.value}_phase_diagram.pdf"
                )
            else:
                for i in range(0, len(all_chempots) - 1):
                    combined_chempots = combine_extrinsic(
                        all_chempots[i], all_chempots[i + 1], extrinsic_species[i + 1]
                    )
                dumpfn(combined_chempots, f"{temp}/{prefix.value}_chempots.json")
                chempots = orm.SinglefileData(f"{temp}/{prefix.value}_chempots.json")
                cpd = ChemicalPotentialDiagram(cpa.intrinsic_phase_diagram.entries)
                plot = cpd.get_plot()
                plot.write_image(f"{temp}/{prefix.value}_phase_diagram.pdf")
                phase_diagram_plot = orm.SinglefileData(
                    f"{temp}/{prefix.value}_phase_diagram.pdf"
                )
        else:
            cpa = CompetingPhasesAnalyzer(chem_formula.value)
            cpa.from_csv(f"{temp}/{prefix.value}_competing_phase_energies.csv")
            dumpfn(cpa.chem_limits, f"{temp}/{prefix.value}_chempots.json")
            chempots = orm.SinglefileData(f"{temp}/{prefix.value}_chempots.json")
            cpd = ChemicalPotentialDiagram(cpa.intrinsic_phase_diagram.entries)
            plot = cpd.get_plot()
            plot.write_image(f"{temp}/{prefix.value}_phase_diagram.pdf")
            phase_diagram_plot = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.pdf"
            )
    return {
        "formation_energies": formation_energies,
        "chempots": chempots,
        "phase_diagram_plot": phase_diagram_plot,
    }


class CastepCompetingPhasesWorkChain(WorkChain):
    """
    WorkChain to calculate chemical potential limits for competing phases (needed for defect calculations)
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepCompetingPhasesWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepRelaxWorkChain)
        spec.expose_inputs(CastepConvergeWorkChain, namespace="converge")
        spec.input(
            "doped_settings",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Settings for Doped competing phase generation (optional, e_above_hull=0 by default)",
            required=False,
            default=lambda: orm.Dict(dict={"e_above_hull": 0}),
        )
        spec.input(
            "extrinsic_species",
            valid_type=orm.List,
            serializer=to_aiida_type,
            help="A list of extrinsic species or dopants (optional, none by default)",
            required=False,
            default=lambda: orm.List(),
        )
        spec.input(
            "extrinsic_settings",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Settings for Doped extrinsic competing phase generation (optional, e_above_hull=0 by default)",
            required=False,
            default=lambda: orm.Dict(dict={"e_above_hull": 0}),
        )
        spec.input(
            "file_prefix",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The prefix for the names of output files",
            required=False,
        )

        # The outputs
        spec.output(
            "formation_energies",
            valid_type=orm.SinglefileData,
            help="Formation energies as a csv file",
            required=True,
        )
        spec.output(
            "chemical_potentials",
            valid_type=orm.SinglefileData,
            help="Chemical potential limits as a json file",
            required=True,
        )
        spec.output(
            "phase_diagram_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the mixing free energies and enthalpies for each composition",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            cls.converge_competing_phases,
            cls.relax_competing_phases,
            cls.analyse_competing_phases,
            cls.results,
        )

    def setup(self):
        """Initialise internal variables and generate defect structures"""
        self.ctx.converge_inputs = self.exposed_inputs(
            CastepConvergeWorkChain, namespace="converge"
        )
        self.ctx.relax_inputs = self.exposed_inputs(CastepRelaxWorkChain)
        self.ctx.converge_parameters = (
            self.ctx.converge_inputs.calc.parameters.get_dict()
        )
        self.ctx.relax_parameters = self.ctx.relax_inputs.calc.parameters.get_dict()
        self.ctx.prefix = self.inputs.get(
            "file_prefix",
            f"{self.ctx.relax_inputs.structure.get_formula()}_{self.ctx.relax_parameters['xc_functional']}",
        )
        self.ctx.chem_formula = orm.Str(
            self.ctx.relax_inputs.structure.get_formula("count_compact")
        )
        self.ctx.phases = generate_competing_phases(
            self.ctx.chem_formula,
            self.inputs.extrinsic_species,
            self.inputs.doped_settings,
            self.inputs.extrinsic_settings,
        )

    def converge_competing_phases(self):
        """Converge the k-points of the competing phases"""
        inputs = self.ctx.converge_inputs
        converge_parameters = deepcopy(self.ctx.converge_parameters)
        converge_parameters["task"] = "singlepoint"
        inputs.calc.parameters = converge_parameters
        converge_settings = deepcopy(inputs.converge_settings.get_dict())
        converge_settings.update(
            {
                "converge_pwcutoff": False,
                "pwcutoff_end": converge_parameters["cut_off_energy"],
            }
        )
        inputs.converge_settings = converge_settings
        for i, name in enumerate(self.ctx.phases["names"]):
            if self.ctx.phases["molecules"][i] == True:
                continue
            else:
                key = f"{name}_{i}_converge"
                inputs.calc.structure = self.ctx.phases[f"{name}_{i}_structure"]
                running = self.submit(CastepConvergeWorkChain, **inputs)
                self.to_context(**{key: running})
        self.report("Running convergence tests on competing phases")

    def relax_competing_phases(self):
        """Relax the competing phases with converged settings"""
        inputs = self.ctx.relax_inputs
        relax_parameters = deepcopy(self.ctx.relax_parameters)
        relax_parameters["task"] = "geometryoptimization"
        for i, name in enumerate(self.ctx.phases["names"]):
            if self.ctx.phases["molecules"][i] == True:
                kpoints = orm.KpointsData()
                kpoints.set_kpoints_mesh((1, 1, 1))
                inputs.calc.kpoints = kpoints
            else:
                converge_key = f"{name}_{i}_converge"
                inputs.base.kpoints_spacing = self.ctx[
                    converge_key
                ].outputs.converged_kspacing.value
            key = f"{name}_{i}_relax"
            inputs.structure = self.ctx.phases[f"{name}_{i}_structure"]
            inputs.calc.parameters = relax_parameters
            running = self.submit(CastepRelaxWorkChain, **inputs)
            self.to_context(**{key: running})
        self.report("Running relaxations on competing phases")

    def analyse_competing_phases(self):
        """Analyse the relaxed structures"""
        self.ctx.relaxed_structures = orm.List(list=[])
        kwargs = {}
        for i, name in enumerate(self.ctx.phases["names"]):
            key = f"{name}_{i}_relax"
            structure = self.ctx[key].outputs.output_structure
            kwargs[f"{name}_{i}_relaxed_structure"] = structure
            output_parameters = self.ctx[key].outputs.output_parameters
            kwargs[f"{name}_{i}_out_params"] = output_parameters
        outputs = competing_phases_analysis(
            self.ctx.phases["names"],
            self.ctx.chem_formula,
            self.inputs.extrinsic_species,
            orm.Str(self.ctx.prefix),
            **kwargs,
        )
        self.ctx.formation_energies = outputs["formation_energies"]
        self.ctx.chempots = outputs["chempots"]
        self.ctx.phase_diagram_plot = add_metadata(
            outputs["phase_diagram_plot"],
            orm.Str(f"{self.ctx.prefix}_phase_diagram.pdf"),
            orm.Str(self.inputs.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )

    def results(self):
        """Add the relaxed structures, mixing energies and the mixing energy plot to WorkChain outputs"""
        self.out("formation_energies", self.ctx.formation_energies)
        self.out("chemical_potentials", self.ctx.chempots)
        self.out("phase_diagram_plot", self.ctx.phase_diagram_plot)
