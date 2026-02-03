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
from doped.chemical_potentials import (
    CompetingPhases,
    CompetingPhasesAnalyzer,
)
from monty.serialization import dumpfn
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.entries.computed_entries import ComputedStructureEntry


@calcfunction
def generate_competing_phases(chem_formula, extrinsic_species, doped_settings):
    """Use Doped to generate defect structures"""
    competing_phases = CompetingPhases(
        chem_formula.value, extrinsic=extrinsic_species, **doped_settings
    )
    entries = competing_phases.entries
    phase_entries = {}
    names = orm.List()
    molecules = orm.List()
    for i, entry in enumerate(entries):
        names.append(entry.name)
        pmg_structure = entry.structure
        ase_structure = AseAtomsAdaptor.get_atoms(pmg_structure)
        phase_entries[f"{entry.name}_{i}_structure"] = orm.StructureData(
            ase=ase_structure
        )
        molecules.append(entry.data["molecule"])
    phase_entries["names"] = names
    phase_entries["molecules"] = molecules
    return phase_entries


@calcfunction
def competing_phases_analysis(entry_names, chem_formula, prefix, **kwargs):
    """Use Doped and competing phase relaxation output data to calculate chemical potential limits and plot a phase diagram"""
    # Create a list of new pymatgen computed structure entries
    entry_list = []
    for i, name in enumerate(entry_names):
        try:
            structure = kwargs[f"{entry_names[i]}_{i}_relaxed_structure"].get_pymatgen()
            out_params = kwargs[f"{entry_names[i]}_{i}_out_params"]
        except:
            continue
        total_energy = out_params["total_energy"]
        composition = structure.composition
        entry = ComputedStructureEntry(
            structure, total_energy, composition=composition, entry_id=name
        )
        entry_list.append(entry)

    with TemporaryDirectory() as temp:
        # Reading list of pymatgen entries and calculating potential limits
        cpa = CompetingPhasesAnalyzer(
            chem_formula.value,
            entry_list,
        )
        formation_energy_df = cpa.get_formation_energy_df()
        formation_energy_df.to_csv(
            f"{temp}/{prefix.value}_competing_phase_energies.csv", index=False
        )
        formation_energies = orm.SinglefileData(
            f"{temp}/{prefix.value}_competing_phase_energies.csv"
        )
        dumpfn(cpa.chempots, f"{temp}/{prefix.value}_chempots.json")
        chempots = orm.SinglefileData(f"{temp}/{prefix.value}_chempots.json")
        cpd = ChemicalPotentialDiagram(cpa.phase_diagram.entries)
        plot = cpd.get_plot()
        plot.write_image(f"{temp}/{prefix.value}_phase_diagram.pdf")
        phase_diagram_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_phase_diagram.pdf"
        )
        dumpfn(cpa, f"{temp}/{prefix.value}_cpa.json")
        cpa_file = orm.SinglefileData(f"{temp}/{prefix.value}_cpa.json")
    return {
        "formation_energies": formation_energies,
        "chempots": chempots,
        "phase_diagram_plot": phase_diagram_plot,
        "cpa_file": cpa_file,
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
            default=lambda: orm.Dict(dict={"energy_above_hull": 0}),
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
            help="A phase diagram showing the stability regions for all competing phases",
            required=True,
        )
        spec.output(
            "competing_phases_analyzer",
            valid_type=orm.SinglefileData,
            help="The CompetingPhasesAnalyzer object as a json file",
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
                if self.ctx[converge_key].is_finished_ok:
                    inputs.base.kpoints_spacing = self.ctx[
                        converge_key
                    ].outputs.converged_kspacing.value
                else:
                    continue
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
            try:
                if self.ctx[key].is_finished_ok:
                    structure = self.ctx[key].outputs.output_structure
                    kwargs[f"{name}_{i}_relaxed_structure"] = structure
                    output_parameters = self.ctx[key].outputs.output_parameters
                    kwargs[f"{name}_{i}_out_params"] = output_parameters
                else:
                    self.ctx.relaxed_structures.append("failed")
            except:
                continue
        outputs = competing_phases_analysis(
            self.ctx.phases["names"],
            self.ctx.chem_formula,
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
        self.ctx.cpa_file = outputs["cpa_file"]

    def results(self):
        """Add the formation energies, chemical potentials, phase diagram and the competing phases analyzer to WorkChain outputs"""
        self.out("formation_energies", self.ctx.formation_energies)
        self.out("chemical_potentials", self.ctx.chempots)
        self.out("phase_diagram_plot", self.ctx.phase_diagram_plot)
        self.out("competing_phases_analyzer", self.ctx.cpa_file)
