"""
Module for Defects WorkChain
"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import WorkChain, calcfunction, while_, if_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.relax import CastepRelaxWorkChain
from aiida_castep_addons.utils import add_metadata
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from doped.generation import DefectsGenerator
from doped import analysis, thermodynamics
from monty.serialization import dumpfn, loadfn
from pymatgen.io.ase import AseAtomsAdaptor


@calcfunction
def generate_defects(structure, doped_settings, prefix):
    """Use Doped to generate defect structures"""
    prim_cell = structure.get_pymatgen()
    defect_gen = DefectsGenerator(prim_cell, **doped_settings)
    with TemporaryDirectory() as temp:
        defect_gen.to_json(f"{temp}/{prefix.value}_defects_generator.json")
        defects_generator = orm.SinglefileData(
            f"{temp}/{prefix.value}_defects_generator.json"
        )
    return defects_generator


@calcfunction
def defect_analysis(
    keys,
    defects_generator,
    corrections,
    defect_metadata,
    chempots,
    chempot_limit,
    prefix,
    **kwargs,
):
    """Use defect relaxation output data to calculate and plot defect formation energies"""
    defect_dict = {}
    for i, key in enumerate(keys):
        try:
            pmg_structure = kwargs[f"structure_{i}"].get_pymatgen()
            final_energy = kwargs[f"out_params_{i}"]["total_energy"]
        except:
            continue
        cse = ComputedStructureEntry(
            pmg_structure,
            final_energy,
            composition=pmg_structure.composition,
            entry_id=key,
        )
        if key == "bulk":
            bulk_supercell = pmg_structure
            bulk_entry = cse
            continue
        with defects_generator.as_path() as defects_path:
            defect_gen = DefectsGenerator.from_json(defects_path)
        defect_entries = defect_gen.defect_entries
        entry = defect_entries[key]
        entry.bulk_entry = bulk_entry
        entry.sc_entry = cse
        defect_structures = analysis.defect_from_structures(
            bulk_supercell, pmg_structure, return_all_info=True
        )
        entry.corrections = corrections.get_dict()
        band_gap = defect_metadata.get("gap", 0)
        calculation_metadata = defect_metadata.get_dict()
        calculation_metadata.update(
            {
                "defect_structure": pmg_structure,
                "guessed_initial_defect_structure": defect_structures[5],
                "unrelaxed_defect_structure": defect_structures[6],
                "final_defect_structure": pmg_structure,
            }
        )
        entry.calculation_metadata = calculation_metadata
        defect_dict[key] = entry
    with chempots.as_path() as chempots_path:
        defect_thermo = thermodynamics.DefectThermodynamics(
            defect_dict, loadfn(chempots_path)
        )
    formation_energy_df = defect_thermo.get_formation_energies(
        limit=chempot_limit.value
    )
    with TemporaryDirectory() as temp:
        dumpfn(defect_thermo, fn=f"{temp}/{prefix.value}_defect_thermo.json")
        defect_thermodynamics = orm.SinglefileData(
            f"{temp}/{prefix.value}_defect_thermo.json"
        )
        formation_energy_df.to_csv(f"{temp}/{prefix.value}_formation_energies.csv")
        formation_energies = orm.SinglefileData(
            f"{temp}/{prefix.value}_formation_energies.csv"
        )
        if band_gap > 0:
            def_plot = defect_thermo.plot(limit=chempot_limit.value)
            def_plot.savefig(
                f"{temp}/{prefix.value}_defect_plot.pdf", bbox_inches="tight"
            )
            plt.close("all")
            defect_plot = orm.SinglefileData(f"{temp}/{prefix.value}_defect_plot.pdf")
            return {
                "defect_thermodynamics": defect_thermodynamics,
                "formation_energies": formation_energies,
                "defect_plot": defect_plot,
            }
        else:
            return {
                "defect_thermodynamics": defect_thermodynamics,
                "formation_energies": formation_energies,
            }


class CastepDefectsWorkChain(WorkChain):
    """
    WorkChain to generate defect structures and calculate defect formation energies
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepDefectsWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepRelaxWorkChain)
        spec.input(
            "defect_metadata",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="A dictionary of metadata needed for Doped defect analysis. Must include ",
            required=False,
            default=lambda: orm.Dict(
                dict={
                    "dielectric": 0,
                    "cbm": 0,
                    "vbm": 0,
                    "gap": 0,
                    "num_elec_cbm": 0,
                    "num_hole_vbm": 0,
                    "bandfilling_meta": {
                        "num_elec_cbm": 0,
                        "num_hole_vbm": 0,
                        "potalign": 0,
                        "bandfilling_correction": 0,
                    },
                    "is_compatible": True,
                    "phasediagram_meta": {
                        "vbm": 0,
                        "gap": 0,
                    },
                }
            ),
        )
        spec.input(
            "doped_settings",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Settings for Doped defect structure generation (optional, none by default)",
            required=False,
            default=lambda: orm.Dict(dict={}),
        )
        spec.input(
            "corrections",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Corrections for defects (none by default)",
            required=False,
            default=lambda: orm.Dict(dict={}),
        )
        spec.input(
            "chempots",
            valid_type=orm.SinglefileData,
            serializer=to_aiida_type,
            help="The chemical potentials as a json file",
            required=True,
        )
        spec.input(
            "chempot_limit",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The chemical potential limit to be used for defect analysis",
            required=True,
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
            "relaxed_structures",
            valid_type=orm.List,
            help="A list of the relaxed defect structures",
            required=True,
        )
        spec.output(
            "defects_generator",
            valid_type=orm.SinglefileData,
            help="The DefectsGenerator object used for defect generation as a json file",
            required=True,
        )
        spec.output(
            "formation_energies",
            valid_type=orm.SinglefileData,
            help="A table of formation energies as a csv file",
            required=True,
        )
        spec.output(
            "defect_thermodynamics",
            valid_type=orm.SinglefileData,
            help="The DefectThermodynamics object as a json file",
            required=True,
        )
        spec.output(
            "defect_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the defect transition level diagram",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            cls.relax_defects,
            cls.analyse_defects,
            cls.results,
        )

    def setup(self):
        """Initialise internal variables and generate defect structures"""
        self.ctx.inputs = self.exposed_inputs(CastepRelaxWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.prefix = self.inputs.get(
            "file_prefix",
            f"{self.ctx.inputs.structure.get_formula()}_{self.ctx.parameters['xc_functional']}",
        )
        defects_generator = generate_defects(
            self.inputs.structure, self.inputs.doped_settings, self.ctx.prefix
        )
        self.ctx.defects_generator = defects_generator
        with defects_generator.as_path() as defects_path:
            defect_gen = DefectsGenerator.from_json(defects_path)
        defect_entries = defect_gen.defect_entries
        if self.inputs.defect_metadata.get("gap", 0) == 0:
            defect_keys = list(defect_entries.keys())
            for key in defect_keys:
                if key[-1] != "0":
                    defect_entries.pop(key)
        charges = {}
        keys = ["bulk"]
        keys += list(defect_entries.keys())
        bulk_supercell = defect_entries[keys[1]].bulk_supercell
        ase_bulk = AseAtomsAdaptor.get_atoms(bulk_supercell)
        structures = {"bulk": orm.StructureData(ase=ase_bulk)}
        for key in keys:
            if key == "bulk":
                continue
            charges[key] = defect_entries[key].charge_state
            defect_supercell = defect_entries[key].defect_supercell
            ase_defect = AseAtomsAdaptor.get_atoms(defect_supercell)
            structures[key] = orm.StructureData(ase=ase_defect)
        self.ctx.defect_keys = orm.List(list=keys)
        self.ctx.defect_charges = orm.Dict(dict=charges)
        self.ctx.structures = structures

    def relax_defects(self):
        """Relax the defect structures"""
        inputs = self.ctx.inputs
        relax_parameters = deepcopy(self.ctx.parameters)
        relax_parameters["task"] = "geometryoptimization"
        for key in self.ctx.defect_keys:
            inputs.structure = self.ctx.structures[key]
            if key != "bulk":
                relax_parameters["charge"] = self.ctx.defect_charges[key]
            inputs.calc.parameters = relax_parameters
            running = self.submit(CastepRelaxWorkChain, **inputs)
            self.to_context(**{key: running})
        self.report("Running relaxations on defect structures")

    def analyse_defects(self):
        """Analyse the relaxed defect structures"""
        self.ctx.relaxed_structures = orm.List(list=[])
        kwargs = {}
        for i, key in enumerate(self.ctx.defect_keys):
            if self.ctx[key].is_finished_ok:
                structure = self.ctx[key].outputs.output_structure
                self.ctx.relaxed_structures.append(structure.uuid)
                output_parameters = self.ctx[key].outputs.output_parameters
                kwargs[f"out_params_{i}"] = output_parameters
                kwargs[f"structure_{i}"] = structure
            else:
                self.ctx.relaxed_structures.append("failed")
        outputs = defect_analysis(
            self.ctx.defect_keys,
            self.ctx.defects_generator,
            self.inputs.corrections,
            self.inputs.defect_metadata,
            self.inputs.chempots,
            self.inputs.chempot_limit,
            orm.Str(self.ctx.prefix),
            **kwargs,
        )
        self.ctx.defect_thermodynamics = outputs["defect_thermodynamics"]
        self.ctx.formation_energies = outputs["formation_energies"]
        self.ctx.defect_plot = add_metadata(
            outputs["defect_plot"],
            orm.Str(f"{self.ctx.prefix}_defect_plot.pdf"),
            orm.Str(self.ctx.inputs.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )

    def results(self):
        """Add the relaxed structures, defect thermodynamics, formation energies and the defect plot to WorkChain outputs"""
        self.out("relaxed_structures", self.ctx.relaxed_structures)
        self.out("defects_generator", self.ctx.defects_generator)
        self.out("defect_thermodynamics", self.ctx.defect_thermodynamics)
        self.out("formation_energies", self.ctx.formation_energies)
        self.out("defect_plot", self.ctx.defect_plot)
