"""
Module for Magnetic WorkChain
Inspired by the VASP workchain of the same name in aiida-user-addons
"""
from __future__ import absolute_import

from copy import deepcopy

import aiida.orm as orm
import numpy as np
from aiida.engine import WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.relax import CastepRelaxWorkChain
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

__version__ = "0.0.1"


@calcfunction
def enumerate_spins(structure, enum_options):
    """Use Pymatgen to enumerate all magnetic orderings for a structure"""
    enum = MagneticStructureEnumerator(
        structure.get_pymatgen(), **enum_options.get_dict()
    )
    enum_structures = {}
    enum_structures["origins"] = orm.List(list=enum.ordered_structure_origins)
    for i, structure in enumerate(enum.ordered_structures):
        spins = [specie.spin for specie in structure.species]
        enum_structures[f"structure_{i+1}_spins"] = orm.List(list=spins)
        structure.remove_spin()
        enum_structures[f"structure_{i+1}"] = orm.StructureData(pymatgen=structure)
    return enum_structures


@calcfunction
def assemble_enum_data(workchains, origins):
    enum_data = orm.List(list=[])
    for i, wc in enumerate(workchains):
        wc_node = orm.load_node(wc)
        if wc_node.is_finished_ok:
            initial_structure = wc_node.inputs.structure
            final_structure = wc_node.outputs.output_structure
            num_formula_units = final_structure.get_pymatgen().composition.get_reduced_composition_and_factor()[
                1
            ]
            energy = (
                wc_node.outputs.output_parameters["total_energy"] / num_formula_units
            )
            data_dict = {
                "index": i + 1,
                "initial_structure": initial_structure.uuid,
                "origin": origins[i],
                "final_structure": final_structure.uuid,
                "final_energy": energy,
            }
            enum_data.append(data_dict)
    return enum_data


class CastepMagneticWorkChain(WorkChain):
    """
    WorkChain to enumerate spins for magnetic materials
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepMagneticWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepRelaxWorkChain)
        spec.input(
            "enum_options",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Options for the Pymatgen spin enumerator",
            required=False,
        )

        # The outputs
        spec.output(
            "enum_data",
            valid_type=orm.List,
            help="Data for the enumerated structures",
            required=True,
        )
        spec.output(
            "gs_structure",
            valid_type=orm.StructureData,
            help="The ground-state (lowest energy) structure",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            cls.run_relax,
            cls.analyse_relax,
            cls.results,
        )

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepRelaxWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.enum_options = self.inputs.get("enum_options", {})

    def run_relax(self):
        """Run relaxations on enumerated structures from Pymatgen's spin enumerator"""
        inputs = self.ctx.inputs
        parameters = deepcopy(self.ctx.parameters)
        parameters.update(
            {
                "task": "geometryoptimization",
                "spin_polarised": True,
            }
        )
        inputs.calc.parameters = parameters
        self.ctx.enum_structures = enumerate_spins(
            inputs.structure, orm.Dict(dict=self.ctx.enum_options)
        )
        for i in range(len(self.ctx.enum_structures["origins"])):
            key = f"structure_{i+1}"
            inputs.structure = self.ctx.enum_structures[key]
            inputs.calc.update(
                {"settings": {"SPINS": self.ctx.enum_structures[f"{key}_spins"]}}
            )
            running = self.submit(CastepRelaxWorkChain, **inputs)
            self.to_context(**{key: running})
        self.report("Running relaxations on enumerated structures")

    def analyse_relax(self):
        """Analyse the relaxations"""
        workchains = orm.List(list=[])
        origins = self.ctx.enum_structures["origins"]
        for i in range(len(origins)):
            key = f"structure_{i+1}"
            workchains.append(self.ctx[key].uuid)
        self.ctx.enum_data = assemble_enum_data(workchains, origins)
        gs_ordering = min(self.ctx.enum_data, key=lambda x: x["final_energy"])
        self.report(
            f"Data for all successful orderings saved. Ordering {gs_ordering['index']} is the magnetic ground state ({gs_ordering['final_energy']} eV). Returning final relaxed structure."
        )
        self.ctx.gs_structure = orm.load_node(gs_ordering["final_structure"])

    def results(self):
        """Add the enumeration data and lowest energy structure to WorkChain outputs"""
        self.out("enum_data", self.ctx.enum_data)
        self.out("gs_structure", self.ctx.gs_structure)
