"""
Module for Alloy WorkChain
"""
from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.relax import CastepRelaxWorkChain
from aiida_castep_addons.utils import add_metadata
from bsym.interface.pymatgen import unique_structure_substitutions
from pymatgen.core.periodic_table import Element

__version__ = "0.0.1"


@calcfunction
def generate_structures(structure, to_substitute, susbtituent, supercell_size):
    """Use Pymatgen and Bsym to generate symmetry-inequivalent configurations at different compositions"""
    supercell = structure.get_pymatgen() * supercell_size
    element = Element(to_substitute)
    num_atoms = supercell.species.count(element)
    structures = {"structure_0_0": orm.StructureData(pymatgen=supercell)}
    xs = [0]
    lens = [1]
    for i in range(1, num_atoms + 1):
        strucs = unique_structure_substitutions(
            supercell,
            to_substitute.value,
            {susbtituent.value: i, to_substitute.value: num_atoms - i},
        )
        lens.append(len(strucs))
        x = i / num_atoms
        xs.append(x)
        for j, struc in enumerate(strucs):
            structures[f"structure_{i}_{j}"] = orm.StructureData(pymatgen=struc)
    structures["xs"] = orm.List(list=xs)
    structures["lens"] = orm.List(list=lens)
    return structures


@calcfunction
def analysis(xs, lens, temperatures, prefix, **kwargs):
    """Use thermodynamics output data to calculate and plot mixing free energies and enthalpies for the lowest energy configuration at each composition"""
    total_energies = []
    min_energies = []
    mixing_entropies = []

    # Storing the minimum energy for each composition in a list and its unique key in lists
    for i in range(len(xs)):
        for j in range(lens[i]):
            key = f"out_params_{i}_{j}"
            total_energy_per_atom = (
                kwargs[key]["total_energy"] / kwargs[key]["num_ions"]
            )
            total_energies.append(total_energy_per_atom)
        min_energies.append(min(total_energies))
        min_energy_index = np.argmin(total_energies)
        min_energy_key = f"{i}_{min_energy_index}"
        total_energies = []

    # Calculating and storing mixing enthalpies and free energies
    mixing_enthalpies = [
        (min_energies[i] - ((1 - xs[i]) * min_energies[0]) - (xs[i] * min_energies[-1]))
        for i in range(0, len(min_energies))
    ]
    for i, x in enumerate(xs):
        if x == 0 or x == 1:
            mixing_entropies.append(0)
        else:
            mixing_entropies.append(
                -8.31 * ((x * np.log(x)) + ((1 - x) * np.log(1 - x)))
            )

    mixing_free_energies = [
        [
            mixing_enthalpies[i] - (t * mixing_entropies[i] / 96000)
            for i, _ in enumerate(xs)
        ]
        for t in temperatures
    ]

    mixing_energies = orm.Dict(
        dict={
            "x_values": xs.get_list(),
            "temperatures": temperatures,
            "mixing_free_energies": mixing_free_energies,
            "mixing_enthalpies": mixing_enthalpies,
        }
    )

    # Plotting the mixing enthalpies and free energies
    labels = [
        (r"$\Delta G_{mix}$" + f" {temperature} K") for temperature in temperatures
    ]
    for i, label in enumerate(labels):
        plt.plot(xs, mixing_free_energies[i], label=label)
    plt.plot(xs, mixing_enthalpies, linestyle="dashed", label=r"$\Delta H_{mix}$")
    plt.xlim(left=0, right=1)
    plt.xlabel("x")
    plt.ylabel("Mixing energy (eV per atom)")
    plt.legend(loc="best")
    with TemporaryDirectory() as temp:
        plt.savefig(
            fname=f"{temp}/{prefix.value}_mixing_energies.pdf", bbox_inches="tight"
        )
        mixing_energy_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_mixing_energies.pdf"
        )

    return {
        "mixing_energies": mixing_energies,
        "mixing_energy_plot": mixing_energy_plot,
    }


class CastepAlloyWorkChain(WorkChain):
    """
    WorkChain to create and relax alloys and plot mixing enthalpies and free energies
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepAlloyWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepRelaxWorkChain)
        spec.input(
            "to_substitute",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The element to substitute in the input structure",
            required=True,
        )
        spec.input(
            "substituent",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The element to be substituted into the input structure",
            required=True,
        )
        spec.input(
            "supercell_matrix",
            valid_type=orm.List,
            serializer=to_aiida_type,
            help="The transformation matrix for the supercell to be used as an array.",
            required=False,
            default=lambda: orm.List(list=[1, 1, 1]),
        )
        spec.input(
            "temperatures",
            valid_type=orm.List,
            serializer=to_aiida_type,
            help="The temperatures to use in K as an array.",
            required=False,
            default=lambda: orm.List(list=[298]),
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
            help="A list of the relaxed symmetry-inequivalent structures for different compositions",
            required=True,
        )
        spec.output(
            "mixing_energies",
            valid_type=orm.Dict,
            help="The mixing free energies and enthalpies for each composition",
            required=True,
        )
        spec.output(
            "mixing_energy_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the mixing free energies and enthalpies for each composition",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            cls.run_relax,
            cls.analyse_calcs,
            cls.results,
        )

    def setup(self):
        """Initialise internal variables and generate symmetry-inequivalent structures for different compositions"""
        self.ctx.inputs = self.exposed_inputs(CastepRelaxWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.prefix = self.inputs.get(
            "file_prefix",
            f"{self.ctx.inputs.structure.get_formula()}_{self.ctx.parameters['xc_functional']}",
        )
        self.ctx.structures = generate_structures(
            self.ctx.inputs.structure,
            self.inputs.to_substitute,
            self.inputs.substituent,
            self.inputs.supercell_matrix,
        )
        self.ctx.xs = self.ctx.structures["xs"]
        self.ctx.lens = self.ctx.structures["lens"]

    def run_relax(self):
        """Relax the symmetry-inequivalent structures for all compositions"""
        inputs = self.ctx.inputs
        relax_parameters = deepcopy(self.ctx.parameters)
        relax_parameters["task"] = "geometryoptimization"
        inputs.calc.parameters = relax_parameters
        for i in range(len(self.ctx.xs)):
            for j in range(self.ctx.lens[i]):
                inputs.structure = self.ctx.structures[f"structure_{i}_{j}"]
                key = f"{i}_{j}_relax"
                running = self.submit(CastepRelaxWorkChain, **inputs)
                self.to_context(**{key: running})
        self.report("Running relaxations on symmetry-inequivalent structures")

    def analyse_calcs(self):
        """Analyse the relaxed structures"""
        self.ctx.relaxed_structures = orm.List(list=[])
        relaxed_structures = []
        kwargs = {}
        for i in range(len(self.ctx.xs)):
            for j in range(self.ctx.lens[i]):
                key = f"{i}_{j}_relax"
                structure = self.ctx[key].outputs.output_structure
                relaxed_structures.append(structure.uuid)
                output_parameters = self.ctx[f"{i}_{j}_relax"].outputs.output_parameters
                kwargs[f"out_params_{i}_{j}"] = output_parameters
            self.ctx.relaxed_structures.append(relaxed_structures)
            relaxed_structures = []
        outputs = analysis(
            self.ctx.xs,
            self.ctx.lens,
            self.inputs.temperatures,
            orm.Str(self.ctx.prefix),
            **kwargs,
        )
        self.ctx.mixing_energies = outputs["mixing_energies"]
        self.ctx.mixing_energy_plot = add_metadata(
            outputs["mixing_energy_plot"],
            orm.Str(f"{self.ctx.prefix}_mixing_energies.pdf"),
            orm.Str(self.ctx.inputs.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )

    def results(self):
        """Add the relaxed structures, mixing energies and the mixing energy plot to WorkChain outputs"""
        self.out("relaxed_structures", self.ctx.relaxed_structures)
        self.out("mixing_energies", self.ctx.mixing_energies)
        self.out("mixing_energy_plot", self.ctx.mixing_energy_plot)
