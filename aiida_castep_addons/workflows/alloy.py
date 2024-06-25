"""
Module for Alloy WorkChain
"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import WorkChain, calcfunction, if_, while_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.relax import CastepRelaxWorkChain
from bsym.interface.pymatgen import unique_structure_substitutions
from icet import ClusterExpansion, ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import brentq, curve_fit, minimize_scalar

from aiida_castep_addons.utils import add_metadata


@calcfunction
def generate_bsym_structures(
    structure, to_substitute, susbtituent, supercell_matrix, xs
):
    """Use Pymatgen and Bsym to generate symmetry-inequivalent configurations at different compositions"""
    supercell = structure.get_pymatgen() * supercell_matrix
    ase_supercell = AseAtomsAdaptor.get_atoms(supercell)
    try:
        supercell.remove_site_property("kind_name")
    except:
        pass
    element = Element(to_substitute)
    num_atoms = supercell.species.count(element)
    # charged_supercell = structure.get_pymatgen() * supercell_matrix
    # charged_supercell.add_oxidation_state_by_guess()
    structures = {"structure_0_0": orm.StructureData(ase=ase_supercell)}
    degens = [[1]]
    if not xs:
        xs = [0]
        lens = [1]
        for i in range(1, num_atoms + 1):
            strucs = unique_structure_substitutions(
                supercell,
                to_substitute.value,
                {susbtituent.value: i, to_substitute.value: num_atoms - i},
            )
            degens.append([struc.full_configuration_degeneracy for struc in strucs])
            lens.append(len(strucs))
            x = i / num_atoms
            xs.append(x)
            for j, struc in enumerate(strucs):
                # struc.add_oxidation_state_by_guess()
                ase_struc = AseAtomsAdaptor.get_atoms(struc)
                structures[f"structure_{i}_{j}"] = orm.StructureData(ase=ase_struc)
    else:
        xs = xs.get_list()
        if xs[0] != 0:
            xs = [0] + xs
        if xs[-1] != 1:
            xs.append(1)
        lens = [1]
        for i, x in enumerate(xs):
            if i == 0:
                continue
            num_subs = int(x * num_atoms)
            strucs = unique_structure_substitutions(
                supercell,
                to_substitute.value,
                {
                    susbtituent.value: num_subs,
                    to_substitute.value: num_atoms - num_subs,
                },
            )
            degens.append([struc.full_configuration_degeneracy for struc in strucs])
            lens.append(len(strucs))
            for j, struc in enumerate(strucs):
                # struc.add_oxidation_state_by_guess()
                ase_struc = AseAtomsAdaptor.get_atoms(struc)
                structures[f"structure_{i}_{j}"] = orm.StructureData(ase=ase_struc)
    structures["xs"] = orm.List(list=xs)
    structures["lens"] = orm.List(list=lens)
    structures["degens"] = orm.List(list=degens)
    return structures


@calcfunction
def generate_sqs_structures(
    structure, to_substitute, susbtituent, supercell_matrix, xs
):
    """Use ICET and ASE to generate special quasirandom structures at different compositions"""
    chemical_symbols = []
    degens = [[1]]
    num_atoms = 0
    ase_structure = structure.get_ase()
    for i, symbol in enumerate(ase_structure.get_chemical_symbols()):
        if symbol == to_substitute.value:
            chemical_symbols.append([symbol, susbtituent.value])
            num_atoms += 1
        else:
            chemical_symbols.append([symbol])
    supercells = [ase_structure.repeat(supercell_matrix.get_list())]
    num_atoms *= np.prod(supercell_matrix.get_list())
    cs = ClusterSpace(ase_structure, [8.0, 4.0], chemical_symbols)
    structures = {"structure_0_0": orm.StructureData(ase=supercells[0])}
    if not xs:
        xs = [0]
        for i in range(1, num_atoms + 1):
            x = i / num_atoms
            xs.append(x)
            if x < 1:
                target_concentrations = {
                    to_substitute.value: 1 - x,
                    susbtituent.value: x,
                }
                sqs = generate_sqs_from_supercells(
                    cs,
                    supercells=supercells,
                    target_concentrations=target_concentrations,
                )
                structures[f"structure_{i}_0"] = orm.StructureData(ase=sqs)
            else:
                pmg_structure = structure.get_pymatgen()
                pmg_structure.replace_species({to_substitute.value: susbtituent.value})
                ase_structure = AseAtomsAdaptor.get_atoms(pmg_structure)
                structures[f"structure_{i}_0"] = orm.StructureData(ase=ase_structure)
            degens.append([1])
    else:
        xs = xs.get_list()
        if xs[0] != 0:
            xs = [0] + xs
        if xs[-1] != 1:
            xs.append(1)
        for i, x in enumerate(xs):
            if i == 0:
                continue
            elif x < 1:
                target_concentrations = {
                    to_substitute.value: 1 - x,
                    susbtituent.value: x,
                }
                sqs = generate_sqs_from_supercells(
                    cs,
                    supercells=supercells,
                    target_concentrations=target_concentrations,
                )
                structures[f"structure_{i}_0"] = orm.StructureData(ase=sqs)
            else:
                pmg_structure = structure.get_pymatgen()
                pmg_structure.replace_species({to_substitute.value: susbtituent.value})
                ase_structure = AseAtomsAdaptor.get_atoms(pmg_structure)
                structures[f"structure_{i}_0"] = orm.StructureData(ase=ase_structure)
            degens.append([1])
    lens = [1] * len(xs)
    structures["xs"] = orm.List(list=xs)
    structures["lens"] = orm.List(list=lens)
    structures["degens"] = orm.List(list=degens)
    return structures


def cubic(x, a, b, c, d):
    return a * (x**3) + b * (x**2) + c * x + d


def interp_free_energy(x, T, *popt):
    """Use interpolation to calculate the Gibbs free energy for any value of x"""
    enthalpy = cubic(x, *popt)
    if x == 0 or x == 1:
        entropy = 0
    else:
        entropy = -8.31 * ((x * np.log(x)) + ((1 - x) * np.log(1 - x)))
    return enthalpy - (T * entropy / 96000)


def d2free(x, T, *popt):
    """Use finite differences to calculate the 2nd derivative of the free energy"""
    if x == 0 or x == 1:
        return np.inf
    h = 0.00001
    x1 = x - h
    x2 = x + h
    y = interp_free_energy(x, T, *popt)
    y1 = interp_free_energy(x1, T, *popt)
    y2 = interp_free_energy(x2, T, *popt)
    return (y2 - 2.0 * y + y1) / h**2


@calcfunction
def analysis(xs, lens, degens, temperatures, prefix, **kwargs):
    """Use relaxation output data to calculate and plot mixing free energies and enthalpies"""
    total_energies = []
    all_energies = []
    mixing_enthalpies = []
    mixing_entropies = []
    Zs = []
    Z = 0
    kB = 8.617343e-5
    T = 298

    # Storing the minimum energy for each composition in a list
    for i in range(len(xs)):
        for j in range(lens[i]):
            key = f"out_params_{i}_{j}"
            if key in kwargs:
                total_energy_per_atom = (
                    kwargs[key]["total_energy"] / kwargs[key]["num_ions"]
                )
                total_energies.append(total_energy_per_atom)
            else:
                total_energies.append(0.0)
        all_energies.append(total_energies)
        total_energies = []

    # Calculating and storing mixing enthalpies and free energies
    mixing_enthalpies = np.full((len(xs), max(lens)), np.nan)
    for i in range(len(xs)):
        for j in range(lens[i]):
            if all_energies[i][j] == 0.0 or i == 0 or i == len(xs) - 1:
                mixing_enthalpy = 0.0
            else:
                mixing_enthalpy = (
                    all_energies[i][j]
                    - ((1 - xs[i]) * all_energies[0][0])
                    - (xs[i] * all_energies[-1][-1])
                )
            mixing_enthalpies[i][j] = mixing_enthalpy
            Z += degens[i][j] * np.e ** (-mixing_enthalpy / (kB * T))
        Zs.append(Z)
        Z = 0

    boltzmann_probabilities = [
        [
            1 / Zs[i] * degens[i][j] * np.e ** (-mixing_enthalpies[i][j] / (kB * T))
            for j in range(lens[i])
        ]
        for i, _ in enumerate(xs)
    ]
    boltzmann_enthalpies = [
        sum(
            [
                boltzmann_probabilities[i][j] * mixing_enthalpies[i][j]
                for j in range(lens[i])
            ]
        )
        for i, _ in enumerate(xs)
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
            boltzmann_enthalpies[i] - (t * mixing_entropies[i] / 96000)
            for i, _ in enumerate(xs)
        ]
        for t in temperatures
    ]

    mixing_energies = orm.Dict(
        dict={
            "x_values": xs.get_list(),
            "temperatures": temperatures,
            "total_energies": all_energies,
            "boltzmann_enthalpies": boltzmann_enthalpies,
            "mixing_free_energies": mixing_free_energies,
        }
    )

    # Calculating binodal and spinodal points for phase diagram
    with TemporaryDirectory() as temp:
        T0 = 195.0
        dT = 5.0
        xc = 0.5
        x1_bin = 0.0001
        x2_bin = 0.9999
        popt, _ = curve_fit(cubic, xs.get_list(), boltzmann_enthalpies)
        fail_count = 0
        x_diff = x2_bin - x1_bin
        while x_diff > 0.001:
            T = T0 + dT
            try:
                x1_spin = brentq(d2free, x1_bin, xc, args=(T, *popt))
                x2_spin = brentq(d2free, xc, x2_bin, args=(T, *popt))
            except ValueError:
                fail_count += 1
                if fail_count > 10:
                    break
                else:
                    continue
            xc = (x1_spin + x2_spin) / 2.0
            x1 = x1_spin
            x2 = x2_spin

            def f1(x):
                return (
                    (xc - x) * interp_free_energy(x2, T, *popt)
                    + (x2 - xc) * interp_free_energy(x, T, *popt)
                ) / (x2 - x)

            def f2(x):
                return (
                    (xc - x1) * interp_free_energy(x, T, *popt)
                    + (x - xc) * interp_free_energy(x1, T, *popt)
                ) / (x - x1)

            h1 = 1.0
            h2 = 1.0
            k = 0
            while h1 > 1e-6 or h2 > 1e-6:
                res = minimize_scalar(f1, bounds=(x1_bin, x1_spin), method="bounded")
                h1 = abs(x1 - res.x)
                x1 = res.x
                res = minimize_scalar(f2, bounds=(x2_spin, x2_bin), method="bounded")
                h2 = abs(x2 - res.x)
                x2 = res.x
                k += 1
                if k == 10:
                    break
            x1_bin = x1
            x2_bin = x2
            with open(f"{temp}/{prefix.value}_phase_diagram.dat", "a") as output:
                output.write(
                    "%.2f %.5f %.5f %.5f %.5f\n" % (T, x1_bin, x2_bin, x1_spin, x2_spin)
                )
            T0 = T
            x_diff = x2_bin - x1_bin
            fail_count = 0

        try:
            phase_diagram_data = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.dat"
            )
            # Plotting the phase diagram
            T = []
            xb = []
            xsp = []
            with open(f"{temp}/{prefix.value}_phase_diagram.dat", "r") as input_file:
                for line in input_file:
                    data = line.split()
                    T.append(float(data[0]))
                    T.append(float(data[0]))
                    xb.append(float(data[1]))
                    xb.append(float(data[2]))
                    xsp.append(float(data[3]))
                    xsp.append(float(data[4]))
            x_bin = [x for (x, y) in sorted(zip(xb, T))]
            T_bin = [y for (x, y) in sorted(zip(xb, T))]
            x_spin = [x for (x, y) in sorted(zip(xsp, T))]
            T_spin = [y for (x, y) in sorted(zip(xsp, T))]
            plt.plot(x_bin, T_bin, "b", lw=1, label="binodal")
            plt.fill_between(x_bin, T_bin, color="b", alpha=0.3)
            plt.plot(x_spin, T_spin, "r", lw=1, label="spinodal")
            plt.fill_between(x_spin, T_spin, color="r", alpha=0.3)
            plt.xlim(left=0, right=1)
            plt.ylim(bottom=200)
            plt.xlabel("x")
            plt.ylabel("T (K)")
            plt.legend(loc="best")
            plt.savefig(f"{temp}/{prefix.value}_phase_diagram.pdf", bbox_inches="tight")
            phase_diagram_plot = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.pdf"
            )
            plt.close("all")
        except:
            open(f"{temp}/{prefix.value}_phase_diagram.dat", "a")
            phase_diagram_data = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.dat"
            )
            plt.plot()
            plt.savefig(f"{temp}/{prefix.value}_phase_diagram.pdf")
            phase_diagram_plot = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.pdf"
            )
            plt.close("all")

        # Plotting the mixing enthalpies and free energies
        labels = [
            (r"$\Delta G_{mix}$" + f" {temperature} K") for temperature in temperatures
        ]
        plt.style.use("default")
        for i, label in enumerate(labels):
            plt.plot(xs, mixing_free_energies[i], "o-", label=label)
        plt.plot(xs, boltzmann_enthalpies, "o--", label=r"$\Delta H_{mix}$")
        plt.xlim(left=0, right=1)
        plt.xlabel("x")
        plt.ylabel("Mixing energy (eV per atom)")
        plt.legend(loc="best")
        plt.savefig(
            fname=f"{temp}/{prefix.value}_mixing_energies.pdf", bbox_inches="tight"
        )
        mixing_energy_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_mixing_energies.pdf"
        )
        plt.close("all")

        all_enthalpies = np.transpose(mixing_enthalpies)
        for i in range(len(all_enthalpies)):
            plt.plot(xs, all_enthalpies[i], "ro")
        plt.xlabel("x")
        plt.ylabel("Mixing enthalpy (eV per atom)")
        plt.savefig(
            fname=f"{temp}/{prefix.value}_all_enthalpies.pdf", bbox_inches="tight"
        )
        all_enthalpy_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_all_enthalpies.pdf"
        )
        plt.close("all")

    return {
        "mixing_energies": mixing_energies,
        "mixing_energy_plot": mixing_energy_plot,
        "all_enthalpy_plot": all_enthalpy_plot,
        "phase_diagram_data": phase_diagram_data,
        "phase_diagram_plot": phase_diagram_plot,
    }


@calcfunction
def ce_analysis(ce_file, structures, xs, lens, degens, temperatures, prefix):
    """Use cluster expansion to predict and plot mixing enthalpies and free energies"""
    mixing_entropies = []
    Zs = []
    Z = 0
    kB = 8.617343e-5
    T = 298

    # Reading cluster expansion
    with ce_file.as_path() as ce_path:
        ce = ClusterExpansion.read(ce_path)

    # Calculating and storing mixing enthalpies and free energies
    mixing_enthalpies = np.full((len(xs), max(lens)), np.nan)
    for i in range(len(xs)):
        for j in range(lens[i]):
            if i == 0 or i == len(xs) - 1:
                mixing_enthalpy = 0
            else:
                mixing_enthalpy = ce.predict(
                    orm.load_node(structures[f"structure_{i}_{j}"]).get_ase()
                )
            mixing_enthalpies[i][j] = mixing_enthalpy
            Z += degens[i][j] * np.e ** (-mixing_enthalpy / (kB * T))
        Zs.append(Z)
        Z = 0

    boltzmann_probabilities = [
        [
            1 / Zs[i] * degens[i][j] * np.e ** (-mixing_enthalpies[i][j] / (kB * T))
            for j in range(lens[i])
        ]
        for i, _ in enumerate(xs)
    ]
    boltzmann_enthalpies = [
        sum(
            [
                boltzmann_probabilities[i][j] * mixing_enthalpies[i][j]
                for j in range(lens[i])
            ]
        )
        for i, _ in enumerate(xs)
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
            boltzmann_enthalpies[i] - (t * mixing_entropies[i] / 96000)
            for i, _ in enumerate(xs)
        ]
        for t in temperatures
    ]

    mixing_energies = orm.Dict(
        dict={
            "x_values": xs.get_list(),
            "temperatures": temperatures,
            "boltzmann_enthalpies": boltzmann_enthalpies,
            "mixing_free_energies": mixing_free_energies,
        }
    )

    # Calculating binodal and spinodal points for phase diagram
    with TemporaryDirectory() as temp:
        T0 = 195.0
        dT = 5.0
        xc = 0.5
        x1_bin = 0.0001
        x2_bin = 0.9999
        popt, _ = curve_fit(cubic, xs.get_list(), boltzmann_enthalpies)
        fail_count = 0
        x_diff = x2_bin - x1_bin
        while x_diff > 0.001:
            T = T0 + dT
            try:
                x1_spin = brentq(d2free, x1_bin, xc, args=(T, *popt))
                x2_spin = brentq(d2free, xc, x2_bin, args=(T, *popt))
            except ValueError:
                fail_count += 1
                if fail_count > 10:
                    break
                else:
                    continue
            xc = (x1_spin + x2_spin) / 2.0
            x1 = x1_spin
            x2 = x2_spin

            def f1(x):
                return (
                    (xc - x) * interp_free_energy(x2, T, *popt)
                    + (x2 - xc) * interp_free_energy(x, T, *popt)
                ) / (x2 - x)

            def f2(x):
                return (
                    (xc - x1) * interp_free_energy(x, T, *popt)
                    + (x - xc) * interp_free_energy(x1, T, *popt)
                ) / (x - x1)

            h1 = 1.0
            h2 = 1.0
            k = 0
            while h1 > 1e-6 or h2 > 1e-6:
                res = minimize_scalar(f1, bounds=(x1_bin, x1_spin), method="bounded")
                h1 = abs(x1 - res.x)
                x1 = res.x
                res = minimize_scalar(f2, bounds=(x2_spin, x2_bin), method="bounded")
                h2 = abs(x2 - res.x)
                x2 = res.x
                k += 1
                if k == 10:
                    break
            x1_bin = x1
            x2_bin = x2
            with open(f"{temp}/{prefix.value}_phase_diagram.dat", "a") as output:
                output.write(
                    "%.2f %.5f %.5f %.5f %.5f\n" % (T, x1_bin, x2_bin, x1_spin, x2_spin)
                )
            T0 = T
            x_diff = x2_bin - x1_bin
            fail_count = 0

        try:
            phase_diagram_data = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.dat"
            )
            # Plotting the phase diagram
            T = []
            xb = []
            xsp = []
            with open(f"{temp}/{prefix.value}_phase_diagram.dat", "r") as input_file:
                for line in input_file:
                    data = line.split()
                    T.append(float(data[0]))
                    T.append(float(data[0]))
                    xb.append(float(data[1]))
                    xb.append(float(data[2]))
                    xsp.append(float(data[3]))
                    xsp.append(float(data[4]))
            x_bin = [x for (x, y) in sorted(zip(xb, T))]
            T_bin = [y for (x, y) in sorted(zip(xb, T))]
            x_spin = [x for (x, y) in sorted(zip(xsp, T))]
            T_spin = [y for (x, y) in sorted(zip(xsp, T))]
            plt.plot(x_bin, T_bin, "b", lw=1, label="binodal")
            plt.fill_between(x_bin, T_bin, color="b", alpha=0.3)
            plt.plot(x_spin, T_spin, "r", lw=1, label="spinodal")
            plt.fill_between(x_spin, T_spin, color="r", alpha=0.3)
            plt.xlim(left=0, right=1)
            plt.ylim(bottom=200)
            plt.xlabel("x")
            plt.ylabel("T (K)")
            plt.legend(loc="best")
            plt.savefig(f"{temp}/{prefix.value}_phase_diagram.pdf", bbox_inches="tight")
            phase_diagram_plot = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.pdf"
            )
            plt.close("all")
        except:
            open(f"{temp}/{prefix.value}_phase_diagram.dat", "a")
            phase_diagram_data = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.dat"
            )
            plt.plot()
            plt.savefig(f"{temp}/{prefix.value}_phase_diagram.pdf")
            phase_diagram_plot = orm.SinglefileData(
                f"{temp}/{prefix.value}_phase_diagram.pdf"
            )
            plt.close("all")

        # Plotting the mixing enthalpies and free energies
        labels = [
            (r"$\Delta G_{mix}$" + f" {temperature} K") for temperature in temperatures
        ]
        plt.style.use("default")
        for i, label in enumerate(labels):
            plt.plot(xs, mixing_free_energies[i], "o-", label=label)
        plt.plot(xs, boltzmann_enthalpies, "o--", label=r"$\Delta H_{mix}$")
        plt.xlim(left=0, right=1)
        plt.xlabel("x")
        plt.ylabel("Mixing energy (eV per atom)")
        plt.legend(loc="best")
        plt.savefig(
            fname=f"{temp}/{prefix.value}_mixing_energies.pdf", bbox_inches="tight"
        )
        mixing_energy_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_mixing_energies.pdf"
        )
        plt.close("all")

        all_enthalpies = np.transpose(mixing_enthalpies)
        for i in range(len(all_enthalpies)):
            plt.plot(xs, all_enthalpies[i], "ro")
        plt.xlabel("x")
        plt.ylabel("Mixing enthalpy (eV per atom)")
        plt.savefig(
            fname=f"{temp}/{prefix.value}_all_enthalpies.pdf", bbox_inches="tight"
        )
        all_enthalpy_plot = orm.SinglefileData(
            f"{temp}/{prefix.value}_all_enthalpies.pdf"
        )
        plt.close("all")

    return {
        "mixing_energies": mixing_energies,
        "mixing_energy_plot": mixing_energy_plot,
        "all_enthalpy_plot": all_enthalpy_plot,
        "phase_diagram_data": phase_diagram_data,
        "phase_diagram_plot": phase_diagram_plot,
    }


class CastepAlloyWorkChain(WorkChain):
    """
    WorkChain to create and relax alloy/solid solution structures and plot mixing enthalpies and free energies
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepAlloyWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepRelaxWorkChain)
        spec.input(
            "method",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The method used to generate alloy structures ('enum' for brute force enumeration or 'sqs' for special quasirandom structures (SQS))",
            required=False,
            default=lambda: orm.Str("enum"),
        )
        spec.input(
            "use_ce",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            help="Whether to use cluster expansion or not (False by default)",
            required=False,
            default=lambda: orm.Bool(False),
        )
        spec.input(
            "ce_file",
            valid_type=orm.SinglefileData,
            serializer=to_aiida_type,
            help="The cluster expansion file to use as SingleFileData",
            required=False,
        )
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
            help="The transformation matrix for the supercell to be used for the calculations as an array.",
            required=False,
            default=lambda: orm.List(list=[1, 1, 1]),
        )
        spec.input(
            "xs",
            valid_type=orm.List,
            serializer=to_aiida_type,
            help="The percentage(s) of substituent to be used as a list of numbers from 0 to 1. All valid compositions for the specified supercell will be used if not provided.",
            required=False,
            default=lambda: orm.List(),
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
        spec.output(
            "all_enthalpy_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the mixing enthalpies of all configurations for each composition",
            required=True,
        )
        spec.output(
            "phase_diagram_data",
            valid_type=orm.SinglefileData,
            help="Data for the phase diagram",
            required=True,
        )
        spec.output(
            "phase_diagram_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the phase diagram with binodal and spinodal points",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            while_(cls.should_run_relax)(cls.run_relax),
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
        if self.inputs.method == "enum":
            self.ctx.structures = generate_bsym_structures(
                self.ctx.inputs.structure,
                self.inputs.to_substitute,
                self.inputs.substituent,
                self.inputs.supercell_matrix,
                self.inputs.xs,
            )
            self.ctx.degens = self.ctx.structures["degens"]
        elif self.inputs.method == "sqs":
            self.ctx.structures = generate_sqs_structures(
                self.ctx.inputs.structure,
                self.inputs.to_substitute,
                self.inputs.substituent,
                self.inputs.supercell_matrix,
                self.inputs.xs,
            )
            self.ctx.degens = self.ctx.structures["degens"]
        self.ctx.xs = self.ctx.structures["xs"]
        self.ctx.lens = self.ctx.structures["lens"]
        if sum(self.ctx.lens) > 100:
            self.ctx.num_groups = np.ceil(sum(self.ctx.lens) / 100)
        elif self.inputs.use_ce:
            self.ctx.num_groups = 0
            self.ctx.ce_file = self.inputs.ce_file
        else:
            self.ctx.num_groups = 1
        self.ctx.current_x = 0
        self.ctx.current_len = 0

    def should_run_relax(self):
        return self.ctx.num_groups > 0

    def run_relax(self):
        """Relax the symmetry-inequivalent structures for all compositions"""
        inputs = self.ctx.inputs
        relax_parameters = deepcopy(self.ctx.parameters)
        relax_parameters["task"] = "geometryoptimization"
        inputs.calc.parameters = relax_parameters
        count = 0
        end_of_group = False
        for i in range(self.ctx.current_x, len(self.ctx.xs)):
            # Change current_len to 0 after a new group and new x
            for j in range(self.ctx.current_len, self.ctx.lens[i]):
                if count == 100:
                    end_of_group = True
                    self.ctx.current_x = i
                    self.ctx.current_len = j
                    break
                inputs.structure = self.ctx.structures[f"structure_{i}_{j}"]
                inputs.calc.metadata.options.additional_retrieve_list = ["*.castep_bin"]
                key = f"{i}_{j}_relax"
                running = self.submit(CastepRelaxWorkChain, **inputs)
                self.to_context(**{key: running})
                count += 1
            if end_of_group:
                break
            self.ctx.current_len = 0
        self.ctx.num_groups -= 1
        self.report("Running relaxations on symmetry-inequivalent structures")

    def analyse_calcs(self):
        """Analyse the relaxed structures"""
        if self.inputs.use_ce:
            structures = orm.Dict(dict={})
            for i in range(len(self.ctx.xs)):
                for j in range(self.ctx.lens[i]):
                    key = f"structure_{i}_{j}"
                    structures[key] = self.ctx.structures[key].uuid

            outputs = ce_analysis(
                self.ctx.ce_file,
                structures,
                self.ctx.xs,
                self.ctx.lens,
                self.ctx.degens,
                self.inputs.temperatures,
                orm.Str(self.ctx.prefix),
            )
        else:
            self.ctx.relaxed_structures = orm.List(list=[])
            relaxed_structures = []
            kwargs = {}
            for i in range(len(self.ctx.xs)):
                for j in range(self.ctx.lens[i]):
                    key = f"{i}_{j}_relax"
                    if self.ctx[key].is_finished_ok:
                        structure = self.ctx[key].outputs.output_structure
                        relaxed_structures.append(structure.uuid)
                        output_parameters = self.ctx[
                            f"{i}_{j}_relax"
                        ].outputs.output_parameters
                        kwargs[f"out_params_{i}_{j}"] = output_parameters
                    else:
                        relaxed_structures.append("failed")
                self.ctx.relaxed_structures.append(relaxed_structures)
                relaxed_structures = []
            outputs = analysis(
                self.ctx.xs,
                self.ctx.lens,
                self.ctx.degens,
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
        self.ctx.all_enthalpy_plot = add_metadata(
            outputs["all_enthalpy_plot"],
            orm.Str(f"{self.ctx.prefix}_all_enthalpies.pdf"),
            orm.Str(self.ctx.inputs.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.phase_diagram_data = outputs["phase_diagram_data"]
        self.ctx.phase_diagram_plot = add_metadata(
            outputs["phase_diagram_plot"],
            orm.Str(f"{self.ctx.prefix}_phase_diagram.pdf"),
            orm.Str(self.ctx.inputs.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )

    def results(self):
        """Add the relaxed structures, mixing energies and the mixing energy plot to WorkChain outputs"""
        if not self.inputs.use_ce:
            self.out("relaxed_structures", self.ctx.relaxed_structures)
        self.out("mixing_energies", self.ctx.mixing_energies)
        self.out("mixing_energy_plot", self.ctx.mixing_energy_plot)
        self.out("all_enthalpy_plot", self.ctx.all_enthalpy_plot)
        self.out("phase_diagram_data", self.ctx.phase_diagram_data)
        self.out("phase_diagram_plot", self.ctx.phase_diagram_plot)
