"""Module for Density of States and Band Structure WorkChain"""
from __future__ import absolute_import

import subprocess
from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import matplotlib.pyplot as plt
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.workflows.base import CastepBaseWorkChain
from PyPDF2 import PdfFileReader, PdfFileWriter

__version__ = "0.0.1"


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
def plot_core_loss(folder, uuid, label, description, prefix):
    """Plot the EELS/XANES spectrum using optaDOS"""
    files = [".bands", "-out.cell", ".dome_bin", ".elnes_bin"]
    labels = []
    energies = []
    intensities = []
    all_energies = []
    all_intensities = []
    read = False
    with TemporaryDirectory() as temp:
        for file in files:
            if file[-3:] == "bin":
                data = folder.get_object_content(f"aiida{file}", mode="rb")
                with open(f"{temp}/aiida{file}", "wb") as f:
                    f.write(data)
            else:
                data = folder.get_object_content(f"aiida{file}")
                with open(f"{temp}/aiida{file}", "w") as f:
                    f.write(data)
        with open(f"{temp}/aiida.odi", "w") as odi:
            odi.writelines(
                [
                    "task: core \n",
                    "core_lai_broadening: True \n",
                    "lai_gaussian_width: 1.0 \n",
                    "lai_lorentzian_width: 0.5 \n",
                ]
            )
        subprocess.run("optados.mpi aiida", cwd=temp, shell=True, check=True)
        with open(f"{temp}/aiida_core_edge.dat", "r+") as dat:
            lines = dat.readlines()
            dat.writelines(
                [
                    f"Workflow uuid: {uuid.value} \n",
                    f"Workflow label: {label.value} \n",
                    f"Workflow description: {description.value} \n",
                ]
            )
        optados_data = orm.SinglefileData(f"{temp}/aiida_core_edge.dat")
        lines = [line.strip() for line in lines]
        for i, line in enumerate(lines):
            if line == "#" and i > 3:
                read = True
                continue

            if read:
                if len(line) != 0:
                    if line[0] == "#":
                        labels.append(line[2:])
                    else:
                        line = line.split()
                        energies.append(float(line[0]))
                        intensities.append(float(line[-1]))
                else:
                    all_energies.append(energies)
                    all_intensities.append(intensities)
                    energies = []
                    intensities = []
        for i, label in enumerate(labels):
            plt.plot(all_energies[i], all_intensities[i], label=label)
        plt.xlabel(f"Energy above core edge onset (eV)")
        plt.xlim(left=0)
        plt.ylabel("Intensity (arbitrary units)")
        plt.ylim(bottom=0)
        plt.legend(loc="best")
        plt.savefig(fname=f"{temp}/{prefix.value}_core_loss.pdf")
        core_loss_spectrum = orm.SinglefileData(f"{temp}/{prefix.value}_core_loss.pdf")
        return {
            "optados_data": optados_data,
            "core_loss_spectrum": core_loss_spectrum,
        }


class CastepCoreLossWorkChain(WorkChain):
    """
    WorkChain for core loss calculations and core edge EELS/XANES plots with optaDOS
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepCoreLossWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input(
            "file_prefix",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The prefix for the names of output files",
            required=False,
        )

        # The outputs
        spec.output(
            "optados_data",
            valid_type=orm.SinglefileData,
            help="The optaDOS .dat file for the XANES spectrum",
            required=True,
        )
        spec.output(
            "core_loss_spectrum",
            valid_type=orm.SinglefileData,
            help="A plot of the core loss spectrum",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(cls.setup, cls.run_core_loss, cls.analyse_core_loss, cls.results)

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        prefix = self.inputs.get("file_prefix", None)
        if prefix is None:
            self.ctx.prefix = f'{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters["xc_functional"]}'
        else:
            self.ctx.prefix = prefix
        self.ctx.seekpath_parameters = self.inputs.get("seekpath_parameters", {})

    def run_core_loss(self):
        """Run the spectral density of states calculation"""
        inputs = self.ctx.inputs
        dos_parameters = deepcopy(self.ctx.parameters)
        if ("spectral_kpoints" not in inputs.calc) and (
            "spectral_kpoint_mp_spacing" not in dos_parameters
        ):
            if "kpoints_spacing" not in inputs:
                inputs.calc.spectral_kpoints = inputs.calc.kpoints
            else:
                dos_parameters.update(
                    {"spectral_kpoint_mp_spacing": inputs.kpoints_spacing}
                )
        dos_parameters.update(
            {
                "task": "spectral",
                "spectral_task": "coreloss",
            }
        )
        inputs.calc.parameters = dos_parameters
        inputs.calc.metadata.options.additional_retrieve_list = [
            "*.dome_bin",
            "*.elnes_bin",
        ]
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running spectral core loss calculation")
        return ToContext(core_loss=running)

    def analyse_core_loss(self):
        """Analyse the core loss calculation to plot the EELS/XANES spectrum"""
        outputs = plot_core_loss(
            self.ctx.core_loss.called[-1].outputs.retrieved,
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
            orm.Str(self.ctx.prefix),
        )
        self.ctx.optados_data = outputs["optados_data"]
        self.ctx.core_loss_spectrum = add_metadata(
            outputs["core_loss_spectrum"],
            orm.Str(f"{self.ctx.prefix}_core_loss.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )

        self.report("Core loss spectrum plotted")

    def results(self):
        """Add the plots to WorkChain outputs along with their raw data"""
        self.out("optados_data", self.ctx.optados_data)
        self.out("core_loss_spectrum", self.ctx.core_loss_spectrum)