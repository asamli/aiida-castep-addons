"""
Module for Density of States and Band Structure WorkChain
"""

from __future__ import absolute_import

from copy import deepcopy
from tempfile import TemporaryDirectory

import aiida.orm as orm
import galore
import galore.plot
import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.utils.dos import DOSProcessor
from aiida_castep.workflows.base import CastepBaseWorkChain
from castepxbin.pdos import compute_pdos
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from sumo.electronic_structure.dos import get_pdos
from sumo.plotting.dos_plotter import SDOSPlotter

from aiida_castep_addons.utils import add_metadata, seekpath_analysis
from aiida_castep_addons.utils.sumo_plotter import (
    get_pmg_bandstructure,
    get_sumo_bands_plotter,
)


@calcfunction
def analysis(
    dos_data,
    dos_folder,
    structure,
    band_data,
    band_kpoints,
    prefix,
    experimental_spectra,
):
    """Plot the density of states and band structure with Sumo as well as the UPS, XPS and HAXPES spectra with Galore"""
    # Preparing total DOS using parsed BandsData
    dos_processor = DOSProcessor(
        dos_data.get_bands(), dos_data.get_array("weights"), smearing=0.2
    )
    dos = dos_processor.get_dos()
    dos_energies = dos[0]
    dos_densities = dos[1]
    if dos_data.get_attribute("nspins") > 1:
        dos_efermi = dos_data.get_attribute("efermi")[0]
        pmg_dos = Dos(
            dos_efermi,
            dos_energies,
            {Spin.up: dos_densities[0], Spin.down: dos_densities[1]},
        )
    else:
        dos_efermi = dos_data.get_attribute("efermi")
        pmg_dos = Dos(dos_efermi, dos_energies, {Spin.up: dos_densities[0]})

    with TemporaryDirectory() as temp:
        # Preparing projected DOS from the pdos_bin file
        pdos_bin_data = dos_folder.get_object_content("aiida.pdos_bin", mode="rb")
        with open(f"{temp}/aiida.pdos_bin", "ab") as pdos_bin:
            pdos_bin.write(pdos_bin_data)
        if dos_data.get_attribute("nspins") > 1:
            eigenvalues = {
                Spin.up: dos_data.get_bands()[0].T,
                Spin.down: dos_data.get_bands()[1].T,
            }
        else:
            eigenvalues = {Spin.up: dos_data.get_bands().T}
        weights = dos_data.get_array("weights") * np.ones(
            (eigenvalues[Spin.up].shape[0], 1)
        )
        pmg_pdos = compute_pdos(
            f"{temp}/aiida.pdos_bin",
            eigenvalues,
            weights,
            np.linspace(dos_energies.min(), dos_energies.max(), 2001),
        )
        sorted_structure = structure.get_pymatgen().get_sorted_structure(
            key=lambda x: x.specie.Z
        )
        for i, site in enumerate(sorted_structure.sites):
            pmg_pdos[site] = pmg_pdos.pop(i)
        pmg_complete_dos = CompleteDos(sorted_structure, pmg_dos, pmg_pdos)
        sumo_pdos = get_pdos(pmg_complete_dos)
        pdos_total = np.zeros(2000)
        for orbs in sumo_pdos.values():
            for dos in orbs.values():
                dos.densities = dos.get_smeared_densities(0.2)
                pdos_total += dos.densities[Spin.up]
        scaling_factor = (
            pmg_dos.densities[Spin.up][pdos_total.argmax()] / pdos_total.max()
        )
        for spin in pmg_dos.densities:
            pmg_dos.densities[spin] /= scaling_factor

        # Plotting projected DOS
        dos_plotter = SDOSPlotter(pmg_dos, sumo_pdos).get_plot(xmin=-12, xmax=12)
        dos_plotter.savefig(fname=f"{temp}/{prefix.value}_dos.pdf", bbox_inches="tight")
        dos_plotter.close()
        dos_plot = orm.SinglefileData(f"{temp}/{prefix.value}_dos.pdf")

        # Plotting UPS spectrum
        plt.style.use("default")
        ups_data = galore.process_pdos(
            input=pmg_complete_dos,
            gaussian=0.3,
            lorentzian=0.2,
            xmin=-10 + dos_efermi,
            xmax=4 + dos_efermi,
            weighting="he2",
        )
        ups_plot = galore.plot.plot_pdos(
            ups_data,
            show_orbitals=True,
            units="eV",
            flipx=True,
            offset=dos_efermi,
        )
        try:
            experimental_ups = experimental_spectra.get_array("ups")
            ups_lines = []
            for i, energy in enumerate(experimental_ups[0]):
                ups_lines.append(f"{energy},{experimental_ups[1][i]} \n")
            with open(f"{temp}/ups_data.csv", "w") as csv:
                csv.writelines(ups_lines)
            new_ups_plot = galore.plot.add_overlay(
                plt=ups_plot,
                overlay=f"{temp}/ups_data.csv",
                overlay_label="experimental_ups",
                overlay_style="-",
            )
            new_ups_plot.savefig(
                fname=f"{temp}/{prefix.value}_ups.pdf", bbox_inches="tight"
            )
        except:
            ups_plot.savefig(
                fname=f"{temp}/{prefix.value}_ups.pdf", bbox_inches="tight"
            )
        plt.close("all")
        ups_spectrum = orm.SinglefileData(f"{temp}/{prefix.value}_ups.pdf")

        # Plotting XPS spectrum
        xps_data = galore.process_pdos(
            input=pmg_complete_dos,
            gaussian=0.3,
            lorentzian=0.2,
            xmin=-10 + dos_efermi,
            xmax=4 + dos_efermi,
            weighting="alka",
        )
        xps_plot = galore.plot.plot_pdos(
            xps_data,
            show_orbitals=True,
            units="eV",
            flipx=True,
            offset=dos_efermi,
        )
        try:
            experimental_xps = experimental_spectra.get_array("xps")
            xps_lines = []
            for i, energy in enumerate(experimental_xps[0]):
                xps_lines.append(f"{energy},{experimental_xps[1][i]} \n")
            with open(f"{temp}/xps_data.csv", "w") as csv:
                csv.writelines(xps_lines)
            new_xps_plot = galore.plot.add_overlay(
                plt=xps_plot,
                overlay=f"{temp}/xps_data.csv",
                overlay_label="experimental_xps",
                overlay_style="-",
            )
            new_xps_plot.savefig(
                fname=f"{temp}/{prefix.value}_xps.pdf", bbox_inches="tight"
            )
        except:
            xps_plot.savefig(
                fname=f"{temp}/{prefix.value}_xps.pdf", bbox_inches="tight"
            )
        plt.close("all")
        xps_spectrum = orm.SinglefileData(f"{temp}/{prefix.value}_xps.pdf")

        # Plotting HAXPES spectrum
        haxpes_data = galore.process_pdos(
            input=pmg_complete_dos,
            gaussian=0.3,
            lorentzian=0.2,
            xmin=-10 + dos_efermi,
            xmax=4 + dos_efermi,
            weighting="yeh_haxpes",
        )
        haxpes_plot = galore.plot.plot_pdos(
            haxpes_data,
            show_orbitals=True,
            units="eV",
            flipx=True,
            offset=dos_efermi,
        )
        try:
            experimental_haxpes = experimental_spectra.get_array("haxpes")
            haxpes_lines = []
            for i, energy in enumerate(experimental_haxpes[0]):
                haxpes_lines.append(f"{energy},{experimental_haxpes[1][i]} \n")
            with open(f"{temp}/haxpes_data.csv", "w") as csv:
                csv.writelines(haxpes_lines)
            new_haxpes_plot = galore.plot.add_overlay(
                plt=haxpes_plot,
                overlay=f"{temp}/haxpes_data.csv",
                overlay_label="experimental_haxpes",
                overlay_style="-",
            )
            new_haxpes_plot.savefig(
                fname=f"{temp}/{prefix.value}_haxpes.pdf", bbox_inches="tight"
            )
        except:
            haxpes_plot.savefig(
                fname=f"{temp}/{prefix.value}_haxpes.pdf", bbox_inches="tight"
            )
        plt.close("all")
        haxpes_spectrum = orm.SinglefileData(f"{temp}/{prefix.value}_haxpes.pdf")

        # Plotting overlaid photoelectron spectra
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        weightings = ("He2", "Alka", "Yeh_HAXPES")
        for i, weighting in enumerate(weightings):
            plotting_data = galore.process_pdos(
                input=pmg_complete_dos,
                gaussian=0.3,
                lorentzian=0.2,
                xmin=-10 + dos_efermi,
                xmax=4 + dos_efermi,
                weighting=weighting,
            )
            galore.plot.plot_pdos(
                plotting_data,
                ax=ax,
                show_orbitals=False,
                units="eV",
                flipx=True,
                offset=dos_efermi,
            )
            line = ax.lines[-1]
            line.set_label(weighting)
            line.set_color(f"C{i}")
            ymax = max(line.get_ydata())
            line.set_data(line.get_xdata(), line.get_ydata() / ymax)
        ax.set_ylim((0, 1.2))
        legend = ax.legend(loc="best")
        legend.set_title("Weighting")
        plt.savefig(fname=f"{temp}/{prefix.value}_pe_spectra.pdf", bbox_inches="tight")
        plt.close("all")
        pe_spectra = orm.SinglefileData(f"{temp}/{prefix.value}_pe_spectra.pdf")

        # Plotting band structure
        labelled_bands = deepcopy(band_data)
        labelled_bands.labels = band_kpoints.labels
        if labelled_bands.get_attribute("nspins") > 1:
            bands_efermi = labelled_bands.get_attribute("efermi")[0]
        else:
            bands_efermi = labelled_bands.get_attribute("efermi")
        pmg_bands = get_pmg_bandstructure(labelled_bands, bands_efermi)
        if labelled_bands.get_attribute("nspins") > 1:
            pmg_bands.efermi = bands_efermi
        band_gap = pmg_bands.get_band_gap()
        band_plotter = get_sumo_bands_plotter(labelled_bands, bands_efermi).get_plot(
            ymin=-12, ymax=12
        )
        band_plotter.savefig(
            fname=f"{temp}/{prefix.value}_bands.pdf", bbox_inches="tight"
        )
        band_plot = orm.SinglefileData(f"{temp}/{prefix.value}_bands.pdf")
        plt.close("all")
        plt.style.use("default")

    return {
        "dos_plot": dos_plot,
        "labelled_bands": labelled_bands,
        "band_gap": orm.Dict(dict=band_gap),
        "band_plot": band_plot,
        "ups_spectrum": ups_spectrum,
        "xps_spectrum": xps_spectrum,
        "haxpes_spectrum": haxpes_spectrum,
        "pe_spectra": pe_spectra,
    }


class CastepBandPlotWorkChain(WorkChain):
    """
    WorkChain to calculate and plot the density of states and band structure
    of a material. Galore is also used to plot the UPS, XPS and HAXPES spectra.
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepBandPlotWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input(
            "file_prefix",
            valid_type=orm.Str,
            serializer=to_aiida_type,
            help="The prefix for the names of output files",
            required=False,
        )
        spec.input(
            "seekpath_parameters",
            valid_type=orm.Dict,
            serializer=to_aiida_type,
            help="Parameters to use with seekpath for the k-point path generation",
            required=False,
            default=lambda: orm.Dict(),
        )
        spec.input(
            "experimental_spectra",
            valid_type=orm.ArrayData,
            help="Experimental valence band UPS, XPS and/or HAXPES spectra as 2D arrays. Use 'ups', 'xps' and 'haxpes' as the array names.",
            required=False,
            default=lambda: orm.ArrayData(),
        )

        # The outputs
        spec.output(
            "dos_data",
            valid_type=orm.BandsData,
            help="The parsed BandsData for the density of states calculation",
            required=True,
        )
        spec.output(
            "dos_folder",
            valid_type=orm.FolderData,
            help="The retrieved folder of DOS output files",
            required=True,
        )
        spec.output(
            "dos_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the density of states as a PDF file",
            required=True,
        )
        spec.output(
            "labelled_band_data",
            valid_type=orm.BandsData,
            help="The labelled BandsData for the band structure calculation",
            required=True,
        )
        spec.output(
            "band_gap",
            valid_type=orm.Dict,
            help="The band gap information from Pymatgen as a dictionary",
            required=True,
        )
        spec.output(
            "band_plot",
            valid_type=orm.SinglefileData,
            help="A plot of the band structure as a PDF file",
            required=True,
        )
        spec.output(
            "ups_spectrum",
            valid_type=orm.SinglefileData,
            help="A plot of the UPS spectrum as a PDF file",
            required=True,
        )
        spec.output(
            "xps_spectrum",
            valid_type=orm.SinglefileData,
            help="A plot of the XPS spectrum as a PDF file",
            required=True,
        )
        spec.output(
            "haxpes_spectrum",
            valid_type=orm.SinglefileData,
            help="A plot of the HAXPES spectrum as a PDF file",
            required=True,
        )
        spec.output(
            "pe_spectra",
            valid_type=orm.SinglefileData,
            help="The overlaid photoelectron spectra as a PDF file",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup, cls.run_dos, cls.run_bands, cls.analyse_calculations, cls.results
        )

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.prefix = self.inputs.get(
            "file_prefix",
            f"{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters['xc_functional']}",
        )

    def run_dos(self):
        """Run the spectral density of states calculation"""
        inputs = self.ctx.inputs
        dos_parameters = deepcopy(self.ctx.parameters)
        if ("spectral_kpoints" not in inputs.calc) and (
            "spectral_kpoint_mp_spacing" not in dos_parameters
        ):
            if "kpoints_spacing" not in inputs:
                dos_parameters.update({"spectral_kpoint_mp_spacing": 0.03})
                self.report(
                    "No spectral k-point spacing or mesh provided. Setting spectral k-point spacing to 0.03 A-1(this may not be converged)."
                )
            else:
                dos_parameters.update(
                    {"spectral_kpoint_mp_spacing": inputs.kpoints_spacing / 3}
                )
        dos_parameters.update(
            {
                "task": "spectral",
                "spectral_task": "dos",
                "pdos_calculate_weights": True,
            }
        )
        inputs.calc.parameters = dos_parameters
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running spectral density of states calculation")
        return ToContext(dos=running)

    def run_bands(self):
        """Run the spectral band structure calculation with k-point path from seekpath"""
        inputs = self.ctx.inputs
        band_parameters = deepcopy(self.ctx.parameters)
        band_parameters.update(
            {
                "task": "spectral",
                "spectral_task": "bandstructure",
            }
        )
        inputs.calc.parameters = band_parameters
        current_structure = inputs.calc.structure
        seekpath_data = seekpath_analysis(
            current_structure, self.inputs.seekpath_parameters
        )
        self.ctx.band_kpoints = seekpath_data["kpoints"]
        inputs.calc.spectral_kpoints = self.ctx.band_kpoints
        inputs.calc.structure = seekpath_data["prim_cell"]
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running spectral band structure calculation")
        return ToContext(bands=running)

    def analyse_calculations(self):
        """Analyse the two calculations to plot the density of states, band structure and photoelectron spectra"""

        outputs = analysis(
            self.ctx.dos.outputs.output_bands,
            self.ctx.dos.called[-1].outputs.retrieved,
            self.ctx.inputs.calc.structure,
            self.ctx.bands.outputs.output_bands,
            self.ctx.band_kpoints,
            orm.Str(self.ctx.prefix),
            self.inputs.experimental_spectra,
        )
        self.ctx.dos_plot = add_metadata(
            outputs["dos_plot"],
            orm.Str(f"{self.ctx.prefix}_dos.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.labelled_bands = outputs["labelled_bands"]
        self.ctx.band_gap = outputs["band_gap"]
        self.ctx.band_plot = add_metadata(
            outputs["band_plot"],
            orm.Str(f"{self.ctx.prefix}_bands.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.ups_spectrum = add_metadata(
            outputs["ups_spectrum"],
            orm.Str(f"{self.ctx.prefix}_ups.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.xps_spectrum = add_metadata(
            outputs["xps_spectrum"],
            orm.Str(f"{self.ctx.prefix}_xps.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.haxpes_spectrum = add_metadata(
            outputs["haxpes_spectrum"],
            orm.Str(f"{self.ctx.prefix}_haxpes.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.ctx.pe_spectra = add_metadata(
            outputs["pe_spectra"],
            orm.Str(f"{self.ctx.prefix}_pe_spectra.pdf"),
            orm.Str(self.ctx.inputs.calc.structure.get_formula()),
            orm.Str(self.uuid),
            orm.Str(self.inputs.metadata.get("label", "")),
            orm.Str(self.inputs.metadata.get("description", "")),
        )
        self.report(
            "Density of states, band structure and photoelectron spectra plotted"
        )

    def results(self):
        """Add the plots to WorkChain outputs along with their raw data"""
        self.out("dos_data", self.ctx.dos.outputs.output_bands)
        self.out("dos_folder", self.ctx.dos.called[-1].outputs.retrieved)
        self.out("dos_plot", self.ctx.dos_plot)
        self.out("labelled_band_data", self.ctx.labelled_bands)
        self.out("band_gap", self.ctx.band_gap)
        self.out("band_plot", self.ctx.band_plot)
        self.out("ups_spectrum", self.ctx.ups_spectrum)
        self.out("xps_spectrum", self.ctx.xps_spectrum)
        self.out("haxpes_spectrum", self.ctx.haxpes_spectrum)
        self.out("pe_spectra", self.ctx.pe_spectra)
