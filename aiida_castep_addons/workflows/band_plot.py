"""Module for Density of States and Band Structure WorkChain"""

from __future__ import absolute_import
from aiida.engine import WorkChain, ToContext, calcfunction
import aiida.orm as orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from .base import CastepBaseWorkChain
import subprocess
import os
from copy import deepcopy
from PyPDF2 import PdfFileReader, PdfFileWriter
from aiida.plugins import DataFactory
from aiida_castep.utils.sumo_plot import get_sumo_bands_plotter

SinglefileData = DataFactory('singlefile')

__version__ = '0.0.1'

@calcfunction
def seekpath_analysis(structure):
    """
    Use seekpath for automatic k-point path generation. 
    The k-point path is only valid for the generated primitive cell which may or may not be the same as the input structure.
    """
    seekpath = get_explicit_kpoints_path(structure)
    return {'kpoints':seekpath['explicit_kpoints'], 'prim_cell':seekpath['primitive_structure']}

@calcfunction
def sumo_dos_plot(folder, prefix, structure, uuid, label, description):
    """Plot the density of states with sumo and add workflow metadata to the PDF file"""
    plot_command = f'sumo-dosplot -c castep -f aiida.bands -g 0.2 -p {prefix.value}'
    subprocess.run('mkdir -p temp', check=True, shell=True, text=True) 
    path = os.getcwd() + '/temp/'
    bands_data = folder.get_object_content('aiida.bands')
    with open(path + 'aiida.bands','x') as bands_file:
        bands_file.write(bands_data)
    subprocess.run(plot_command, cwd=path, check=True, shell=True, text=True)

    #Adding metadata with PyPDF2
    dos_in = open(path + f'{prefix.value}_dos.pdf', 'rb')
    reader = PdfFileReader(dos_in)
    writer = PdfFileWriter()
    writer.appendPagesFromReader(reader)
    metadata = reader.getDocumentInfo()
    writer.addMetadata(metadata)
    writer.addMetadata({'/Formula':structure.get_formula(),
                        '/WorkchainUUID':uuid.value,
                        '/WorkchainLabel':label.value,
                        '/WorkchainDescription':description.value})
    dos_out = open(path + f'{prefix.value}_dos.pdf', 'ab')
    writer.write(dos_out)
    dos_in.close()
    dos_out.close()
    return SinglefileData(path + f'{prefix.value}_dos.pdf')

@calcfunction
def sumo_band_plot(bands, kpoints, prefix, structure, uuid, label, description):
    """Plot the band structure with sumo and add workflow metadata to the PDF file"""
    labelled_bands = deepcopy(bands)
    labelled_bands.labels = kpoints.labels
    band_plotter = get_sumo_bands_plotter(labelled_bands)
    band_plot = band_plotter.get_plot()
    plot_file = os.getcwd() + f'/temp/{prefix.value}_bands.pdf'
    band_plot.plot()
    band_plot.savefig(fname=plot_file, bbox_inches='tight')

    #Adding metadata with PyPDF2
    bands_in = open(plot_file, 'rb')
    reader = PdfFileReader(bands_in)
    writer = PdfFileWriter()
    writer.appendPagesFromReader(reader)
    metadata = reader.getDocumentInfo()
    writer.addMetadata(metadata)
    writer.addMetadata({'/Formula':structure.get_formula(),
                        '/WorkchainUUID':uuid.value,
                        '/WorkchainLabel':label.value,
                        '/WorkchainDescription':description.value})
    bands_out = open(plot_file, 'ab')
    writer.write(bands_out)
    bands_in.close()
    bands_out.close()
    return {'labelled_bands':labelled_bands, 'band_plot':SinglefileData(plot_file)}

class CastepBandPlotWorkChain(WorkChain):
    """
    WorkChain to calculate and plot the density of states and band structure
    of a material
    """

    #Define the workchain
    @classmethod
    def define(cls, spec):
        super(CastepBandPlotWorkChain, cls).define(spec)

        #The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input('file_prefix',
                   valid_type=orm.Str,
                   serializer=to_aiida_type,
                   help='The prefix for the names of output files',
                   required=False)

        #The outputs
        spec.output('dos_data',
                    valid_type=orm.BandsData,
                    help='The parsed BandsData for the density of states calculation',
                    required=True)
        spec.output('dos_plot',
                    valid_type=orm.SinglefileData,
                    help='A plot of the density of states as a PDF file',
                    required=True)
        spec.output('labelled_band_data',
                    valid_type=orm.BandsData,
                    help='The labelled BandsData for the band structure calculation',
                    required=True)
        spec.output('band_plot',
                    valid_type=orm.SinglefileData,
                    help='A plot of the band structure as a PDF file',
                    required=True)
        
        #Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(cls.setup,
                     cls.run_dos,
                     cls.run_bands,
                     cls.plot_dos,
                     cls.plot_bands,
                     cls.results)

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        prefix = self.inputs.get('file_prefix', None)
        if prefix is None:
            self.ctx.prefix = f'{self.ctx.inputs.calc.structure.get_formula()}_{self.ctx.parameters["xc_functional"]}'
        else:
            self.ctx.prefix = prefix

    def run_dos(self):
        """Run the spectral density of states calculation"""
        inputs = self.ctx.inputs
        dos_parameters = deepcopy(self.ctx.parameters)
        if ('spectral_kpoints' not in inputs.calc) and ('spectral_kpoint_mp_spacing' not in dos_parameters):
            if 'kpoints_spacing' not in inputs:
                dos_parameters.update({'spectral_kpoint_mp_spacing':0.03})
                self.report('No k-point spacing or spectral k-point data provided. Setting spectral k-point spacing to 0.03 A-1(this may not be converged).')
            else:
                dos_parameters.update({'spectral_kpoint_mp_spacing':inputs.kpoints_spacing/3})
        dos_parameters.update({'task':'spectral', 'spectral_task':'dos', 'spectral_perc_extra_bands':50})
        inputs.calc.parameters = dos_parameters
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report('Running spectral density of states calculation')
        return ToContext(dos = running)

    def run_bands(self):
        """Run the spectral band structure calculation with k-point path from seekpath"""
        inputs = self.ctx.inputs
        band_parameters = deepcopy(self.ctx.parameters)
        band_parameters.update({'task':'spectral', 'spectral_task':'bandstructure', 'spectral_perc_extra_bands':50})
        inputs.calc.parameters = band_parameters
        current_structure = inputs.calc.structure
        seekpath_data = seekpath_analysis(current_structure)
        self.ctx.band_kpoints = seekpath_data['kpoints']
        inputs.calc.spectral_kpoints = self.ctx.band_kpoints
        inputs.calc.structure = seekpath_data['prim_cell']
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report('Running spectral band structure calculation')
        return ToContext(bands = running)

    def plot_dos(self):
        """Plot the density of states"""
        self.ctx.dos_plot = sumo_dos_plot(self.ctx.dos.called[-1].outputs.retrieved, orm.Str(self.ctx.prefix), self.ctx.inputs.calc.structure, orm.Str(self.uuid), 
                                          orm.Str(self.inputs.metadata.label), orm.Str(self.inputs.metadata.description))
        self.report('Density of states plotted')

    def plot_bands(self):
        """Plot the band structure"""
        self.ctx.band_plot = sumo_band_plot(self.ctx.bands.outputs.output_bands, self.ctx.band_kpoints, orm.Str(self.ctx.prefix), self.ctx.inputs.calc.structure, orm.Str(self.uuid), 
                                            orm.Str(self.inputs.metadata.label), orm.Str(self.inputs.metadata.description))
        self.report('Band structure plotted')

    def results(self):
        """Add the density of states and band structure plots to WorkChain outputs and delete the temporary folder"""
        self.out('dos_plot', self.ctx.dos_plot)
        self.out('band_plot', self.ctx.band_plot)
        subprocess.run('rm -r temp', check=True, shell=True, text=True)