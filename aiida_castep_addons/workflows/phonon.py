"""Module for Phonon WorkChain"""

from __future__ import absolute_import
from aiida.engine import WorkChain, ToContext, calcfunction
import aiida.orm as orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.tools.data.array.kpoints import get_explicit_kpoints_path
from .base import CastepBaseWorkChain
from aiida.plugins import DataFactory

import subprocess
import os
from copy import deepcopy
from PyPDF2 import PdfFileReader, PdfFileWriter
import numpy as np
import matplotlib.pyplot as plt

SinglefileData = DataFactory('singlefile')
KpointsData = DataFactory('array.kpoints')
XyData = DataFactory('array.xy')

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
def phonon_analysis(prefix, ir_folder, raman_folder, structure, uuid, label, description):
    """Plot the phonon band structure with sumo and extract the vibrational spectrum data"""
    band_command = f'sumo-phonon-bandplot -f aiida.phonon -p {prefix.value}'
    subprocess.run('mkdir -p temp', check=True, shell=True, text=True) 
    path = os.getcwd() + '/temp/'
    ir_phonon_data = ir_folder.get_object_content('aiida.phonon')
    with open(path + 'aiida.phonon','x') as phonon_file:
        phonon_file.write(ir_phonon_data)
    raman_phonon_data = raman_folder.get_object_content('aiida.phonon')
    subprocess.run(band_command, cwd=path, check=True, shell=True, text=True)
    
    #Parsing vibrational modes for the IR spectrum from the .phonon file
    ir_lines = ir_phonon_data.split('\n')
    ir_lines = [line.strip() for line in ir_lines]
    ir_unit_lines = [line for line in ir_lines if 'in ' in line]
    ir_frequency_unit = ir_unit_lines[0].replace('Frequencies in', '').lstrip()
    ir_intensity_unit = ir_unit_lines[1].replace('IR intensities in', '').lstrip()
    ir_vib_modes = ir_lines[ir_lines.index('END header') + 2 : ir_lines.index('Phonon Eigenvectors')]
    ir_vib_modes = [line.replace(line[0], '', 1).lstrip() for line in ir_vib_modes]
    ir_frequencies = [line.split(' ')[0] for line in ir_vib_modes]
    ir_intensities = [line.split(' ')[-1] for line in ir_vib_modes]
    ir_data = XyData()
    ir_data.set_x(np.array(ir_frequencies), 'Wavenumber', ir_frequency_unit)
    ir_data.set_y(np.array(ir_intensities), 'Intensity', ir_intensity_unit)

    #Plotting IR spectrum with matplotlib
    plt.stem(ir_frequencies, ir_intensities, markerfmt=' ')
    plt.xlabel(f'Wavenumber ({ir_frequency_unit})')
    plt.ylabel(f'Intensity ({ir_intensity_unit})')
    plt.savefig(fname=path + f'{prefix.value}_ir.pdf', bbox_inches='tight')
    ir_spectrum = SinglefileData(path + f'{prefix.value}_ir.pdf')

    #Parsing vibrational modes for the Raman spectrum from the .phonon file
    raman_lines = raman_phonon_data.split('\n')
    raman_lines = [line.strip() for line in raman_lines]
    raman_unit_lines = [line for line in raman_lines if 'in ' in line]
    raman_frequency_unit = raman_unit_lines[0].replace('Frequencies in', '').lstrip()
    raman_activity_unit = raman_unit_lines[2].replace('Raman activities in', '').lstrip()
    raman_vib_modes = raman_lines[raman_lines.index('END header') + 2 : raman_lines.index('Phonon Eigenvectors')]
    raman_vib_modes = [line.replace(line[0], '', 1).lstrip() for line in raman_vib_modes]
    raman_frequencies = [line.split(' ')[0] for line in raman_vib_modes]
    raman_activities = [line.split(' ')[-1] for line in raman_vib_modes]
    raman_data = XyData()
    raman_data.set_x(np.array(raman_frequencies), 'Raman shift', raman_frequency_unit)
    raman_data.set_y(np.array(raman_activities), 'Intensity', raman_activity_unit)

    #Plotting Raman spectrum with matplotlib
    plt.stem(raman_frequencies, raman_activities, markerfmt=' ')
    plt.xlabel(f'Raman shift ({raman_frequency_unit})')
    plt.ylabel(f'Intensity ({raman_activity_unit})')
    plt.savefig(fname=path + f'{prefix.value}_raman.pdf', bbox_inches='tight')
    raman_spectrum = SinglefileData(path + f'{prefix.value}_raman.pdf')

    #Adding metadata with PyPDF2
    bands_in = open(path + f'/{prefix.value}_phonon_band.pdf', 'rb')
    reader = PdfFileReader(bands_in)
    writer = PdfFileWriter()
    writer.appendPagesFromReader(reader)
    metadata = reader.getDocumentInfo()
    writer.addMetadata(metadata)
    extra_metadata = {'/Formula':structure.get_formula(),
                      '/WorkchainUUID':uuid.value,
                      '/WorkchainLabel':label.value,
                      '/WorkchainDescription':description.value}
    writer.addMetadata(extra_metadata)
    bands_out = open(path + f'/{prefix.value}_phonon_band.pdf', 'ab')
    writer.write(bands_out)
    bands_in.close()
    bands_out.close()

    band_plot = SinglefileData(path + f'/{prefix.value}_phonon_band.pdf')
    return {'bands':band_plot, 'ir_data':ir_data, 'ir_spectrum':ir_spectrum, 'raman_data':raman_data, 'raman_spectrum':raman_spectrum}

class CastepPhononWorkChain(WorkChain):
    """
    WorkChain to calculate the phonons, plot the phonon band structure
    and calculate IR and Raman spectra for materials
    """

    #Define the workchain
    @classmethod
    def define(cls, spec):
        super(CastepPhononWorkChain, cls).define(spec)

        #The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input('file_prefix',
                   valid_type=orm.Str,
                   serializer=to_aiida_type,
                   help='The prefix for the names of output files',
                   required=False)

        #The outputs
        spec.output('phonon_band_plot',
                    valid_type=orm.SinglefileData,
                    help='A plot of the phonon band structure as a PDF file',
                    required=True)
        spec.output('ir_data',
                    valid_type=orm.XyData,
                    help='IR spectrum data for vibrational modes as XyData',
                    required=True)
        spec.output('ir_spectrum',
                    valid_type=orm.SinglefileData,
                    help='IR spectrum of the material as a PDF file',
                    required=True)
        spec.output('raman_data',
                    valid_type=orm.XyData,
                    help='Raman spectrum data for vibrational modes as XyData',
                    required=True)
        spec.output('raman_spectrum',
                    valid_type=orm.SinglefileData,
                    help='Raman spectrum of the material as a PDF file',
                    required=True)
        
        #Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(cls.setup,
                     cls.run_phonon,
                     cls.run_raman,
                     cls.analyse_phonons,
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

    def run_phonon(self):
        """Run the phonon calculation using the linear response method and a phonon fine k-point path from seekpath"""
        inputs = self.exposed_inputs(CastepBaseWorkChain)
        phonon_parameters = deepcopy(self.ctx.parameters)
        if ('phonon_kpoints' not in inputs.calc) and ('phonon_kpoint_mp_spacing' not in phonon_parameters):
            phonon_kpoints = KpointsData()
            phonon_kpoints.set_kpoints_mesh((3, 3, 3))
            inputs.calc.phonon_kpoints = phonon_kpoints
        phonon_parameters.update({'task':'phonon+efield', 'phonon_fine_method':'interpolate'})
        inputs.calc.parameters = phonon_parameters
        current_structure = inputs.calc.structure
        seekpath_data = seekpath_analysis(current_structure)
        self.ctx.band_kpoints = seekpath_data['kpoints']
        inputs.calc.phonon_fine_kpoints = self.ctx.band_kpoints
        inputs.calc.structure = seekpath_data['prim_cell']
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report('Running phonon band structure calculation')
        return ToContext(phonon = running)

    def run_raman(self):
        """Run the gamma-only Raman spectrum calculation"""
        inputs = self.ctx.inputs
        raman_parameters = deepcopy(self.ctx.parameters)
        phonon_kpoints = KpointsData()
        phonon_kpoints.set_kpoints_mesh((1, 1, 1))
        inputs.calc.phonon_kpoints = phonon_kpoints
        raman_parameters.update({'task':'phonon', 'calculate_raman':True})
        inputs.calc.parameters = raman_parameters
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report('Running gamma-only Raman spectrum calculation')
        return ToContext(raman = running)

    def analyse_phonons(self):
        """Analyse the output .phonon file from the calculation to plot the phonon band structure and extract IR and Raman spectrum data"""
        outputs = phonon_analysis(orm.Str(self.ctx.prefix), self.ctx.phonon.called[-1].outputs.retrieved, self.ctx.raman.called[-1].outputs.retrieved, 
                                  self.ctx.inputs.calc.structure, orm.Str(self.uuid), orm.Str(self.inputs.metadata.label), 
                                  orm.Str(self.inputs.metadata.description))
        self.ctx.phonon_band_plot = outputs['bands']
        self.ctx.ir_data = outputs['ir_data']
        self.ctx.ir_spectrum = outputs['ir_spectrum']
        self.ctx.raman_data = outputs['raman_data']
        self.ctx.raman_spectrum = outputs['raman_spectrum']
        self.report('Phonon analysis complete')

    def results(self):
        """Add the phonon band structure plot, IR spectrum data and Raman spectrum data to the WorkChain outputs"""
        self.out('phonon_band_plot', self.ctx.phonon_band_plot)
        self.out('ir_data', self.ctx.ir_data)
        self.out('ir_spectrum', self.ctx.ir_spectrum)
        self.out('raman_data', self.ctx.raman_data)
        self.out('raman_spectrum', self.ctx.raman_spectrum)
        subprocess.run('rm -r temp', check=True, shell=True, text=True)
