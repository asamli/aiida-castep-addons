"""Module for Convergence WorkChain"""

from __future__ import absolute_import
from aiida.engine import WorkChain, while_
import aiida.orm as orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida_castep.calculations.tools import flat_input_param_validator
from .base import CastepBaseWorkChain
from copy import deepcopy
import numpy as np

__version__ = '0.0.1'

class CastepConvergeWorkChain(WorkChain):
    """
    WorkChain to converge the plane-wave energy cutoff and k-point grid spacing with respect to
    the ground state energy.
    """

    #Define the workchain
    @classmethod
    def define(cls, spec):
        super(CastepConvergeWorkChain, cls).define(spec)

        #The inputs
        spec.expose_inputs(CastepBaseWorkChain)
        spec.input('init_pwcutoff',
                   valid_type=orm.Int,
                   serializer=to_aiida_type,
                   help='Initial plane-wave cutoff value in electron volts (eV)',
                   required=False,
                   default=lambda: orm.Int(200))
        spec.input('final_pwcutoff',
                   valid_type=orm.Int,
                   serializer=to_aiida_type,
                   help='Final plane-wave cutoff value in electron volts (eV)',
                   required=False,
                   default=lambda: orm.Int(500))
        spec.input('pwcutoff_step',
                   valid_type=orm.Int,
                   serializer=to_aiida_type,
                   help='Plane-wave cutoff step (increment) in electron volts (eV)',
                   required=False,
                   default=lambda: orm.Int(50))
        spec.input('coarse_kspacing',
                   valid_type=orm.Float,
                   serializer=to_aiida_type,
                   help='The Monkhorst-Pack k-point spacing for the coarsest grid in inverse Angstroms',
                   required=False,
                   default=lambda: orm.Float(0.1))
        spec.input('fine_kspacing',
                   valid_type=orm.Float,
                   serializer=to_aiida_type,
                   help='The Monkhorst-Pack k-point spacing for the finest grid in inverse Angstroms',
                   required=False,
                   default=lambda: orm.Float(0.02))
        spec.input('kspacing_step',
                   valid_type=orm.Float,
                   serializer=to_aiida_type,
                   help='The Monkhorst-Pack k-point spacing step (reduction) in inverse Angstroms',
                   required=False,
                   default=lambda: orm.Float(0.01))
        spec.input('conv_cutoff',
                   valid_type=orm.Float,
                   serializer=to_aiida_type,
                   help="""The cutoff value for the ground state energy in electron volts (eV). When the energy difference per atom between two
                   convergence calculations goes below this value, the latter is considered converged.""", 
                   required=False,
                   default=lambda: orm.Float(0.01))

        #The outputs
        spec.output('conv_pwcutoff',
                    valid_type=orm.Int,
                    help='Converged plane-wave cutoff value in electron volts (eV)',
                    required=True)
        spec.output('conv_kspacing',
                   valid_type=orm.Float,
                   help='The converged Monkhorst-Pack k-point spacing in inverse Angstroms',
                   required=True)

        #Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(
            cls.setup,
            while_(cls.should_conv_pwcutoff) (cls.run_pwcutoff_conv, cls.analyse_pwcutoff_conv),
            while_(cls.should_conv_kspacing) (cls.run_kspacing_conv, cls.analyse_kspacing_conv),
            cls.results)

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()
        self.ctx.init_pwcutoff = self.inputs.init_pwcutoff.value
        self.ctx.final_pwcutoff = self.inputs.final_pwcutoff.value
        self.ctx.coarse_kspacing = self.inputs.coarse_kspacing.value
        self.ctx.fine_kspacing = self.inputs.fine_kspacing.value
        self.ctx.pwcutoff_converged = False
        self.ctx.kspacing_converged = False

    
    def should_conv_pwcutoff(self):
        """Decide if next plane-wave energy cutoff convergence calculation should run"""
        return not self.ctx.pwcutoff_converged

    
    def should_conv_kspacing(self):
        """Decide if next k-point spacing convergence calculations should run"""
        return not self.ctx.kspacing_converged
    
    def run_pwcutoff_conv(self):
        """Run parallel plane-wave energy cutoff convergence calculations with the energy cutoff range and increment provided"""
        inputs = self.ctx.inputs
        inputs.kpoints_spacing = self.ctx.coarse_kspacing
        for pwcutoff in range (self.ctx.init_pwcutoff, self.ctx.final_pwcutoff + 1, self.inputs.pwcutoff_step.value):
            parameters = deepcopy(self.ctx.parameters)
            parameters['cut_off_energy'] = pwcutoff
            inputs.calc.parameters = parameters
            running = self.submit(CastepBaseWorkChain, **inputs)
            key = 'pwcutoff_{}'.format(pwcutoff)
            self.to_context(**{key: running})
        self.report('Running plane-wave cutoff convergence calculations')

    
    def analyse_pwcutoff_conv(self):
        """Analyse the plane-wave energy cutoff convergence calculations"""
        keys = []
        for pwcutoff in range (self.ctx.init_pwcutoff, self.ctx.final_pwcutoff + 1, self.inputs.pwcutoff_step.value):
            key = 'pwcutoff_{}'.format(pwcutoff)
            keys.append(key)
            last_energy = self.ctx[key].outputs.output_parameters['total_energy']
            if len(keys) == 1:
                second_last_energy = 0.0
            else:
                second_last_energy = self.ctx[keys[-2]].outputs.output_parameters['total_energy']
            energy_diff_per_atom = abs(last_energy - second_last_energy) / self.ctx[key].outputs.output_parameters['num_ions']
            if energy_diff_per_atom < self.inputs.conv_cutoff:
                self.report('Plane-wave energy cutoff converged at {} eV'.format(pwcutoff))
                self.ctx.conv_pwcutoff = orm.Int(pwcutoff)
                self.ctx.pwcutoff_converged = True
                return
        self.ctx.init_pwcutoff = self.ctx.final_pwcutoff
        self.ctx.final_pwcutoff += 200 
        self.report('Plane-wave energy cutoff not converged. Increasing upper limit by 200 eV.')
                
    def run_kspacing_conv(self):
        """Run parallel k-point spacing convergence calculations with the k-point spacing range and step provided"""
        inputs = self.ctx.inputs
        inputs.kpoints_spacing = self.ctx.coarse_kspacing
        kspacings = np.arange(self.ctx.coarse_kspacing, self.ctx.fine_kspacing, -self.inputs.kspacing_step.value)
        for kspacing in kspacings:
            inputs.kpoints_spacing = kspacing
            running = self.submit(CastepBaseWorkChain, **inputs)
            key = 'kspacing_{}'.format(kspacing)
            self.to_context(**{key: running})
        self.report('Running k-point grid spacing convergence calculations.')

    def analyse_kspacing_conv(self):
        """Analyse previous k-point spacing convergence calculations"""
        keys = []
        kspacings = np.arange(self.ctx.coarse_kspacing, self.ctx.fine_kspacing - 0.01, -self.inputs.kspacing_step.value)
        for kspacing in kspacings:
            key = 'kspacing_{}'.format(kspacing)
            keys.append(key)
            last_energy = self.ctx[key].outputs.output_parameters['total_energy']
            if len(keys) == 1:
                second_last_energy = 0.0
            else:
                second_last_energy = self.ctx[keys[-2]].outputs.output_parameters['total_energy']
            energy_diff_per_atom = abs(last_energy - second_last_energy) / self.ctx[key].outputs.output_parameters['num_ions']
            if energy_diff_per_atom < self.inputs.conv_cutoff:
                self.report('K-point spacing converged at {} A-1'.format(kspacing))
                self.ctx.conv_kspacing = orm.Float(kspacing)
                self.ctx.kspacing_converged = True
                return
        self.ctx.coarse_kspacing = self.ctx.fine_kspacing
        if self.ctx.coarse_spacing <= 0.02:
            self.ctx.fine_kspacing = 0.0
        else:
            self.ctx.fine_kspacing -= 0.02
        self.report('K-point spacing not converged. Decreasing lower limit by 0.02 eV.')

    def results(self):
        """Add converged cutoff and k-point spacing to WorkChain outputs"""
        self.out('conv_pwcutoff', self.ctx.conv_pwcutoff)
        self.out('conv_kspacing', self.ctx.conv_kspacing)
        
                 
            
            
            
            
        
    
