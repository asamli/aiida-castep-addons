"""
Module for NMR WorkChain
"""

from __future__ import absolute_import

from copy import deepcopy

import aiida.orm as orm
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida_castep.workflows.base import CastepBaseWorkChain


@calcfunction
def nmr_analysis(folder):
    """Parse the NMR data from the .castep file and plot the NMR spectrum"""
    nmr_dot_castep = folder.get_object_content("aiida.castep")
    lines = nmr_dot_castep.split("\n")
    nmr_lines = iter(lines)
    species = []
    iso_shieldings = []
    aniso_shieldings = []
    asym = []
    cq = []
    eta = []
    read = False
    for line in nmr_lines:
        if " Chemical Shielding and Electric Field Gradient Tensors " in line:
            read = True
            next(nmr_lines)
            next(nmr_lines)
            next(nmr_lines)
            continue
        elif read and len(line.split()) < 2:
            break

        if read:
            line = line.split()
            species.append(f"{line[1]} {line[2]}")
            iso_shieldings.append(line[3])
            aniso_shieldings.append(line[4])
            asym.append(line[5])
            cq.append(line[6])
            eta.append(line[7])

    nmr_data = orm.Dict(
        dict={
            "species": species,
            "isotropic_shieldings": iso_shieldings,
            "anisotropic_shieldings": aniso_shieldings,
            "asymmetry": asym,
            "cq": cq,
            "eta": eta,
        }
    )

    return nmr_data


class CastepNMRWorkChain(WorkChain):
    """
    WorkChain for calculations of nuclear magnetic resonance (NMR) chemical shifts
    """

    @classmethod
    def define(cls, spec):
        """Define the WorkChain"""
        super(CastepNMRWorkChain, cls).define(spec)

        # The inputs
        spec.expose_inputs(CastepBaseWorkChain)

        # The outputs
        spec.output(
            "nmr_data",
            valid_type=orm.Dict,
            help="NMR chemical shifts and asymmetries as a dictionary",
            required=True,
        )

        # Outline of the WorkChain (the class methods to be run and their order)
        spec.outline(cls.setup, cls.run_nmr, cls.analyse_nmr, cls.results)

    def setup(self):
        """Initialise internal variables"""
        self.ctx.inputs = self.exposed_inputs(CastepBaseWorkChain)
        self.ctx.parameters = self.ctx.inputs.calc.parameters.get_dict()

    def run_nmr(self):
        """Run the NMR calculation"""
        inputs = self.ctx.inputs
        parameters = deepcopy(self.ctx.parameters)
        parameters.update(
            {
                "task": "magres",
                "magres_task": "nmr",
            }
        )
        inputs.calc.parameters = parameters
        running = self.submit(CastepBaseWorkChain, **inputs)
        self.report("Running NMR calculation")
        return ToContext(nmr=running)

    def analyse_nmr(self):
        """Analyse the NMR calculation to extract chemical shifts"""
        self.ctx.nmr_data = nmr_analysis(self.ctx.nmr.called[-1].outputs.retrieved)

        self.report("NMR data extracted")

    def results(self):
        """Add the NMR data to WorkChain outputs"""
        self.out("nmr_data", self.ctx.nmr_data)
