#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a test calculation on localhost.

Usage: ./example_01.py
"""
from os import path
import click
from aiida import cmdline, engine
from aiida.plugins import DataFactory, CalculationFactory
from aiida_castep_addons import helpers

INPUT_DIR = path.join(path.dirname(path.realpath(__file__)), 'input_files')

def test_run(castep_addons_code):
    """Run a calculation on the localhost computer.

    Uses test helpers to create AiiDA Code on the fly.
    """
    if not castep_addons_code:
        # get code
        computer = helpers.get_computer()
        castep_addons_code = helpers.get_code(entry_point='castep_addons', computer=computer)

    # Prepare input parameters
    DiffParameters = DataFactory('castep_addons')  
    parameters = DiffParameters({'ignore-case': True})

    SinglefileData = DataFactory('singlefile')
    file1 = SinglefileData(
        file=path.join(INPUT_DIR, 'file1.txt'))
    file2 = SinglefileData(
        file=path.join(INPUT_DIR, 'file2.txt'))

    # set up calculation
    inputs = {
        'code': castep_addons_code,
        'parameters': parameters,
        'file1': file1,
        'file2': file2,
        'metadata': {
            'description': 'Test job submission with the aiida_castep_addons plugin',
        },
    }

    # Note: in order to submit your calculation to the aiida daemon, do:
    # from aiida.engine import submit
    # future = submit(CalculationFactory('castep_addons'), **inputs)
    result = engine.run(CalculationFactory('castep_addons'), **inputs)

    computed_diff = result['castep_addons'].get_content()
    print('Computed diff between files: \n{}'.format(computed_diff))


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE()
def cli(code):
    """Run example.

    Example usage: $ ./example_01.py --code diff@localhost

    Alternative (creates diff@localhost-test code): $ ./example_01.py

    Help: $ ./example_01.py --help
    """
    test_run(code)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
