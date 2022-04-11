"""Module for the CASTEP .castep file magnetic parser"""

from pymatgen.core.structure import Structure


class MagneticParser:
    """Parser for final spin data from .castep output files"""

    def __init__(self, lines):
        """Read the lines in the file and run the other class methods"""
        self.lines = lines
        self.lines = [line.strip() for line in self.lines]
        self.parse_spins()

    def parse_spins(self):
        """Parse the spins from the Mullikan population lines"""
        self.spins = []
        read = False
        for line in self.lines:
            line = line.split()
            if line == []:
                read = False
            elif line[0] == "Species":
                read = True
                continue

            if read:
                if len(line) == 10:
                    self.spins.append(float(line[-1]))
        self.total_spin = sum(self.spins)
