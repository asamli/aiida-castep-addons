"""Module for the CASTEP .phonon file parser"""

from pymatgen.core.structure import Structure


class PhononParser:
    """Parser for .phonon output files from CASTEP calculations"""

    def __init__(self, lines):
        """Read the lines in the file and run the other class methods"""
        self.lines = lines
        self.lines = [line.strip() for line in self.lines]
        self.parse_structure()
        self.parse_units()
        self.parse_ir_raman()
        self.parse_qpts()
        self.parse_frequencies()
        self.parse_eigenvectors()

    def parse_structure(self):
        """Parse the structure from the header"""
        species = []
        positions = []
        self.cell = []
        read = False
        for i, line in enumerate(self.lines):
            if "Unit" in line:
                self.cell.append(self.lines[i + 1].split())
                self.cell.append(self.lines[i + 2].split())
                self.cell.append(self.lines[i + 3].split())
            elif "Fractional" in line:
                read = True
                continue
            elif line == "END header":
                break

            if read:
                line = line.split()
                species.append(line[-2])
                positions.append(line[1:4])
        positions = [[float(coord) for coord in line] for line in positions]
        self.cell = [[float(num) for num in line] for line in self.cell]
        self.structure = Structure(self.cell, species, positions)

    def parse_units(self):
        """Parse the frequency and intensity units for vibrational modes from the header"""
        unit_lines = [line.split() for line in self.lines if "in " in line]
        self.frequency_unit = unit_lines[0][-1]
        self.ir_unit = unit_lines[1][-1]
        self.raman_unit = unit_lines[2][-1]

    def parse_ir_raman(self):
        """Parse the vibrational mode frequencies and IR and Raman intensities from the gamma point"""
        vib_modes = self.lines[
            self.lines.index("END header") + 2 : self.lines.index("Phonon Eigenvectors")
        ]
        vib_modes = [line.split() for line in vib_modes]
        self.vib_frequencies = [float(line[1]) for line in vib_modes]
        if len(vib_modes[0]) >= 3:
            self.ir_intensities = [float(line[2]) for line in vib_modes]
        if len(vib_modes[0]) == 4:
            self.raman_intensities = [float(line[-1]) for line in vib_modes]

    def parse_qpts(self):
        """Parse the q-points"""
        qpt_lines = [line for line in self.lines if "q-pt" in line]
        self.qpoints = [line.split()[2:5] for line in qpt_lines]
        self.qpoints = [[float(coord) for coord in qpt] for qpt in self.qpoints]

    def parse_frequencies(self):
        """Parse the frequencies of all q-points"""
        self.frequencies = []
        freqs = []
        for i, line in enumerate(self.lines):
            if "q-pt" in line:
                for j in range(1, len(self.lines)):
                    next_line = self.lines[i + j].split()
                    if "Phonon" in next_line:
                        break
                    freqs.append(float(next_line[1]))
                self.frequencies.append(freqs)
                freqs = []

    def parse_eigenvectors(self):
        """Parse the eigenvectors for all q-points"""
        self.eigenvectors = {}
        eigenvectors = []
        count = 0
        for i, line in enumerate(self.lines):
            if "Mode" in line:
                for j in range(1, len(self.lines) - i):
                    next_line = self.lines[i + j].split()
                    if "q-pt=" in next_line:
                        break
                    eigenvectors.append(next_line[2:])
                eigenvectors = [
                    [float(num) for num in eigenvector] for eigenvector in eigenvectors
                ]
                self.eigenvectors.update(
                    {f"Q-point: {self.qpoints[count]}": eigenvectors}
                )
                eigenvectors = []
                count += 1
