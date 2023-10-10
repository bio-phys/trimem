from pathlib import Path
import lammps
with lammps.lammps() as lmp:
    lmp.commands_string(Path('in.pair_python_melt').read_text())