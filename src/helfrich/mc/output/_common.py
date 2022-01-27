"""Internal common functionality."""

import re

def _remsuffix(string, suffix):
    """removesuffix for py <3.9 and without checks. Internal use only."""
    return string[:-len(suffix)]
    
def _create_part(fname):
    """Create enumerated file-part '{fname.stem}-NUM.suffix'."""

    suffix = fname.suffix
    stem   = fname.stem

    pattern = f"{stem}.p{{}}{suffix}"

    # check how many parts already exist
    parts = []
    for f in fname.parent.iterdir():
        repat = pattern.replace(".", "\.")
        m = re.match(repat.format("[0-9]+"), f.name)
        if not m is None:
            num_str = _remsuffix(m.string, suffix).split(".")[-1][1:]
            parts.append(int(num_str))

    new_part = 0 if len(parts) == 0 else len(parts)

    # rename with new part-number
    return fname.with_name(pattern.format(new_part))
