"""Internal common functionality."""

import re

def _remsuffix(string, suffix):
    """removesuffix for py <3.9 and without checks. Internal use only."""
    return string[:-len(suffix)]

def _get_pattern(fname):
    """Create part enumeration pattern '{fname.stem}.p{}.{fname.suffix}'."""
    suffix = fname.suffix
    stem   = fname.stem
    return f"{stem}.p{{}}{suffix}"

def _get_parts(fname):
    """Get number of exisiting parts with enumeration pattern (_get_pattern).

    This routine currently just counts exisiting files with the pattern
    and returns the count. Potential gaps in file enumeration are thus not
    recognized nor supported currently.
    """

    pattern = _get_pattern(fname)

    # check how many parts already exist
    parts = []
    for f in fname.parent.iterdir():
        repat = pattern.replace(".", "\.")
        m = re.match(repat.format("[0-9]+"), f.name)
        if not m is None:
            num_str = _remsuffix(m.string, fname.suffix).split(".")[-1][1:]
            parts.append(int(num_str))

    return len(parts)

def _create_part(fname):
    """Create enumerated file-part."""

    pattern = _get_pattern(fname)

    new_part = _get_parts(fname)

    # rename with new part-number
    return fname.with_name(pattern.format(new_part))
