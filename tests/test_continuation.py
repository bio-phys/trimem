import numpy as np
from math import exp

import trimem.core as m


def test_continuation():
    """test parameter continuation."""

    c = m.ContinuationTuple(1.234)
    c.update()
    assert c.get() == 1.234

    c = m.ContinuationTuple(1.0,2.0,0.1,0.0)
    for i in range(1,10):
        c.update()
        assert np.allclose(c.get(), 1.0+i*0.1)

    c = m.ContinuationTuple(["exp(x)", 11, 0.1, 0.0])
    v = 0.0
    for i in range(11):
        assert np.allclose(c.get(), exp(v))
        c.update()
        v += 0.1

    c = m.ContinuationTuple([1.234])
    c.update()
    assert c.get() == 1.234

    c = m.ContinuationTuple([1.0,2.0,0.1,0.0])
    for i in range(1,10):
        c.update()
        assert np.allclose(c.get(), 1.0+i*0.1)
