import numpy as np

def _vv_integration(x0, p0, force, m, dt, N):
    """Velocity verlet integration (using momentum instead of velocity)."""

    x = x0
    p = p0
    a = force(x,p)
    xx = [p.copy()]
    pp = [p.copy()]
    for i in range(N):
        x += (p * dt + 0.5 * a * dt**2) / m
        an = force(x)
        p += 0.5 * (a + an) * dt
        a  = an
        xx.append(x.copy())
        pp.append(p.copy())

    return np.array(xx),np.array(pp)

def hmc(x0, nlog_prob, grad_nlog_prob, m, N, dt, L, cooling, callback):
    """Hybrid monte carlo.

    Parameters:
    ----------
    x0:             initial state
    nlog_prob:      negative log of probability density function
    grad_nlog_prob: gradient of negative log of pdf
    m:              variance of momentum variables
    N:              numper of samples to generate
    dt:             time step for integration of hamiltonian system
    L:              number of integration steps of hamiltonian system
    cooling:        callable cooling schedule that takes step as argument
    callback:       callback which takes step and acceptance rate as args
    """

    x = x0.copy()
    xx = []
    acc = 0
    T = cooling(0)
    for i in range(N):

        # sample momenta
        p = np.random.normal(size=x.shape)*np.sqrt(m*T)

        # integrate trajectory
        force = lambda x: -grad_nlog_prob(x,T)
        xn, pn = vv(x, p, force, m, dt, L)

        # evaluate energies
        de  = nlog_prob(xn,T) - nlog_prob(x,T)
        de += 0.5 * ( pn.dot(pn) - p.dot(p) ) / m / T

        # compute acceptance probability
        a = min(1, np.exp(-de))
        u = np.random.uniform()
        if u<=a:
            x = xn
            xx.append(xn.copy())
            acc += 1

        T = cooling(i)

        # perform user defined things
        callback(i, acc/i)

    return np.array(xx)
