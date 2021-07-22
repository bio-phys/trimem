import numpy as np

def _vv_integration(x0, p0, force, m, dt, N):
    """Velocity verlet integration (using momentum instead of velocity)."""

    x = x0
    p = p0
    a = force(x)
    for i in range(N):
        x  = x + (p * dt + 0.5 * a * dt**2) / m
        an = force(x)
        p  = p + 0.5 * (a + an) * dt
        a  = an

    return x, p

def hmc(x0, nlog_prob, grad_nlog_prob, m, N, dt, L, callback, thin=1):
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
    cb:             callback which signature {state, step, step_accepted}
    thin:           keep only every thin'th sample
    """

    def hamiltonian(x,p):
        return nlog_prob(x) + 0.5 * p.ravel().dot(p.ravel()) / m

    x = x0
    xx = []
    acc = 0
    for i in range(N):

        # sample momenta
        p = np.random.normal(size=x.shape)*np.sqrt(m)

        # integrate trajectory
        force = lambda x: -grad_nlog_prob(x)
        xn, pn = _vv_integration(x, p, force, m, dt, L)

        # evaluate energies
        dh = hamiltonian(xn, pn) - hamiltonian(x,p)

        # compute acceptance probability: min(1, np.exp(-de))
        a = 1.0 if dh<=0 else np.exp(-dh)
        u = np.random.uniform()
        acc = u<=a
        if acc:
            x = xn

        if i%thin == 0:
            xx.append(x.copy())

        # user callback
        callback(x, i, acc)

    return x, np.array(xx)
