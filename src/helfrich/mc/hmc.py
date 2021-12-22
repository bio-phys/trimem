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

_hmc_default_options = {"mass" : 1.0,
                        "number_of_steps" : 1000,
                        "time_step": 1.0e-4,
                        "num_integration_steps" : 100,
                        "callback": None,
                        "thin": 1,
                        "initial_temperature": 1.0,
                        "minimal_temperature": 1.0e-6,
                        "cooling_factor": 0.0,
                        "cooling_start_step": 0}

def hmc(x0, nlog_prob, grad_nlog_prob, options={}):
    """Simple hybrid monte carlo (with optional cooling)

    Parameters:
    ----------
    x0:             initial state
    nlog_prob:      negative log of probability density function
    grad_nlog_prob: gradient of negative log of pdf
    options:        see _hmc_default_options
    """

    # init
    options = {**_hmc_default_options, **options}
    m,N,dt,L,cb,thin,Tinit,Tmin,fT,cN = options.values()
    print(options)

    def hamiltonian(x,p):
        return nlog_prob(x) + 0.5 * p.ravel().dot(p.ravel()) / m

    x = x0
    xx = []
    acc = 0
    T = Tinit
    pvar = m*T
    for i in range(N):

        # sample momenta
        p = np.random.normal(size=x.shape)*np.sqrt(pvar)

        # integrate trajectory
        force = lambda x: -grad_nlog_prob(x)
        xn, pn = _vv_integration(x, p, force, m, dt, L)

        # evaluate energies
        dh = (hamiltonian(xn, pn) - hamiltonian(x,p))/T

        # compute acceptance probability: min(1, np.exp(-de))
        a = 1.0 if dh<=0 else np.exp(-dh)
        u = np.random.uniform()
        acc = u<=a
        if acc:
            x = xn

        if i%thin == 0:
            xx.append(x.copy())

        # update temperature (in case)
        T = max(min(np.exp(-fT*(i-cN))*Tinit, Tinit), Tmin)
        pvar = m*T

        # user callback
        if not cb is None:
            cb(x, i, acc, T)

    return x, np.array(xx)
