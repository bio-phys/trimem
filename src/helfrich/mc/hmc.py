"""Sampling functionality.

This module provides a plain Hamiltonian Monte Carlo implementation with
optional cooling. Additionally a multi-proposal Monte Carlo algorithm is
available that is capable to integrate trimem-specific edge-flip functionality
as flip proposal into the Monte Carlo framework.
"""

import numpy as np

from .. import _core as m

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

_hmc_default_options = {
    "mass":                  1.0,
    "time_step":             1.0e-4,
    "num_integration_steps": 100,
    "initial_temperature":   1.0,
    "minimal_temperature":   1.0e-6,
    "cooling_factor":        0.0,
    "cooling_start_step":    0,
    "info_step":             100,
    "init_step":             0,
}

class HMC:
    """Simple Hamiltonian Monte Carlo (with optional cooling).

    This class implements only the marching. Recording of the generated
    chain must be provided by the user within the callable 'callback' given
    as optional argument to the contructor. It must implement the signature
    'callback(x)', x being an element of the sample space.
    """

    def __init__(
        self,
        x, 
        nlog_prob,
        grad_nlog_prob,
        callback=lambda x: None,
        options={},
    ):
        """Initialization.

        Parameters:
        ----------
        x:              initial state
        nlog_prob:      negative log of probability density function
        grad_nlog_prob: gradient of negative log of pdf
        callback:       step callback with signature: callback(x)
        options:        see _hmc_default_options
        """

        # function, gradient and callback evaluation
        self.nlog_prob      = nlog_prob
        self.grad_nlog_prob = grad_nlog_prob
        self.cb             = callback

        # init options
        options    = {**_hmc_default_options, **options}
        self.m     = options["mass"]
        self.dt    = options["time_step"]
        self.L     = options["num_integration_steps"]
        self.Tinit = options["initial_temperature"]
        self.Tmin  = options["minimal_temperature"]
        self.fT    = options["cooling_factor"]
        self.cN    = options["cooling_start_step"]
        self.istep = options["info_step"]

        # pretty print options
        print("\n---------------------------------------")
        print("Hamiltonian Monte Carlo Initialization:")
        width = max([len(str(k)) for k in options.keys()])
        for k, v in options.items():
            print(f"  {k: <{width}}: {v}")

        # init algorithm
        self.i   = options["init_step"]
        self.acc = 0
        self.T   = self.Tinit

        # initial state
        self.x = x

    def _hamiltonian(self,x,p):
        """Evaluate Hamiltonian."""
        return self.nlog_prob(x) + 0.5 * p.ravel().dot(p.ravel()) / self.m

    def _step(self):
        """Metropolis step."""

        # adjust momentum variance due to current temperature
        p_var = self.m*self.T

        # sample momenta
        p = np.random.normal(size=self.x.shape)*np.sqrt(p_var)

        # integrate trajectory
        force = lambda x: -self.grad_nlog_prob(x)
        xn, pn = _vv_integration(self.x, p, force, self.m, self.dt, self.L)

        # evaluate energies
        dh = (self._hamiltonian(xn, pn) - self._hamiltonian(self.x,p)) / self.T

        # compute acceptance probability: min(1, np.exp(-de))
        a = 1.0 if dh<=0 else np.exp(-dh)
        u = np.random.uniform()
        acc = u<=a
        if acc:
            self.x    = xn
            self.acc += 1

    def _info(self):
        """Print algorithmic information."""
        if self.istep and self.i % self.istep == 0:
            print("\n-- HMC-Step ",self.i)
            print("----- acc-rate:   ", self.acc/self.istep)
            print("----- temperature:", self.T)
            self.acc = 0

    def step(self):
        """Make one step."""

        # update temperature
        Tn = np.exp(-self.fT * (self.i - self.cN)) * self.Tinit
        self.T = max(min(Tn, self.Tinit), self.Tmin)

        # make a step
        self._step()

        # info
        self._info()

        # update internal step counter
        self.i += 1

    def run(self, N):
        """Run HMC for N steps."""
        for i in range(N):
            self.step()
            self.cb(self.x)


class MeshHMC(HMC):
    """HMC but with mesh object as state."""

    def __init__(
        self,
        mesh, 
        nlog_prob,
        grad_nlog_prob,
        callback=lambda x: None,
        options={},
    ):
        """Init."""
        super().__init__(mesh.x, nlog_prob, grad_nlog_prob, callback, options)
        self.mesh = mesh

    def step(self):
        """Make a step and explicitly update the mesh vertices."""
        super().step()
        self.mesh.x = self.x


_mc_flip_default_options = {
    "flip_type": "parallel",
    "flip_ratio": 0.1,
    "info_step":  100,
    "init_step":  0,
}

class MeshFlips:
    """Flipping edges as a step in a Markov Chain.

    This class wraps the flip functionality available from the _core
    C++-module such that it fits into a multi-proposal Monte Carlo framework.
    """
    def __init__(self, mesh, estore, options={}):
        """Init."""

        self.mesh   = mesh
        self.estore = estore

        # init options
        options    = {**_mc_flip_default_options, **options}
        self.istep = options["info_step"]
        self.fr    = options["flip_ratio"]
        self.ft    = options["flip_type"]

        # pretty print options
        print("\n--------------------------------")
        print("Flips Monte Carlo Initialization:")
        width = max([len(str(k)) for k in options.keys()])
        for k, v in options.items():
            print(f"  {k: <{width}}: {v}")

        if self.ft == "none" or self.fr == 0.0:
            self._flips = lambda: 0
        elif self.ft == "serial":
            self._flips = lambda: m.flip(self.mesh.trimesh, self.estore, self.fr)
        elif self.ft == "parallel":
            self._flips = lambda: m.pflip(self.mesh.trimesh, self.estore, self.fr)
        else:
            raise ValueError("Wrong flip-type: {}".format(self.ft))

        self.i   = options["init_step"]
        self.acc = 0

    def _info(self):
        """Print algorithmic information."""
        if self.istep and self.i % self.istep == 0:
            n_edges = self.mesh.trimesh.n_edges()
            print("\n-- MCFlips-Step ",self.i)
            print("----- flip-accept: ", self.acc/(self.istep * n_edges))
            print("----- flip-rate:   ", self.fr)
            self.acc = 0

    def step(self):
        """Make one step."""
        self.acc += self._flips()
        self._info()
        self.i += 1


class MeshMonteCarlo:
    """MonteCarlo with two-step moves."""

    def __init__(self, hmc, flips, callback=lambda x: None):
        """Initialize."""
        self.hmc   = hmc
        self.flips = flips
        self.cb    = callback

    def step(self):
        """Make one step each with each algorithm."""
        self.hmc.step()
        self.flips.step()

    def run(self, N):
        """Run for N steps."""
        for i in range(N):
            self.step()
            self.cb(self.flips.mesh.x)
