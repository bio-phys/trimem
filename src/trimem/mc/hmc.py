"""Monte Carlo Sampling.

A vanilla Hamiltonian Monte Carlo implementation with optional cooling.
Additionally, a multi-proposal Monte Carlo algorithm is available that is
capable to integrate trimem-specific edge-flip functionality as flip
proposals into the Monte Carlo framework.
"""

import numpy as np
from collections import Counter

from .. import core as m

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

class HMC:
    """Simple Hamiltonian Monte Carlo (with optional cooling).

    This class implements the `marching` only. Recording of the generated
    chain/trajectory must be provided by the user within the callback. For
    a version of this algorithm that keeps `x` in sync with some more
    general state (e.g. a mesh as in trimem) use :class:`MeshHMC`.

    Args:
        x (ndarray[float]): initial state
        nlog_prob (callable): negative log of probability density function
        grad_nlog_prob (callable): gradient of negative log of pdf

    Keyword Args:
        callback (callable): step callback with signature callback(x)
            (defaults to no-op.)
        counter (collections.Counter): step counter
        mass (float): scaling factor for unit-diagonal mass-matrix
            used in the time integration (default: 1.0)
        time_step (float): time step for time integration (default 1.0e-4)
        num_integration_steps (int): number of time integration steps
            (default 100)
        initial_temperature (float): initial temperature for simulated
            annealing (default 1.0)
        minimal_temperature (float): minimal temperature for annealing
            (default 1.0e-6)
        cooling_factor (float): factor for exponential cooling (default 0.0)
        cooling_start_step (int): start simulated annealing at this step
            (default 0)
        info_step (int): print info every n'th step (default 100)

    """

    def __init__(
        self,
        x,
        nlog_prob,
        grad_nlog_prob,
        callback=None,
        counter=Counter(),
        mass=1.0,
        time_step=1.0e-4,
        num_integration_steps=100,
        initial_temperature=1.0,
        minimal_temperature=1.0e-6,
        cooling_factor=0.0,
        cooling_start_step=0,
        info_step=100,
    ):
        """Initialization."""

        # function, gradient and callback evaluation
        self.nlog_prob      = nlog_prob
        self.grad_nlog_prob = grad_nlog_prob
        self.cb             = lambda x,s: None if callback is None else callback

        # init options
        self.m     = mass
        self.dt    = time_step
        self.L     = num_integration_steps
        self.Tinit = initial_temperature
        self.Tmin  = minimal_temperature
        self.fT    = cooling_factor
        self.cN    = cooling_start_step
        self.istep = info_step

        # ref to step counters
        self.counter = counter

        # init algorithm
        self.i   = 0
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

        # update internal step counter
        self.i += 1

    def info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.istep and i_total % self.istep == 0:
            ar = self.acc/self.i if not self.i == 0 else 0.0
            print("\n-- HMC-Step ", self.counter["move"])
            print("----- acc-rate:   ", ar)
            print("----- temperature:", self.T)
            self.acc = 0
            self.i   = 0

    def step(self):
        """Make one step."""

        # update temperature
        i = sum(self.counter.values()) #py3.10: self.counter.total()
        Tn = np.exp(-self.fT * (i - self.cN)) * self.Tinit
        self.T = max(min(Tn, self.Tinit), self.Tmin)

        # make a step
        self._step()

        # update step count
        self.counter["move"] += 1

    def run(self, N):
        """Run HMC for N steps."""
        for i in range(N):
            self.step()
            self.info()
            self.cb(self.x, self.counter)

class MeshHMC(HMC):
    """HMC keeping state `x` in sync with a mesh.

    See :class:`HMC`.
    """
    def __init__(
        self,
        estore,
        nlog_prob,
        grad_nlog_prob,
        **kwargs,
    ):
        super().__init__(
            estore.mesh.x,
            nlog_prob,
            grad_nlog_prob,
            **kwargs,
        )
        self.estore = estore

    def step(self):
        super().step()
        # make sure the mesh is consistent with the state from the base
        self.estore.mesh.x = self.x

class MeshMutation:
    """Mutate edges as a step in a Markov Chain.

    This class wraps the flip functionality available from the core
    C++-module such that it fits into a multi-proposal Monte Carlo framework.

    Args:
        mesh (:class:`TriMesh`): initial state.
        estore (:class:`EnergyManager`):
            `backend` for performing flipping on edges.

    Keyword Args:
        options (dict-like): flip parametrization (optional):

            * ``flip_type`` (default: 'parallel'): 'serial' or 'parallel' flip
              evaluation
            * ``flip_ration`` (default: 0.1): proportion of edges in the mesh
              for which a flip is attempted
            * ``info_step`` (default: 100): print info every n'th step
            * ``init_step`` (default: 0): initial value for the step counter
    """
    def __init__(
        self,
        estore,
        name,
        func,
        rate=0.1,
        counter=Counter(),
        info_step=100,
    ):
        """Init."""

        self.estore = estore
        self.name   = name
        self.mutate = func
        self.mr     = rate
        self.istep  = info_step

        self.i   = 0
        self.acc = 0
        self.counter = counter

    def info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.istep and i_total % self.istep == 0:
            ar = self.acc / self.i if not self.i == 0 else 0.0
            print(f"\n-- {self.name}-Step ", self.counter[self.name])
            print(f"----- {self.name}-accept: ", ar)
            self.acc = 0
            self.i   = 0

    def step(self):
        """Make one step."""
        self.acc += self.mutate(self.estore, self.mr)
        self.i += 1
        self.counter[self.name] += 1

    def run(self, N):
        """Make N flip-sweeps."""
        for i in range(N):
            self.step()
            self.info()

class MeshMonteCarlo:
    """MonteCarlo with two-step moves.

    Bundles :class:`HMC` and :class:`MeshFlips` into a bi-step Monte Carlo
    algorithm where each step comprises a step of the :class:`HMC` or
    a step of :class:`MeshFlips` with equal probability.

    Args:
        hmc (HMC): HMC algorithm
        flips (MeshFlips): Monte Carlo flip algorithm

    Keyword Args:
        callback (callable): step callback with signature callback(x,s) with
            x begin the state and s being compatible with collections.Counter
            (defaults to no-op.)
    """

    def __init__(
        self,
        steps,
        counter=Counter(),
        callback=None
    ):
        """Initialize."""
        self.steps = steps
        self.cb    = (lambda x, s: None) if callback is None else callback

        # make counters consistent
        for s in self.steps:
            s.counter = counter
        self.counter = counter

    def step(self):
        """Make one step each with each algorithm."""
        s = np.random.choice(len(self.steps))
        self.steps[s].step()

    def run(self, N):
        """Run for N steps."""
        for i in range(N):
            self.step()
            for s in self.steps:
                s.info()
            self.cb(self.steps[0].estore.mesh.x, self.counter)
