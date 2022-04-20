"""Evaluators for :class:`helfirch._core.EnergyManager`.

Wrapper classes controlling the access to the functionality of the
:class:`helfrich._core.EnergyManager` that is provided by the _core C++ module.

Evaluators for energy, gradient and callback are defined as functions
of vertex positions.
"""
import numpy as np
import time
from datetime import datetime, timedelta


_eval_default_options = {
    "info_step":    100,
    "output_step":  1000,
    "cpt_step":     0,
    "refresh_step": 10,
    "flatten":      False,
    "num_steps":    None,
    "init_step":    0,
    "write_cpt":    lambda m,e,s: None,
}

class EnergyEvaluators:
    """Functions to evaluate energy, gradient and callback on a per step basis.

    Exposes methods :meth:`fun`, :meth:`grad` as functions of vertex
    positions `x` that are used for generic algorithms that act on a state
    not defined by the mesh (including connectivity) but on the vertex
    positions only.

    A step counter is provided that is used from the :meth:`callback` to
    inject trimem specific features into generic sampling or minimization
    algorithms.

    Args:
        mesh (Mesh): mesh representing the state to be evaluated.
            It's vertices will be updated prior to the evaluation of
            `fun`, `gradient` and `callback`.
        estore (EnergyManager): `backend` to energy and gradient evaluations.
        output (callable): object with callable attribute ``write_points_cells``            having signature `(points, cells)`. Usually one of the writers
            constructed by
            :func:`make_output <helfrich.mc.output.util.make_output>`.
        options (dict): additional (optional) parametrization:

            * ``info_step`` (default: 100): show info every n'th step
            * ``output_step`` (default: 1000): write trajectory output every
              n'th step
            * ``cpt_step`` (default: 0): write checkpoint data every n'th step
            * ``refresh_step`` (default: 10): refresh EnergyManager every
              n'th step
            * ``flatten`` (default: False): ravel gradient to (N,1) array. E.g.,
              needed when used with scipy
            * ``num_steps`` (default: None): number of steps the calling
              algorithm runs
            * ``init_step`` (default: 0): initial step number of the calling
              algorithm
            * ``write_cpt`` (default: ``lambda m,e,s: None``): handle to write
              checkpoint data. Must have signature `(points, cells, stepnum)`.
              See, e.g., :func:`write_checkpoint_handle
              <helfrich.mc.util.write_checkpoint_handle>`

    """

    def __init__(self, mesh, estore, output, options):
        """Initialize."""

        # keep properties to operate
        self.mesh   = mesh
        self.estore = estore
        self.output = output

        # init options
        options    = {**_eval_default_options, **options}

        # pretty print options
        print("\n------------------------")
        print("Energy Evaluators Setup:")
        width = max([len(str(k)) for k in options.keys()])
        for k, v in options.items():
            print(f"  {k: <{width}}: {v}")

        # control info, output and refresh frequencies
        self.info_step    = max(options["info_step"], 0)
        self.out_step     = max(options["output_step"], 0)
        self.cpt_step     = max(options["cpt_step"], 0)
        self.refresh_step = max(options["refresh_step"], 0)

        # register checkpoint handle
        self.write_cpt = options["write_cpt"]

        # output-shape
        if options["flatten"]:
            self._ravel = lambda x: np.ravel(x)
        else:
            self._ravel = lambda x: x

        # init callback counter
        self._step = options["init_step"]

    @property
    def step(self):
        """Step counter."""
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _update_mesh(func):
        """Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, x, \*args, \*\*kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self, x, *args, **kwargs):
            self.mesh.x = x.reshape(self.mesh.x.shape)
            return func(self, x, *args, **kwargs)
        wrap.__doc__ = func.__doc__
        return wrap

    @_update_mesh
    def fun(self, x):
        """Evaluate energy.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.energy(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            float:
                Value of the Energy represented by ``self.estore``.
        """
        return self.estore.energy(self.mesh.trimesh)

    @_update_mesh
    def grad(self, x):
        """Evaluate gradient.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.gradient(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            ndarray[float]:
                Gradient with respect to `x` of the Energy represented by
                ``self.estore``.
        """
        return self._ravel(self.estore.gradient(self.mesh.trimesh))

    @_update_mesh
    def callback(self, x):
        """Callback.

        Updates :attr:`step` and allows for the injection of custom
        trimem functionality into generic sampling and minimization algorithms:

            * stdout verbosity
            * writing of output trajectories
            * writing of checkpoint files
            * update of the internal state of self.estore

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in self.mesh.
            args: ignored

        Keyword Args:
            kwargs: ignored
        """
        if self.info_step and (self.step % self.info_step == 0):
            print("\n-- Energy-Evaluation-Step ",self.step)
            self.estore.print_info(self.mesh.trimesh)
        if self.out_step and (self.step % self.out_step == 0):
            self.output.write_points_cells(self.mesh.x, self.mesh.f)
        if self.cpt_step and (self.step % self.cpt_step == 0):
            self.write_cpt(self.mesh, self.estore, self.step)
        if self.refresh_step and (self.step % self.refresh_step == 0):
            self.estore.update_repulsion(self.mesh.trimesh)
        self.estore.update_reference_properties()
        self.step += 1


class TimingEnergyEvaluators(EnergyEvaluators):
    """EnergyEvaluators with timings for steps.

    Extends :class:`EnergyEvaluators` with periodic measurements of
    the simulation performance and an estimate on the expected runtime.
    """
    def __init__(self, mesh, estore, output, options):
        super().__init__(mesh, estore, output, options)
        self.timestamps = []
        self.n          = options["num_steps"] // self.info_step
        self.start      = datetime.now()

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if self.info_step and (self.step % self.info_step == 0):
            self.timestamps.append(time.time())
            if len(self.timestamps) == 2:
                tspan  = self.timestamps[1] - self.timestamps[0]
                speed  = tspan / self.info_step
                finish = self.start + timedelta(seconds=tspan) * self.n
                print("\n-- Performance measurements")
                print(f"----- estimated speed: {speed:.3e} s/step")
                print(f"----- estimated end:   {finish}")
                self.timestamps.pop(0)
        self._step = value
