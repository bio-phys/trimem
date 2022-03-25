"""Easy access evaluator for the EnergyManager.

This module wraps access to the functionality of the EnergyManager class
provided by _core C++ module.

Evaluators for energy, gradient and callback are provided as functions
of vertex positions 'x'.
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
    """Provide function to evaluate energies, gradients and callbacks."""

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
        """Access step counter."""
        return self._step

    @step.setter
    def step(self, value):
        self._step = value
      
    def _update_mesh(func):
        """Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, x, *args, **kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self, x, *args, **kwargs):
            self.mesh.x = x.reshape(self.mesh.x.shape)
            return func(self, x, *args, **kwargs)
        return wrap

    @_update_mesh
    def fun(self, x):
        """Evaluate energy."""
        return self.estore.energy(self.mesh.trimesh)

    @_update_mesh
    def grad(self, x):
        """Evaluate gradient."""
        return self._ravel(self.estore.gradient(self.mesh.trimesh))

    @_update_mesh
    def callback(self, x):
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
    """EnergyEvaluator with timings for steps."""
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
