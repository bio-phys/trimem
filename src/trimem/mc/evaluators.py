

"""Evaluators for :class:`helfirch._core.EnergyManager`.

Wrapper classes controlling the access to the functionality of the
:class:`helfrich._core.EnergyManager` that is provided by the _core C++ module.

Evaluators for energy, gradient and callback are defined as functions
of vertex positions.
"""
import numpy as np
import time
from datetime import datetime, timedelta
import psutil
import copy


_eval_default_options = {
    "info_step":    100,
    "output_step":  1000,
    "cpt_step":     0,
    "refresh_step": 10,
    "flatten":      False,
    "num_steps":    None,
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

    def _update_mesh(func):
        """Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, x, \*args, \*\*kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self, x, *args, **kwargs):
            self.mesh.x = x.reshape(self.mesh.x.shape)
            return func(self, x, *args, **kwargs)
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__
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
    def callback(self, x, steps):
        """Callback.

        Allows for the injection of custom trimem functionality into generic
        sampling and minimization algorithms:

            * stdout verbosity
            * writing of output trajectories
            * writing of checkpoint files
            * update of the internal state of self.estore

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in self.mesh.
            steps (collections.Counter): step counter dictionary
            args: ignored

        Keyword Args:
            kwargs: ignored
        """
        i = sum(steps.values()) #py3.10: steps.total()
        if self.info_step and (i % self.info_step == 0):
            print("\n-- Energy-Evaluation-Step ", i)
            self.estore.print_info(self.mesh.trimesh)
        if self.out_step and (i % self.out_step == 0):
            self.output.write_points_cells(self.mesh.x, self.mesh.f)
        if self.cpt_step and (i % self.cpt_step == 0):
            self.write_cpt(self.mesh, self.estore, steps)
        if self.refresh_step and (i % self.refresh_step == 0):
            self.estore.update_repulsion(self.mesh.trimesh)
        self.estore.update_reference_properties()

        #######
        #if i==1:
        #    open('energy_tot.dat','w')
        #if i>50 and (i % 50 == 0):
        #    with open('energy_tot.dat','a') as file:
        #        file.write(f'{self.estore.energy(self.mesh.trimesh)}\n')



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

    def callback(self, x, steps):
        """Callback with timings.

        Wraps :meth:`EnergyEvaluators.callback` with timing functionality.
        """
        super().callback(x, steps)
        i = sum(steps.values()) #py3.10: steps.total()
        if self.info_step and (i % self.info_step == 0):
            self.timestamps.append(time.time())
            if len(self.timestamps) == 2:
                tspan  = self.timestamps[1] - self.timestamps[0]
                speed  = tspan / self.info_step
                finish = self.start + timedelta(seconds=tspan) * self.n
                print("\n-- Performance measurements")
                print(f"----- estimated speed: {speed:.3e} s/step")
                print(f"----- estimated end:   {finish}")
                self.timestamps.pop(0)



class PerformanceEnergyEvaluators(EnergyEvaluators):
    """EnergyEvaluators with timings for steps.

    Extends :class:`EnergyEvaluators` with periodic measurements of
    the simulation performance and an estimate on the expected runtime.

    ADDS OUTPUT TO performance_measurement.dat containing timeseries of different
    """
    def __init__(self, mesh, estore, output, options):
        super().__init__(mesh, estore, output, options)
        self.timestamps = []
        self.n          = options["num_steps"] // self.info_step
        self.start      = datetime.now()

        self.performance_start = time.time()
        self.performance_timestamps = []
        self.timearray = np.zeros(2)
        self.timearray_new = np.zeros(2)
        self.performance_increment = 1000
        self.prefix=options['prefix']

    def callback(self, x, steps):
        """Callback with timings.

        Wraps :meth:`EnergyEvaluators.callback` with timing functionality.
        """
        super().callback(x, steps)
        i = sum(steps.values()) #py3.10: steps.total()
        if self.info_step and (i % self.info_step == 0):
            self.timestamps.append(time.time())
            if len(self.timestamps) == 2:
                tspan  = self.timestamps[1] - self.timestamps[0]
                speed  = tspan / self.info_step
                finish = self.start + timedelta(seconds=tspan) * self.n
                print("\n-- Performance measurements")
                print(f"----- estimated speed: {speed:.3e} s/step")
                print(f"----- estimated end:   {finish}")
                self.timestamps.pop(0)



        # Section for the preformance measurement of the code
        if i==1:
            with open(f'{self.prefix}_performance.dat','w') as file:
                file.write('#Step Elapsed_Time Time_Per_Step %Vertex_Moves %Mesh_Flips %Residue %CPU RAM_USAGE %RAM RAM_AVAILABLE_PRC RAM_TOTAL\n')

        if (i % self.performance_increment == 0):
            self.performance_timestamps.append(time.time())
            section_time = self.timearray_new - self.timearray
            self.timearray = copy.copy(self.timearray_new)

            if len(self.performance_timestamps) == 2:
                performance_tspan = self.performance_timestamps[1] - self.performance_timestamps[0]



                with open(f'{self.prefix}_performance.dat', 'a') as file:
                    file.write(f'{i} {self.performance_timestamps[1]-self.performance_start}'
                               f' {performance_tspan/self.performance_increment}'
                               f' {section_time[0]/performance_tspan} {section_time[1]/performance_tspan}'
                               f' {(performance_tspan-section_time[0]-section_time[1])/performance_tspan}'
                               f' {psutil.cpu_percent(interval=None)} {psutil.virtual_memory()[3]/1000000000}'
                               f' {psutil.virtual_memory()[2]} {psutil.virtual_memory()[1]/1000000000}'
                               f' {psutil.virtual_memory()[0]/1000000000}\n'
                               )

                self.performance_timestamps.pop(0)

        if i==1:
            open(f'{self.prefix}_energy_tot.dat','w')
        if i>5000000 and (i % 250 == 0):
            with open('energy_tot.dat','a') as file:
                file.write(f'{self.estore.energy(self.mesh.trimesh)}\n')

    def extra_callback(self, timearray_loc):
        self.timearray_new=timearray_loc




