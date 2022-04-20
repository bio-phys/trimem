/** \file energy.cpp
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#include "energy.h"

#include "numpy_util.h"
#include "mesh_tether.h"
#include "nlists/nlist.h"
#include "mesh_repulsion.h"
#include "kernel.h"

#include "pybind11/iostream.h"

namespace trimem {

EnergyManager::EnergyManager(const TriMesh& mesh,
                             const EnergyParams& energy_params) :
  params(energy_params)
{
    // setup bond potential
    bonds = make_bonds(params.bond_params);

    // setup neighbour list
    nlist = make_nlist(mesh, params);

    // setup mesh repulsion
    repulse = make_repulsion(*nlist, params.repulse_params);

    // evaluate properties from mesh
    initial_props = properties(mesh);
}

VertexProperties EnergyManager::interpolate_reference_properties() const
{
    auto& cparams = params.continuation_params;

    real i_af = 1.0 - params.area_frac;
    real i_vf = 1.0 - params.volume_frac;
    real i_cf = 1.0 - params.curvature_frac;
    const real& lam = cparams.lambda;

    VertexProperties ref_props{0, 0, 0, 0, 0, 0};
    ref_props.area      = ( 1.0 - lam * i_af ) * initial_props.area;
    ref_props.volume    = ( 1.0 - lam * i_vf ) * initial_props.volume;
    ref_props.curvature = ( 1.0 - lam * i_cf ) * initial_props.curvature;

    return ref_props;
}

void EnergyManager::update_reference_properties()
{
    auto& cparams = params.continuation_params;

    if (cparams.lambda < 1.0)
    {
        cparams.lambda += cparams.delta;
    }
}

void EnergyManager::update_repulsion(const TriMesh& mesh)
{
    nlist   = make_nlist(mesh, params);
    repulse = make_repulsion(*nlist, params.repulse_params);
}

VertexProperties EnergyManager::properties(const TriMesh& mesh)
{
    TrimemProperties kernel(params, mesh, *bonds, *repulse);

    VertexProperties props{ 0, 0, 0, 0, 0, 0};
    parallel_reduction(mesh.n_vertices(), kernel, props);

    return props;
}

real EnergyManager::energy(const TriMesh& mesh)
{
    auto ref_props = interpolate_reference_properties();
    auto props     = properties(mesh);

    return trimem_energy(params, props, ref_props);
}

real EnergyManager::energy(const VertexProperties& props)
{
    auto ref_props = interpolate_reference_properties();
    return trimem_energy(params, props, ref_props);
}

std::vector<Point> EnergyManager::gradient(const TriMesh& mesh)
{
    size_t n = mesh.n_vertices();

    // update global properties
    auto props     = properties(mesh);
    auto ref_props = interpolate_reference_properties();

    // properties gradients
    VertexPropertiesGradient zeros
      { Point(0), Point(0), Point(0), Point(0), Point(0), Point(0) };
    std::vector<VertexPropertiesGradient> gprops(n, zeros);
    TrimemPropsGradient pg_kernel(mesh, *bonds, *repulse, gprops);
    parallel_for(n, pg_kernel);

    // evaluate gradient
    std::vector<Point> gradient(n, Point(0));
    TrimemGradient g_kernel(params, props, ref_props, gprops, gradient);
    parallel_for(n, g_kernel);

    return gradient;
}

void EnergyManager::print_info(const TriMesh& mesh)
{
  auto props     = properties(mesh);
  auto ref_props = interpolate_reference_properties();

  std::ostream& out = std::cout;

  out << "----- EnergyManager info\n";
  out << "reference properties:\n";
  out << "  area:      " << ref_props.area << "\n";
  out << "  volume:    " << ref_props.volume << "\n";
  out << "  curvature: " << ref_props.curvature << "\n";
  out << "current properties:\n";
  out << "  area:      " << props.area << "\n";
  out << "  volume:    " << props.volume << "\n";
  out << "  curvature: " << props.curvature << "\n";
  out << "energies:\n";
  out << "  area:      " << area_penalty(params, props, ref_props) << "\n";
  out << "  volume:    " << volume_penalty(params, props, ref_props) << "\n";
  out << "  area diff: " << curvature_penalty(params, props, ref_props) << "\n";
  out << "  bending:   " << helfrich_energy(params, props) << "\n";
  out << "  tether:    " << tether_penalty(params, props) << "\n";
  out << "  repulsion: " << repulsion_penalty(params, props) << "\n";
  out << "  total:     " << trimem_energy(params, props, ref_props) << "\n";
  out << std::endl;
}

void expose_energy(py::module& m){

    py::class_<EnergyManager>(
            m,
            "EnergyManager",
            R"pbdoc(
            Helfrich functional evaluation.

            Manages a particular parametrization of the Helfrich functional
            with additional penalties and tether-regularization. At its core
            it provides methods :func:`energy` and :func:`gradient` for the
            evaluation of the full Hamiltonian and its gradient.
            )pbdoc"
        )

        .def(
            py::init<const TriMesh&, const EnergyParams&>(),
            py::arg("mesh"),
            py::arg("eparams"),
            R"pbdoc(
            Initialization.

            Initializes the EnergyManager's state from the initial ``mesh``
            and the parametrization provided by ``eparams``. This comprises
            the setup of the initial state of the parameter continuation,
            the set up of the reference properties for `area`, `volume` and
            `curvature` (see :attr:`initial_props`) according to the current
            state of the parameter continuation as well as the construction of
            the neighbour list structures for the repulsion penalty.
            )pbdoc"
        )

        .def(
            "properties",
            &EnergyManager::properties,
            py::arg("mesh"),
            R"pbdoc(
            Evaluation of vertex averaged properties.

            Triggers the evaluation of a vector of vertex-averaged properties
            :class:`VertexProperties` that comprises the basic per-vertex
            quantities.

            Args:
                mesh (TriMesh): mesh representing the state to be evaluated
                    defined by vertex positions as well as connectivity.

            Returns:
                (N,1) array of :class:`VertexProperties` with N
                being the number of vertices.
            )pbdoc"
        )

        .def(
            "energy",
                static_cast<real (EnergyManager::*)(const TriMesh&)>(
                    &EnergyManager::energy),
            py::arg("mesh"),
            R"pbdoc(
            Evaluation of the Hamiltonian.

            Args:
                mesh (TriMesh): mesh representing the state to be evaluated
                    defined by vertex positions as well as connectivity.

            Returns:
                The value of the nonlinear Hamiltonian by computing the
                vector of VertexProperties and reducing it to the value
                of the Hamiltonian.
            )pbdoc"
        )

        .def(
            "energy",
            static_cast<real (EnergyManager::*)(const VertexProperties&)>
                (&EnergyManager::energy),
            py::arg("vprops"),
            R"pbdoc(
            Evaluation of the Hamiltonian.

            Args:
                vprops (VertexProperties): vector of VertexProperties that has
                    already been evaluated beforehand by :func:`properties`.

            Returns:
                The value of the nonlinear Hamiltonian by directly reducing
                on the provided VertexProperties ``vprops``.
            )pbdoc"
        )

        .def(
            "gradient",
            [](EnergyManager& _self, const TriMesh& mesh){
                auto grad = _self.gradient(mesh);
                return tonumpy(grad[0], grad.size());
            },
            py::arg("mesh"),
            R"pbdoc(
            Evaluate gradient of the Hamiltonian.

            Args:
                mesh (TriMesh): mesh representing the state to be evaluated
                    defined by vertex positions as well as connectivity.

            Returns:
                (N,3) array of the gradient of the Hamiltonian given by
                :func:`energy` with respect to the vertex positions.
                N is the number of vertices in ``mesh``.
            )pbdoc"
        )

        .def(
            "update_reference_properties",
            &EnergyManager::update_reference_properties,
            R"pbdoc(
            Update reference configurations.

            Uses the parameter continuation defined in the parametrization
            :attr:`eparams` to update reference values for `area`, `volume`
            and `curvature` from the target configuration.
            )pbdoc"
        )

        .def(
            "update_repulsion",
            &EnergyManager::update_repulsion,
            R"pbdoc(
            Update repulsion penalty.

            Updates internal references to neighbour lists and repulsion
            penalties based on the state of the mesh passed in as ``arg0``.
            )pbdoc"
        )

        .def(
            "print_info",
            &EnergyManager::print_info,
            py::call_guard<py::scoped_ostream_redirect,
            py::scoped_estream_redirect>(),
            py::arg("mesh"),
            "Print energy information evaluated on the state given by ``mesh``."
        )

        .def_readonly(
            "eparams",
            &EnergyManager::params,
            R"pbdoc(
            Parametrization of the Hamiltonian.

            :type: EnergyParams
            )pbdoc"
         )

        .def_readonly(
            "initial_props",
            &EnergyManager::initial_props,
            R"pbdoc(
            Initial reference properties.

            Initial reference properties computed from the definition of the
            target properties for `area`, `volume` and `curvature`. Created
            upon construction.
            )pbdoc"
        );
}

}
