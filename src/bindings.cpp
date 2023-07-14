#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/iostream.h"
#include "pybind11/stl.h"

#include "defs.h"
#include "mesh.h"
#include "mesh_util.h"
#include "mesh_py.h"
#include "numpy_util.h"
#include "energy.h"
#include "flips.h"
#include "mesh_repulsion.h"
#include "mesh_tether.h"
#include "mesh_properties.h"
#include "nlists/nlist.h"
#include "util.h"
#include "params.h"

namespace py = pybind11;

namespace trimem {

// FD gradient for debugging
void gradient(TriMesh& mesh,
              EnergyManager& estore,
              py::array_t<real>& grad,
              real eps=1.0e-6)
{
    // unperturbed energy
    real e0 = estore.energy(mesh);

    auto r_grad = grad.mutable_unchecked<2>();
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        Point& point = mesh.point(mesh.vertex_handle(i));
        for (int j=0; j<3; j++)
        {
            // do perturbation
            point[j] += eps;

            // evaluate differential energy
            real de = ( estore.energy(mesh) - e0 ) / eps;
            r_grad(i,j) = de;

            // undo perturbation
            point[j] -= eps;
        }
    }
}

void expose_mesh(py::module& m)
{

    // not really necessary but used in testing
    py::class_<OpenMesh::HalfedgeHandle>(
        m,
        "HalfedgeHandle",
        R"pbdoc(
        ``OpenMesh::HalfedgeHandle``.

        Handle/Reference to halfedges used by several functions within trimem.
        )pbdoc"
        )
        .def(py::init());

    py::class_<TriMesh>(
        m,
        "TriMesh",
        R"pbdoc(
        Triangulation.

        Specialization of ``OpenMesh::TriMesh_ArrayKernelT<MeshTraits>``.
        )pbdoc"
        )

        .def(
            py::init(),
            R"pbdoc(
            Initialize empty mesh.

            Returns:
                TriMesh with zero vertices and no faces.
            )pbdoc"
        )

		    .def(
            py::init([](
                py::array_t<typename TriMesh::Point::value_type> points,
                py::array_t<int>                                  faces
                )
                {
                    TriMesh mesh;
                    return from_points_cells(points, faces);
			          }
            ),
            py::arg("points"),
            py::arg("faces"),
            R"pbdoc(
            Initialize from arrays of vertices and faces.

            Args:
                points ((N,3) float-array): array of vertices in 3 dimensions.
                faces ((N,3) int-array): array of face indices.

            Returns:
                TriMesh with vertices and faces defined by the input.
            )pbdoc"
        )

        .def(
            "fv_indices",
            &fv_indices,
            R"pbdoc(
            Get face-vertex indices.

            Returns:
                An (N,3) array of type int with N being the number of faces
                that contains the vertex indices defining each face.
            )pbdoc"
         )

        .def(
            "points",
            &points,
            R"pbdoc(
            Get vertex positions.

            Returns:
                An (N,3) array of type float with N being the number of
                vertices.
            )pbdoc"
        )

        .def(
            "n_vertices",
            [](TriMesh& mesh) {return mesh.n_vertices();},
            "Number of vertices in the mesh."
        )

        .def(
            "n_edges",
            [](TriMesh& mesh) {return mesh.n_edges();},
            "Number of edges in the mesh."
        )

        .def(
            "halfedge_handle",
            [](TriMesh& mesh, int i)
            {
                return mesh.halfedge_handle(i);
            },
            py::arg("edge"),
            "Returns handle to first halfedge of ``edge`` i."
        )

        .def(
            "opposite_halfedge_handle",
            [](TriMesh& mesh, const OpenMesh::HalfedgeHandle& he)
            {
                return mesh.opposite_halfedge_handle(he);
            },
            py::arg("edge"),
            "Returns handle to the opposite of ``halfedge`` he."
        )

        .def(
            "next_halfedge_handle",
            [](TriMesh& mesh, const OpenMesh::HalfedgeHandle& he)
            {
                return mesh.next_halfedge_handle(he);
            },
            py::arg("edge"),
            "Returns handle to ``halfedge`` next to ``halfedge`` he."
        );

    m.def(
        "read_mesh",
        &read_mesh,
        py::arg("fname"),
        R"pbdoc(
        Read mesh from fname

        Uses ``OpenMesh::IO::read_mesh``.

        Args:
            fname (str): file name to read mesh from

        Returns:
            An instance of TriMesh read from fname.
        )pbdoc"
        );
}

void expose_properties(py::module& m)
{
    py::class_<VertexProperties>(
        m,
        "VertexProperties",
        R"pbdoc(Container for vertex-averaged properties.

        The vertex averaged quantities `area`, `volume`, `curvature` and
        `bending energy` are at the core of the evaluations involved in the
        Helfrich functional that is represented by :class:`EnergyManager`.
        class:`VertexProperties` encapsulates access to these quantities. It
        further allows for the automatic reduction on vectors of
        :class:`VertexProperties` by implementing the operators
        ``+=`` and ``-=``.

        Its current usage in trimem implies an AoS approach that might be
        replaced by an SoA approach if required by performance considerations.
        In this case the idea is to still keep the container 'interface' as
        is and manage the data layout behind the scenes.
        )pbdoc"
        )
        .def(py::init())
        .def_readwrite(
            "area",
            &VertexProperties::area,
            "Face area"
        )
        .def_readwrite(
            "volume",
            &VertexProperties::volume,
            "Face volume"
        )
        .def_readwrite(
            "curvature",
            &VertexProperties::curvature,
            "Edge curvature"
        )
        .def_readwrite(
            "bending",
            &VertexProperties::bending,
            "Bending energy"
        )
        .def_readwrite(
            "tethering",
            &VertexProperties::tethering,
            "Tether regularization"
        )
        .def_readwrite(
            "repulsion",
            &VertexProperties::repulsion,
            "Repulsion penalty"
        );
}

void expose_mesh_utils(py::module& m)
{
    m.def(
        "edge_length",
        &edge_length,
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate edge length.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The lenght of the edge referred to by ``halfedge_handle``.
        )pbdoc"
    );

    m.def(
        "edge_length_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = edge_length_grad<3>(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of edge length.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the edge length referred to by ``halfedge_handle``
            wrt.\ the coordinates of the vertices that are connected to
            ``halfedge_handle``.
        )pbdoc"
    );

    m.def(
        "face_area",
        &face_area,
        R"pbdoc(
        Evaluate face area.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The area of the face associated to ``halfedge_handle``.
        )pbdoc"
    );

    m.def(
        "face_area_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_area_grad<7>(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of face area.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the area of the face that is associated to
            ``halfedge_handle`` wrt. the coordinates of the vertices of that
            face.
        )pbdoc"
    );

    m.def(
        "face_volume",
        &face_volume,
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate face volume.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The volume of the tetrahedron given by the face associated to
            ``halfedge_handle`` and the origin.
        )pbdoc"
    );

    m.def(
        "face_volume_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_volume_grad<7>(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of face volume.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the volume of the tetrahedron given by the face
            associated to ``halfedge_handle`` and the origin wrt to the
            coordinates of the face's vertices.
        )pbdoc"
    );

    m.def(
        "dihedral_angle",
        &dihedral_angle,
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate dihedral angle.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The dihedral angle between the face associated to
            ``halfedge_handle`` and the face associated to the
            opposite halfedge_handle.
        )pbdoc"
    );

    m.def(
        "dihedral_angle_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = dihedral_angle_grad<15>(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of dihedral angle.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the dihedral angle between the face associated to
            ``halfedge_handle`` and the face associated to the opposite
            halfedge_handle wrt to the coordinates of both faces.
        )pbdoc"
    );
}

void expose_parameters(py::module& m)
{
    // these classes are POD, so no need for __init__ signatures from python
    py::options options;
    options.disable_function_signatures();

    py::enum_<BondType>(
        m,
        "BondType",
        "Types for the tether potential."
        )
        .value("Edge", BondType::Edge, "Smoothed well/box potential on edges.")
        .value("Area", BondType::Area, "Harmonic potential on face area.")
        .value("None", BondType::None, "None")
        .export_values();

    py::class_<BondParams>(
        m,
        "BondParams",
        "Tether regularization parameters."
        )
        .def(py::init())
        .def_readwrite(
            "lc0",
            &BondParams::lc0,
            R"pbdoc(
            Onset distance for attracting force (for BondType::Edge).

            :type: float
            )pbdoc"
        )
        .def_readwrite("lc1",
            &BondParams::lc1,
            R"pbdoc(
            Onset distance for repelling force (for BondType::Edge).

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "a0",
            &BondParams::a0,
            R"pbdoc(
            Reference face area (for BondType::Area).

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "r",
            &BondParams::r,
            R"pbdoc(
            Steepness of regularization potential (must be >=1).

            :type: int.
            )pbdoc"
        )
        .def_readwrite(
            "type",
            &BondParams::type,
            R"pbdoc(
            Type of potential (edge-based, area-based).

            :type: BondType
            )pbdoc"
         );

    py::class_<SurfaceRepulsionParams>(
        m,
        "SurfaceRepulsionParams",
        "Parameters for the surface repulsion penalty."
        )
        .def(py::init())
        .def_readwrite(
            "lc1",
            &SurfaceRepulsionParams::lc1,
            R"pbdoc(
            Onset distance for repelling force.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "r",
            &SurfaceRepulsionParams::r,
            R"pbdoc(
            Steepness of repelling potential (must be >=1).

            :type: int.
            )pbdoc"
        )
        .def_readwrite(
            "n_search",
            &SurfaceRepulsionParams::n_search,
            R"pbdoc(
            Type of neighbour list structures.

            :type: str

            Can be ``cell_list`` or ``verlet_list``.
            )pbdoc"
        )
        .def_readwrite(
            "rlist",
            &SurfaceRepulsionParams::rlist,
            "Neighbour search distance cutoff."
        )
        .def_readwrite(
            "exclusion_level",
            &SurfaceRepulsionParams::exclusion_level,
            R"pbdoc(
            Connected neighbourhood exclusion for neighbour lists.

            :type: int

            Levels of exclusion are inclusive, i.e. 0<1<2. Can be one of:
                * 0: exclude self
                * 1: exclude directly connected neighbourhood (1 edge)
                * 2: exclude indirectly connected neighbourhood (2 edges)

            )pbdoc"
        );

    py::class_<ContinuationParams>(
        m,
        "ContinuationParams",
        "Parameters used for smooth continuation."
        )
        .def(py::init())
        .def_readwrite(
            "delta",
            &ContinuationParams::delta,
            "Interpolation blending `time` step."
        )
        .def_readwrite(
            "lam",
            &ContinuationParams::lambda,
            "Interpolation state."
        );

    py::class_<EnergyParams>(
        m,
        "EnergyParams",
        R"pbdoc(
        Parametrization of the Helfrich functional.

        Modularized POD structure containing parameters for the Helfrich
        functional, the `area`, `volume` and `area-difference` penalties,
        the repulsion penalty and the tether regularization.
        )pbdoc"
        )

        .def(py::init())
        .def_readwrite(
            "kappa_b",
            &EnergyParams::kappa_b,
            R"pbdoc(
            Weight of the Helfrich functional.

            :type: float
            )pbdoc"
         )
        .def_readwrite(
            "kappa_a",
            &EnergyParams::kappa_a,
            R"pbdoc(
            Weight of the surface area penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_v",
            &EnergyParams::kappa_v,
            R"pbdoc(
            Weight of the volume penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_c",
            &EnergyParams::kappa_c,
            R"pbdoc(
            Weight of the area-difference penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_t",
            &EnergyParams::kappa_t,
            R"pbdoc(
            Weight of the tether regularization.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_r",
            &EnergyParams::kappa_r,
            R"pbdoc(
            Weight of the surface repulsion penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "area_frac",
            &EnergyParams::area_frac,
            R"pbdoc(
            Target surface area fraction wrt. the initial geometry.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "volume_frac",
            &EnergyParams::volume_frac,
            R"pbdoc(
            Target volume fraction wrt. the initial geometry.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "curvature_frac",
            &EnergyParams::curvature_frac,
            R"pbdoc(
            Target curvature fraction wrt. the initial geometry.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "bond_params",
            &EnergyParams::bond_params,
            R"pbdoc(
            Parameters for the tether regularization.

            :type: BondParams
            )pbdoc"
        )
        .def_readwrite(
            "repulse_params",
            &EnergyParams::repulse_params,
            R"pbdoc(
            Parameters for the surface repulsion.

            :type: SurfaceRepulsionParams
            )pbdoc"
        )
        .def_readwrite(
            "continuation_params",
            &EnergyParams::continuation_params,
            R"pbdoc(
            Parameters for the parameter continuation.

            :type: ContinuationParams
            )pbdoc"
        );

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
            py::init<const TriMesh&, const EnergyParams&, const VertexProperties &>(),
            py::arg("mesh"),
            py::arg("eparams"),
            py::arg("vertex_properties"),
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

void expose_flips(py::module& m)
{
    m.def(
        "flip",
        &flip_serial,
        py::arg("mesh"),
        py::arg("estore"),
        py::arg("flip_ratio"),
        R"pbdoc(
        Serial flip sweep.

        Performs a sweep over a fraction ``flip_ratio`` of edges in ``mesh``
        trying to flip each edge sequentially and evaluating the energies
        associated to the flip against the Metropolis criterion.

        Args:
            mesh (TriMesh): input mesh to be used
            estore (EnergyManager): instance of :class:`EnergyManager` used in
                combination with the ``mesh`` to evaluate energy differences
                necessary for flip acceptance/rejection.
            flip_ratio (float): ratio of edges to test (must be in [0,1]).
        )pbdoc"
    );

    m.def(
        "pflip",
        &flip_parallel_batches,
        py::arg("mesh"),
        py::arg("estore"),
        py::arg("flip_ratio"),
        R"pbdoc(
        Batch parallel flip sweep.

        Performs a sweep over a fraction ``flip_ratio`` of edges in ``mesh``
        in a batch parallel fashion albeit maintaining chain ergodicity.
        To this end, a batch of edges is selected at random. If an edge is free
        to be flipped independently, i.e., no overlap of its patch with the
        patch of other edges (realized by a locking mechanism), it is flipped
        and its differential contribution to the Hamiltonian is evaluated in
        parallel for the whole batch. Ergodicity is maintained by subsequently
        evaluating the Metropolis criterion for each edge in the batch
        sequentially. This is repeated for a number of
        ``flip_ratio / batch_size * mesh.n_edges`` times.

        Args:
            mesh (TriMesh): input mesh to be used
            estore (EnergyManager): instance of :class:`EnergyManager` used in
                combination with the ``mesh`` to evaluate energy differences
                necessary for flip acceptance/rejection.
            flip_ratio (float): ratio of edges to test (must be in [0,1]).
        )pbdoc"
    );
}

void expose_nlists(py::module& m)
{
    // stump: needed by NeighbourList::point_distances
    py::class_<Point>(
        m,
        "Point",
        R"pbdoc(
        ``OpenMesh::TriMesh::Point``

        Typedef of a vector-like quantity describing vertices/points in 3D.
        )pbdoc"
    );

    // NeighbourList
    py::class_<NeighbourList>(
        m,
        "NeighbourList",
        R"pbdoc(
        Neighbour list interface

        Abstract representation of a neighbour list data structure operating
        on a :class:`TriMesh`. Can be either a ``cell_list`` or a
        ``verlet_list``. See :func:`make_nlist` for its construction.
        )pbdoc"
        )

        .def(
            "distance_matrix",
            &NeighbourList::distance_matrix,
            py::arg("mesh"),
            py::arg("rdist"),
            R"pbdoc(
            Compute sparse distance matrix.

            Args:
                mesh (TriMesh): mesh subject to pair-wise vertex distance
                    compuation
                rdist (float): distance cutoff. This cutoff is additional to the
                    cutoff that was used during list creation! That is, it is
                    meaningful to specify a smaller cutoff than the cutoff used
                    at list creation but a larger cutoff has no effect.

            Returns:
                A sparse distance matrix (as a tuple of lists) containing
                all vertices within a distance ``< rdist``.
            )pbdoc"
        )

        .def(
            "point_distances",
            &NeighbourList::point_distances,
            py::arg("mesh"),
            py::arg("pid"),
            py::arg("rdist"),
            R"pbdoc(
            Compute distances for vertex pid.

            Args:
                mesh (TriMesh): mesh containing vertices to be tested against
                    ``pid``.
                pid (int): vertex in ``mesh`` for which distances to other
                    vertices are to be found.
                rdist (float): distance cutoff. This cutoff is additional to the
                    cutoff that was used during list creation! That is, it is
                    meaningful to specify a smaller cutoff than the cutoff used
                    at list creation but a larger cutoff has no effect.

            Returns:
                A tuple (``distances``, ``ids``) of vectors of distance
                components (in 3D) between ``pid`` and vertices in ``ids``
                with distance ``< rdist``.
            )pbdoc"
        );

    // expose factory
    m.def(
        "make_nlist",
        &make_nlist,
        py::arg("mesh"),
        py::arg("eparams"),
        R"pbdoc(
        Neighbour list factory

        Args:
            mesh (TriMesh): a mesh whose vertices are subject to neighbour search
            eparams (EnergyParams): parametrization of the list structure. Parameters are extracted from the sub-structure :class:`SurfaceRepulsionParams`.

        Returns:
            An interface of type :class:`NeighbourList` representing either
            a cell list or a verlet-list.
        )pbdoc"
    );
}

void expose_utils(py::module& m)
{
    m.def(
        "area",
        &area,
        py::arg("mesh"),
        R"pbdoc(
        Compute surface area.

        Args:
            mesh (TriMesh): input mesh to be evaluated

        Returns:
            The value of the surface area of ``mesh``.
        )pbdoc"
   );

    m.def(
        "edges_length",
        &edges_length,
        py::arg("mesh"),
        R"pbdoc(
        Compute cumulative edge length.

        Args:
            mesh (TriMesh): input mesh to be evaluated

        Returns:
            The cumulated value of the length of all edges in ``mesh``.
        )pbdoc"
    );

    m.def(
        "avg_tri_props",
        &mean_tri_props,
        py::arg("mesh"),
        R"pbdoc(
        Average triangle area and edge length

        Args:
            mesh (TriMesh): mesh to process.

        Returns:
            A tuple (`a`, `l`) with `a` being the ``mesh``'s average face
            area and `l` being the mesh's average edge length. (Used for
            automatic detection of the characteristic lengths involved in
            the tether and repulsion penalties.)
        )pbdoc"
    );
}

PYBIND11_MODULE(core, m) {
    m.doc() = R"pbdoc(
        C++ library with python bindings for trimem.

        This module encapsulates the heavy lifting involved in the
        energy/gradient evaluations associated with trimem in a C++
        library offering bindings to be called from python.
    )pbdoc";

    // expose mesh
    expose_mesh(m);

    // expose mesh properties
    expose_properties(m);

    // expose mesh utils
    expose_mesh_utils(m);

    // expose parameters
    expose_parameters(m);

    // expose energy
    expose_energy(m);

    // expose flips
    expose_flips(m);

    // expose neighbour lists
    expose_nlists(m);

    // expose utils
    expose_utils(m);

    // (debug) energy stuff
    m.def(
        "gradient",
        &gradient,
        py::arg("mesh"),
        py::arg("estore"),
        py::arg("gradient"),
        py::arg("epsilon"),
        R"pbdoc(
        Finite difference gradient of energy.

        This is merely for testing/debugging. Use the gradient function
        exposed by the :class:`EnergyManager` class.

        Args:
            mesh (TriMesh): mesh instance.
            estore (EnergyManager): energy evaluation.
            gradient (numpy.ndarray): (N,3) array filled with the gradient.
            epsilon (float): finite difference perturbation magnitude.

        )pbdoc"
    );
}

}
