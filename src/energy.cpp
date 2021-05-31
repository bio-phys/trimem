#include <tuple>
#include <omp.h>
#include <random>
#include <chrono>
#include <stdexcept>

#include "MeshTypes.hh"
#include "OpenMesh/Core/Geometry/VectorT.hh"
#include "OpenMesh/Core/Mesh/Handles.hh"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/iostream.h"
#include "pybind11/stl.h"

#include "neighbour_list.h"
#include "cell_list.h"


typedef double real;
typedef OpenMesh::VertexHandle VertexHandle;
typedef OpenMesh::EdgeHandle EdgeHandle;

std::tuple<real, real, real, real> properties_vv(TriMesh& mesh);

typedef std::chrono::high_resolution_clock myclock;
static std::mt19937 generator_(myclock::now().time_since_epoch().count());

void seed(unsigned val)
{
    generator_.seed(val);
}

real randomn()
{
    std::uniform_real_distribution<real> dist(0,1);
    return dist(generator_);
}

class EnergyValueStore
{
public:
    EnergyValueStore(real kappa_b,
                     real kappa_a,
                     real kappa_v,
                     real kappa_c,
                     real volume_frac,
                     real area_frac,
                     real curvature_frac) :
        kappa_b_(kappa_b),
        kappa_a_(kappa_a),
        kappa_v_(kappa_v),
        kappa_c_(kappa_c),
        volume_frac_(volume_frac),
        area_frac_(area_frac),
        curvature_frac_(curvature_frac) {}

    EnergyValueStore(const EnergyValueStore& e) = default;
    EnergyValueStore& operator=(const EnergyValueStore& e) = default;

    // init values from mesh
    void init(TriMesh& mesh,
              const real& ref_delta = 1.0,
              const real& ref_lambda = 0.0)
    {
        // evaluate initial properties
        std::tie(energy, area, volume, curvature) = properties_vv(mesh);

        target_area_      = area * area_frac_;
        target_volume_    = volume * volume_frac_;
        target_curvature_ = curvature * curvature_frac_;

        init_area_      = area;
        init_volume_    = volume;
        init_curvature_ = curvature;

        ref_delta_  = ref_delta;
        ref_lambda_ = ref_lambda;
        if (ref_delta_ > 1.0)
            throw std::runtime_error("Use ref_delta in range [0,1]");
        if (ref_lambda_ > 1.0)
            throw std::runtime_error("Use ref_lambda in range [0,1]");
        set_references(ref_lambda_);

        init_ = true;
    }

    void set_references(const real& lambda)
    {
        ref_volume      = (1.0 - lambda) * init_volume_ +
                          lambda * target_volume_;
        ref_area        = (1.0 - lambda) * init_area_ +
                          lambda * target_area_;
        ref_curvature   = (1.0 - lambda) * init_curvature_ +
                          lambda * target_curvature_;
    }

    void update_references()
    {
        if (ref_lambda_ < 1.0)
        {
            ref_lambda_ += ref_delta_;
            set_references(ref_lambda_);
        }
    }

    real area_energy()
    {
        // area penalty
        real area_energy    = area / ref_area - 1;
        area_energy *= kappa_a_ * area_energy;
        return area_energy;
    }

    real volume_energy()
    {
        // volume penalty
        real volume_energy  = volume / ref_volume - 1;
        volume_energy *= kappa_v_ * volume_energy;
        return volume_energy;
    }

    real area_diff_energy()
    {
        //area difference energy
        real area_difference = curvature / ref_curvature - 1;
        area_difference *= kappa_c_ * area_difference;
        return area_difference;
    }

    real get_energy()
    {
        if (not init_)
          throw std::runtime_error("Use of uninitialized EnergyValueStore");

        // energy contributions
        real area = area_energy();
        real volume = volume_energy();
        real area_diff = area_diff_energy();
        // bending energy
        real bending_energy = kappa_b_ * energy;

        return area + volume + bending_energy + area_diff;
    }

    void print_info(const std::string& baseindent)
    {
      std::string indent = baseindent + "  ";
      std::cout << baseindent << "----- EnergyValueStore info" << std::endl;
      std::cout << indent << "reference properties:" << std::endl;
      std::cout << indent << "  area:      " << ref_area << std::endl;
      std::cout << indent << "  volume:    " << ref_volume << std::endl;
      std::cout << indent << "  curvature: " << ref_curvature << std::endl;
      std::cout << indent << "current properties:" << std::endl;
      std::cout << indent << "  area:      " << area << std::endl;
      std::cout << indent << "  volume:    " << volume << std::endl;
      std::cout << indent << "  curvature: " << curvature << std::endl;
      std::cout << indent << "energies:" << std::endl;
      std::cout << indent << "  area:      " << area_energy() << std::endl;
      std::cout << indent << "  volume:    " << volume_energy() << std::endl;
      std::cout << indent << "  area diff: " << area_diff_energy() << std::endl;
      std::cout << indent << "  bending:   " << kappa_b_ * energy << std::endl;
      std::cout << indent << "  total:     " << get_energy() << std::endl;
      std::cout << std::endl;
    }

    real area;
    real volume;
    real curvature;
    real energy;

    real ref_area;
    real ref_volume;
    real ref_curvature;

private:

    real volume_frac_;
    real area_frac_;
    real curvature_frac_;

    real target_area_;
    real target_volume_;
    real target_curvature_;

    real init_area_;
    real init_volume_;
    real init_curvature_;

    real kappa_b_;
    real kappa_a_;
    real kappa_v_;
    real kappa_c_;

    real ref_lambda_ = 0.0;
    real ref_delta_  = 1.0;

    bool init_ = false;
};

std::tuple<real, real, real, real> properties_e(TriMesh& mesh)
{
    real curvature = 0;
    real surface   = 0;
    real volume    = 0;
    real energy    = 0;

    #pragma omp parallel for reduction(+:curvature,surface,volume,energy)
    for (int i=0; i<mesh.n_halfedges(); i++)
    {
        auto eh = mesh.halfedge_handle(i);
        auto fh = mesh.face_handle(eh);
        if ( fh.is_valid() )
        {
            real sector_area = mesh.calc_sector_area(eh);
            real edge_length = mesh.calc_edge_length(eh);
            real edge_angle  = mesh.calc_dihedral_angle(eh);
            auto face_normal = mesh.calc_face_normal(fh);
            auto face_center = mesh.calc_face_centroid(fh);
            real edge_curv   = 0.5 * edge_angle * edge_length;

            curvature += edge_curv;
            surface   += sector_area;
            volume    += dot(face_normal, face_center) * sector_area / 3;
            energy    += edge_curv * edge_curv;
        }
     }

    // correct multiplicity
    energy    /= 2;
    curvature /= 2;
    surface   /= 3;
    volume    /= 3;
    return std::make_tuple(2 * energy, surface, volume, curvature);
}

std::tuple<real, real, real, real> properties_v(TriMesh& mesh)
{
    real curvature = 0;
    real surface   = 0;
    real volume    = 0;
    real energy    = 0;

    #pragma omp parallel for reduction(+:curvature,surface,volume,energy)
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        real c=0.0, s=0.0, v = 0.0;
        auto ve = mesh.vertex_handle(i);
        auto h_it = mesh.voh_iter(ve);
        for(;h_it.is_valid(); ++h_it)
        {
            auto fh = mesh.face_handle(*h_it);

            if ( fh.is_valid() )
            {
                real sector_area = mesh.calc_sector_area(*h_it);
                real edge_length = mesh.calc_edge_length(*h_it);
                real edge_angle  = mesh.calc_dihedral_angle(*h_it);
                auto face_normal = mesh.calc_face_normal(fh);
                auto face_center = mesh.calc_face_centroid(fh);
                real edge_curv   = 0.25 * edge_angle * edge_length;

                c += edge_curv;
                s += sector_area / 3;
                v += dot(face_normal, face_center) * sector_area / 3;
            }
        }

        surface   += s;
        volume    += v;
        curvature += c;
        energy    += 2 * c * c / s;
     }

    // correct multiplicity
    volume    /= 3;
    return std::make_tuple(energy, surface, volume, curvature);
}

real volume_v(TriMesh& mesh)
{
    real volume = 0;

    #pragma omp parallel for reduction(+:volume)
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        real c=0.0, s=0.0, v = 0.0;
        auto ve = mesh.vertex_handle(i);
        auto h_it = mesh.voh_iter(ve);
        for(; h_it.is_valid(); ++h_it)
        {
            auto fh = mesh.face_handle(*h_it);
            if ( fh.is_valid() )
            {
                real sector_area = mesh.calc_sector_area(*h_it);
                auto face_normal = mesh.calc_face_normal(fh);
                auto face_center = mesh.calc_face_centroid(fh);

                v += dot(face_normal, face_center) * sector_area / 3;
            }
        }

        volume    += v;
     }

    // correct multiplicity
    volume    /= 3;
    return volume;
}

real volume_f(TriMesh& mesh)
{
    real volume = 0;

    #pragma omp parallel for reduction(+:volume)
    for (int i=0; i<mesh.n_faces(); i++)
    {
        auto fh     = mesh.face_handle(i);
        auto normal = mesh.calc_face_normal(fh);
        auto heh    = mesh.halfedge_handle(fh);
        real area   = mesh.calc_sector_area(heh);
        auto center = mesh.calc_face_centroid(fh);
        volume += center[2] * normal[2] * area;
    }

    return volume;
}

std::tuple<real, real, real, real> vertex_properties(TriMesh& mesh,
                                                     const VertexHandle& ve)
{
    real curvature = 0.0;
    real surface   = 0.0;
    real volume    = 0.0;

    int invalid_lengths = 0;

    auto h_it = mesh.voh_iter(ve);
    for(; h_it.is_valid(); ++h_it)
    {
        auto fh = mesh.face_handle(*h_it);

        if ( fh.is_valid() )
        {
            real sector_area = mesh.calc_sector_area(*h_it);
            real edge_length = mesh.calc_edge_length(*h_it);
            real edge_angle  = mesh.calc_dihedral_angle(*h_it);
            auto face_normal = mesh.calc_face_normal(fh);
            auto face_center = mesh.calc_face_centroid(fh);
            real edge_curv   = 0.5 * edge_angle * edge_length;

            curvature += edge_curv;
            surface   += sector_area;
            volume    += dot(face_normal, face_center) * sector_area / 3;
        }
    }

    // correct multiplicity
    // (every face contributes to 3 vertices)
    surface /= 3;
    volume  /= 3;
    // (every edge contributes to 2 vertices)
    curvature /= 2;

    if (surface < 0)
      throw std::runtime_error("Surface must be larger zero");

    real energy = 2 * curvature * curvature / surface;

    return std::make_tuple(energy, surface, volume, curvature);
}

std::tuple<real, real, real, real> vertex_vertex_properties(TriMesh& mesh,
                                                            const VertexHandle& ve)
{
    real energy    = 0.0;
    real area      = 0.0;
    real volume    = 0.0;
    real curvature = 0.0;

    // ve's properties
    std::tie(energy, area, volume, curvature) = vertex_properties(mesh, ve);

    // ve's neighbors' properties
    auto ve_it = mesh.vv_iter(ve);
    for (; ve_it.is_valid(); ve_it++)
    {
        auto props = vertex_properties(mesh, *ve_it);
        energy    += std::get<0>(props);
        area      += std::get<1>(props);
        volume    += std::get<2>(props);
        curvature += std::get<3>(props);
    }

    return std::make_tuple(energy, area, volume, curvature);
}

std::tuple<real, real, real, real>
edge_vertex_properties(TriMesh& mesh, const EdgeHandle& eh)
{
    real energy    = 0.0;
    real area      = 0.0;
    real volume    = 0.0;
    real curvature = 0.0;

    // vertex properties of the first face
    auto heh   = mesh.halfedge_handle(eh, 0);

    auto ve = mesh.to_vertex_handle(heh);
    auto props = vertex_properties(mesh, ve);
    energy    += std::get<0>(props);
    area      += std::get<1>(props);
    volume    += std::get<2>(props);
    curvature += std::get<3>(props);

    auto next_heh = mesh.next_halfedge_handle(heh);
    ve = mesh.to_vertex_handle(next_heh);
    props = vertex_properties(mesh, ve);
    energy    += std::get<0>(props);
    area      += std::get<1>(props);
    volume    += std::get<2>(props);
    curvature += std::get<3>(props);

    // vertex properties of the other face
    heh = mesh.halfedge_handle(eh, 1);

    ve = mesh.to_vertex_handle(heh);
    props = vertex_properties(mesh, ve);
    energy    += std::get<0>(props);
    area      += std::get<1>(props);
    volume    += std::get<2>(props);
    curvature += std::get<3>(props);

    next_heh = mesh.next_halfedge_handle(heh);
    ve = mesh.to_vertex_handle(next_heh);
    props = vertex_properties(mesh, ve);
    energy    += std::get<0>(props);
    area      += std::get<1>(props);
    volume    += std::get<2>(props);
    curvature += std::get<3>(props);

    return std::make_tuple(energy, area, volume, curvature);
}

std::tuple<real, real, real, real> properties_vv(TriMesh& mesh)
{
    real curvature = 0;
    real surface   = 0;
    real volume    = 0;
    real energy    = 0;

    //for (TriMesh::VertexIter ve = mesh.vertices_begin(); ve!=mesh.vertices_end(); ve++)
    //for (int i=0; i<mesh.n_vertices(); i++)
    //for (auto ve : mesh.vertices() )
    #pragma omp parallel for reduction(+:curvature,surface,volume,energy)
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        auto ve = mesh.vertex_handle(i);
        auto props = vertex_properties(mesh, ve);

        energy    += std::get<0>(props);
        surface   += std::get<1>(props);
        volume    += std::get<2>(props);
        curvature += std::get<3>(props);
     }

    // correct multiplicity
    return std::make_tuple(energy, surface, volume, curvature);
}

int check_edge_lengths_v(TriMesh& mesh,
                         VertexHandle& ve,
                         const real& min_t,
                         const real& max_t)
{
    int invalid = 0;

    auto e_it = mesh.ve_iter(ve);
    for(; e_it.is_valid(); ++e_it)
    {
        real edge_length = mesh.calc_edge_length(*e_it);
        if (edge_length < min_t or edge_length > max_t)
            invalid += 1;
    }

    return invalid;
}

std::tuple<int, real, real> check_edge_lengths(TriMesh& mesh,
                                               const real& min_tether,
                                               const real& max_tether)
{
    int invalid_edges = 0;
    real min_edge = max_tether;
    real max_edge = 0.0;

    #pragma omp parallel for reduction(+:invalid_edges) reduction(min:min_edge) reduction(max:max_edge)
    for (int i=0; i<mesh.n_edges(); i++)
    {
        auto eh = mesh.edge_handle(i);
        auto el = mesh.calc_edge_length(eh);
        if ((el<min_tether) or (el>max_tether))
            invalid_edges += 1;
        if ( el > max_edge )
            max_edge = el;
        if ( el < min_edge )
            min_edge = el;
    }

    return std::make_tuple(invalid_edges, min_edge, max_edge);
}

std::tuple<int, int> move_serial(TriMesh& mesh,
                                 EnergyValueStore& estore,
                                 const py::array_t<int>& idx,
                                 const py::array_t<real>& val,
                                 const real& min_t,
                                 const real& max_t,
                                 const real& temp = 1.0)
{
    int acc           = 0;
    int invalid_edges = 0;
    std::uniform_real_distribution<real> accept_dist(0,1);

    // get proxy objects
    auto r_idx = idx.unchecked<1>();
    auto r_val = val.unchecked<2>();
    for (py::ssize_t i=0; i<r_idx.shape(0); i++)
    {
        int vi = r_idx(i);
        auto vh = mesh.vertex_handle(vi);

        // initialize new energy properties
        EnergyValueStore estore_new = estore;

        // remove vertex i's and its neighbors' properties
        auto props =  vertex_vertex_properties(mesh, vh);
        estore_new.energy    -= std::get<0>(props);
        estore_new.area      -= std::get<1>(props);
        estore_new.volume    -= std::get<2>(props);
        estore_new.curvature -= std::get<3>(props);

        // move vertex
        auto p = TriMesh::Point(r_val(i,0),r_val(i,1),r_val(i,2));
        mesh.set_point(vh, mesh.point(vh)+p);

        // check for invalid edges
        int invalid_edges_v = check_edge_lengths_v(mesh, vh, min_t, max_t);
        if (invalid_edges_v > 0)
        {
            mesh.set_point(vh, mesh.point(vh)-p);
            invalid_edges += invalid_edges_v;
            continue;
        }

        // evaluate new properties for vertex i and its neighbors
        props = vertex_vertex_properties(mesh, vh);
        estore_new.energy    += std::get<0>(props);
        estore_new.area      += std::get<1>(props);
        estore_new.volume    += std::get<2>(props);
        estore_new.curvature += std::get<3>(props);

        // compute energies
        real energy_new = estore_new.get_energy();
        real energy_old = estore.get_energy();

        real alpha = std::exp(-(energy_new - energy_old)/temp);
        real u     = accept_dist(generator_);
        if (u <= alpha)
        {
            estore = estore_new;
            acc += 1;
        }
        else
        {
            mesh.set_point(vh, mesh.point(vh)-p);
        }
    }

    return std::make_tuple(acc, invalid_edges);
}

std::tuple<int, int> move_global(TriMesh& mesh,
                                 EnergyValueStore& estore,
                                 const real& min_t,
                                 const real& max_t,
                                 const real& temp = 1.0)
{
    std::uniform_real_distribution<real> accept_dist(0,1);

    // check edges first
    int num_invalid = std::get<0>(check_edge_lengths(mesh, min_t, max_t));
    if (num_invalid > 0)
    {
        return std::make_tuple(0, num_invalid);
    }

    EnergyValueStore estore_new = estore;

    // update properties and evaluate new energy
    auto props = properties_vv(mesh);
    estore_new.energy    = std::get<0>(props);
    estore_new.area      = std::get<1>(props);
    estore_new.volume    = std::get<2>(props);
    estore_new.curvature = std::get<3>(props);

    // evaluate energies
    real energy_new = estore_new.get_energy();
    real energy_old = estore.get_energy();

    // evaluate acceptance probability
    real alpha = std::exp(-(energy_new - energy_old)/temp);
    real u     = accept_dist(generator_);
    if (u <= alpha)
    {
        estore = estore_new;
        return std::make_tuple(1,0);
    }

    return std::make_tuple(0, 0);
}

std::tuple<int, int> flip_serial(TriMesh& mesh,
                                 EnergyValueStore& estore,
                                 const py::array_t<int>& idx,
                                 const real& min_t,
                                 const real& max_t,
                                 const real& temp = 1.0)
{
    std::uniform_real_distribution<real> accept_dist(0.0,1.0);

    int flips = 0;
    int invalid_edges = 0;
    auto r_idx = idx.unchecked<1>(); // check for ndim==1
    for (py::ssize_t i=0; i<r_idx.shape(0); i++)
    {
        int ei = r_idx(i);
        auto eh = mesh.edge_handle(ei);
        if (mesh.is_flip_ok(eh) and !mesh.is_boundary(eh))
        {
            EnergyValueStore estore_new = estore;

            // remove old properties
            auto props = edge_vertex_properties(mesh, eh);
            estore_new.energy    -= std::get<0>(props);
            estore_new.area      -= std::get<1>(props);
            estore_new.volume    -= std::get<2>(props);
            estore_new.curvature -= std::get<3>(props);

            mesh.flip(eh);
            real el = mesh.calc_edge_length(eh);
            if ((el<min_t) or (el>max_t))
            {
                mesh.flip(eh);
                invalid_edges += 1;
            }
            else
            {
                // update new properties
                auto props = edge_vertex_properties(mesh, eh);
                estore_new.energy    += std::get<0>(props);
                estore_new.area      += std::get<1>(props);
                estore_new.volume    += std::get<2>(props);
                estore_new.curvature += std::get<3>(props);

                // evaluate energies
                real energy_new = estore_new.get_energy();
                real energy_old = estore.get_energy();

                // evaluate acceptance probability
                real alpha = std::exp(-(energy_new - energy_old)/temp);
                real u     = accept_dist(generator_);
                if (u <= alpha)
                {
                    estore = estore_new;
                    flips += 1;
                }
                else
                {
                    mesh.flip(eh);
                }
            }
        }
    }

    return std::make_tuple(flips, invalid_edges);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    // energy stuff
    m.def("calc_properties_e", &properties_e, "Edge-based evaluation of surface properties");
    m.def("calc_properties_v", &properties_v, "Vertex-based evaluation of surface properties");
    m.def("calc_properties_vv", &properties_vv, "Vertex-based evaluation of surface properties");
     py::class_<EnergyValueStore>(m, "EnergyValueStore")
        .def(py::init<real, real, real, real, real, real, real>())
        .def("get_energy", &EnergyValueStore::get_energy)
        .def("init", &EnergyValueStore::init,
              py::arg("mesh"), py::arg("ref_delta") = 1.0,
              py::arg("ref_lambda") = 0.0)
        .def("update_references", &EnergyValueStore::update_references)
        .def("print_info", &EnergyValueStore::print_info,
              py::call_guard<py::scoped_ostream_redirect,
              py::scoped_estream_redirect>())
        .def_readwrite("energy", &EnergyValueStore::energy)
        .def_readwrite("area", &EnergyValueStore::area)
        .def_readwrite("volume", &EnergyValueStore::volume)
        .def_readwrite("curvature", &EnergyValueStore::curvature)
        .def_readwrite("ref_area", &EnergyValueStore::ref_area)
        .def_readwrite("ref_volume", &EnergyValueStore::ref_volume)
        .def_readwrite("ref_curvature", &EnergyValueStore::ref_curvature);

    // test volumes
    m.def("volume_v", &volume_v, "Volume based on triangle volumes.");
    m.def("volume_f", &volume_f, "Volume based on divergence theorem.");

    // flip stuff
    m.def("check_edges", &check_edge_lengths, "Check for invalid edges");

    // random stuff
    m.def("seed", &seed, "Test random number generator");
    m.def("rand", &randomn, "Test random number generator");

    // monte carlo stuff
    m.def("move_serial", &move_serial, "Test single vertex markov step",
          py::arg("mesh"), py::arg("energy_store"), py::arg("idx"),
          py::arg("vals"), py::arg("min_t"), py::arg("max_t"),
          py::arg("temp") = 1.0);
    m.def("move_global", &move_global, "Test global vertex markov step",
          py::arg("mesh"), py::arg("energy_store"), py::arg("min_t"),
          py::arg("max_t"), py::arg("temp") = 1.0);
    m.def("flip_serial", &flip_serial, "Test global flip markov step",
          py::arg("mesh"), py::arg("energy_store"), py::arg("idx"),
          py::arg("min_t"), py::arg("max_t"), py::arg("temp") = 1.0);


    // reduced (self and one-ring excluded) neighbour list
    py::class_<trimem::NeighbourLists<true,true>>(m, "rNeighbourList")
        .def(py::init<const TriMesh&, double, double>(),
             "Init verlet list.",
             py::arg("mesh"), py::arg("rlist"), py::arg("eps") = 1.0e-6)
        .def_readwrite("neighbours",
                       &trimem::NeighbourLists<true,true>::neighbours)
        .def("distance_matrix",
             &trimem::NeighbourLists<true, true>::distance_matrix,
             "Compute distance matrix.",
             py::arg("mesh"), py::arg("dmax"))
        .def("distance_counts",
             &trimem::NeighbourLists<true, true>::distance_counts,
             "Count distances <= dmax.",
             py::arg("mesh"), py::arg("dmax"));

    // standard neighbour list
    py::class_<trimem::NeighbourLists<false,false>>(m, "NeighbourList")
        .def(py::init<const TriMesh&, double, double>(),
             "Init verlet list.",
             py::arg("mesh"), py::arg("rlist"), py::arg("eps") = 1.0e-6)
        .def_readwrite("neighbours",
                       &trimem::NeighbourLists<false,false>::neighbours)
        .def("distance_matrix",
             &trimem::NeighbourLists<false, false>::distance_matrix,
             "Compute distance matrix.",
             py::arg("mesh"), py::arg("dmax"))
        .def("distance_counts",
             &trimem::NeighbourLists<false, false>::distance_counts,
             "Count distances <= dmax.",
             py::arg("mesh"), py::arg("dmax"));

    // cell list
    py::class_<trimem::CellList>(m, "CellList")
        .def(py::init<const TriMesh&, double, double>(), "Init cell list",
             py::arg("mesh"), py::arg("rlist"), py::arg("eps") = 1.0e-6)
        .def_readwrite("cells", &trimem::CellList::cells)
        .def_readwrite("shape", &trimem::CellList::shape)
        .def_readwrite("strides", &trimem::CellList::strides)
        .def_readwrite("r_list", &trimem::CellList::r_list)
        .def_readwrite("cell_pairs", &trimem::CellList::cell_pairs)
        .def("distance_matrix",
             &trimem::CellList::distance_matrix<false, false>,
             "Compute distance matrix.",
             py::arg("mesh"), py::arg("dmax"))
        .def("r_distance_matrix",
             &trimem::CellList::distance_matrix<true, true>,
             "Compute distance matrix with self and one-ring excluded.",
             py::arg("mesh"), py::arg("dmax"))
        .def("distance_counts",
             &trimem::CellList::distance_counts<false, false>,
             "Count distances <= dmax.",
             py::arg("mesh"), py::arg("dmax"))
        .def("r_distance_counts",
             &trimem::CellList::distance_counts<true, true>,
             "Count distances <= dmax (with self and one-ring excluded).",
             py::arg("mesh"), py::arg("dmax"));
}
