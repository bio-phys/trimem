/** \file neighbours.h
 * \brief Neighbour list tools to be used with openmesh
 */
#ifndef CELL_LIST_H
#define CELL_LIST_H

#include <vector>
#include <map>

#include "MeshTypes.hh"

namespace trimem {

template<class real>
real distance(const real* avec, const real* bvec, int n)
{
    real dist=0.0;
    for (int i=0; i<n; i++)
    {
        real di = avec[i] - bvec[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

struct CellList
{
  //! cells -> points lookup
  std::map<int, std::vector<int> > cells;
  //! points -> cells lookup
  std::vector<int> points;
  //! shape of the cell array
  std::array<int, 3> shape;
  //! strides of cells-array
  std::array<int, 3> strides;
  //! (adjusted) r_list per dimension
  std::array<double, 3> r_list;
  //! cell pairs
  std::vector<std::pair<int, int>> cell_pairs;

  CellList(const TriMesh& mesh, double rlist, double box_eps=1.0e-6)
  {
      //compute box-dims
      std::array<double, 3> box_min = { std::numeric_limits<double>::infinity() };
      std::array<double, 3> box_max = { -1 };
      for (int i=0; i<mesh.n_vertices(); i++)
      {
          auto p = mesh.point(mesh.vertex_handle(i));
          for (int k=0; k<3; k++)
          {
              if (p[k] < box_min[k]) box_min[k] = p[k];
              if (p[k] > box_max[k]) box_max[k] = p[k];
          }
      }

      //adjust rlist per dimension and compute shape of cell_array
      std::array<int, 3> cell_min;
      for (int k=0; k<3; k++)
      {
          double dim = box_max[k] - box_min[k] + box_eps;
          int nk     = int(dim/rlist);
          r_list[k]   = (nk > 0) ? dim/nk : box_eps;
          shape[k]    = (nk > 0) ? nk : 1;
          cell_min[k] = int(box_min[k] / r_list[k]);
      }

      // compute cell coordinates, i.e. (i,j,k)-triplets indexing into the array
      std::vector<std::array<int,3>> cell_coords;
      cell_coords.reserve(mesh.n_vertices());
      for (int i=0; i<mesh.n_vertices(); i++)
      {
          auto point = mesh.point(mesh.vertex_handle(i));
          std::array<int, 3> icoords;
          for (int k=0; k<3; k++)
          {
              icoords[k] = int(point[k] / r_list[k]) - cell_min[k];
          }
          cell_coords.push_back(icoords);
      }

      // compute linear cell indices and sort vertices into CellList
      strides = { 1, shape[0], shape[0]*shape[1] };
      points.reserve(mesh.n_vertices());
      int point = 0;
      for (auto it=cell_coords.begin(); it!=cell_coords.end(); ++it, ++point)
      {
          auto v = *it;
          int id = v[0] * strides[0] + v[1] * strides[1] + v[2] * strides[1];
          cells[id].push_back(point);
          points.push_back(id);
      }

      compute_cell_pairs();
  }

  void compute_cell_pairs()
  {
      for (auto it=cells.begin(); it!=cells.end(); ++it)
      {
          int icell = it->first;

          // get icell's grid coordinates
          int kicell = icell / strides[2];
          int rem    = icell % strides[2];
          int jicell = rem / strides[1];
          int iicell = rem % strides[1];

          // loop over neighbouring cells
          for (int i=-1; i<2; i++)
          {
              int ii = iicell+i;
              if (ii < 0 or ii >= shape[0]) continue;

              for (int j=-1; j<2; j++)
              {
                  int jj = jicell+j;
                  if (jj < 0 or jj >= shape[1]) continue;

                  for (int k=-1; k<2; k++)
                  {
                      int kk = kicell+k;
                      if (kk < 0 or kk >= shape[2]) continue;

                      int ocell = ii * strides[0] + jj * strides[1] + kk * strides[2];
                      auto oit = cells.find(ocell);

                      // pass on if no vertices are in this cell
                      if (oit==cells.end()) continue;

                      if (icell <= ocell)
                          cell_pairs.push_back( { icell, ocell } );
                  }
              }
          }
      }
  }

  template<bool exclude_self=true, bool exclude_one_ring=true>
  int distance_counts(const TriMesh& mesh, const double& rlist)
  {
      const TriMesh::Point& point = mesh.point(mesh.vertex_handle(0));
      const double *data = point.data();

      int ni = 0;
      // loop over all cell pairs
      for (auto it=cell_pairs.begin(); it!=cell_pairs.end(); ++it)
      {
          const std::vector<int>& icell = cells.at(it->first);
          const std::vector<int>& ocell = cells.at(it->second);

          bool is_same_cell = ( it->first == it->second );

          // vertices in icell
          for (auto i_it=icell.begin(); i_it!=icell.end(); ++i_it)
          {
              // this vertex's coordinates
              const double* idata = data+*i_it*3;

              // vertices in ocell
              for (auto o_it=ocell.begin(); o_it!=ocell.end(); ++o_it)
              {
                  if (exclude_self)
                  {
                      if (*i_it == *o_it) continue;
                  }
                  if (exclude_one_ring)
                  {
                      bool in_ring = false;
                      auto vh = mesh.vertex_handle(*i_it);
                      for (auto vit=mesh.cvv_iter(vh); vit.is_valid(); vit++)
                      {
                          if (vit->idx() == *o_it)
                          {
                              in_ring = true;
                              break;
                          }
                      }
                      if (in_ring) continue;
                  }

                  // save some time since the distance in symmetric
                  if (is_same_cell and *i_it > *o_it) continue;

                  // other vertex's coordinates
                  const double* odata = data+*o_it*3;

                  // compute distance and count in case
                  double  dist  = distance<double>(idata, odata, 3);
                  if (dist <= rlist)
                  {
                      ni+=2;
                  }
              }
          }
      }

      return ni - mesh.n_vertices();
  }

  template<bool exclude_self=true, bool exclude_one_ring=true>
  std::tuple<std::vector<double>, std::vector<int>, std::vector<int> >
  distance_matrix(const TriMesh& mesh, const double& rlist)
  {
      const TriMesh::Point& point = mesh.point(mesh.vertex_handle(0));
      const double *data = point.data();

      // sparse matrix data
      std::vector<double> dists;
      std::vector<int>    idx;
      std::vector<int>    jdx;

      // loop over all cell pairs
      for (auto it=cell_pairs.begin(); it!=cell_pairs.end(); ++it)
      {
          const std::vector<int>& icell = cells.at(it->first);
          const std::vector<int>& ocell = cells.at(it->second);

          bool is_same_cell = ( it->first == it->second );

          // vertices in icell
          for (auto i_it=icell.begin(); i_it!=icell.end(); ++i_it)
          {
              // this vertex's coordinates
              const double* idata = data+*i_it*3;

              // vertices in ocell
              for (auto o_it=ocell.begin(); o_it!=ocell.end(); ++o_it)
              {
                  if (exclude_self)
                  {
                      if (*i_it == *o_it) continue;
                  }
                  if (exclude_one_ring)
                  {
                      bool in_ring = false;
                      auto vh = mesh.vertex_handle(*i_it);
                      for (auto vit=mesh.cvv_iter(vh); vit.is_valid(); vit++)
                      {
                          if (vit->idx() == *o_it)
                          {
                              in_ring = true;
                              break;
                          }
                      }
                      if (in_ring) continue;
                  }

                  // save some time since the distance is symmetric
                  if (is_same_cell and *i_it > *o_it) continue;

                  // other vertex's coordinates
                  const double* odata = data+*o_it*3;

                  // compute distance and count in case
                  double dist = distance<double>(idata, odata, 3);
                  if (dist <= rlist)
                  {
                      dists.push_back(dist);
                      if (*i_it < *o_it)
                      {
                          idx.push_back(*i_it);
                          jdx.push_back(*o_it);
                      }
                      else
                      {
                          idx.push_back(*o_it);
                          jdx.push_back(*i_it);
                      }
                  }
              }
          }
      }
      return std::make_tuple(std::move(dists), std::move(idx), std::move(jdx));
  }

};

}
#endif
