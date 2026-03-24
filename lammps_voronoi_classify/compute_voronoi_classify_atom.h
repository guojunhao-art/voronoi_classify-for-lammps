/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(voronoi/classify/atom,ComputeVoronoiClassifyAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_VORONOI_CLASSIFY_ATOM_H
#define LMP_COMPUTE_VORONOI_CLASSIFY_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeVoronoiClassifyAtom : public Compute {
 public:
  ComputeVoronoiClassifyAtom(class LAMMPS *, int, char **);
  ~ComputeVoronoiClassifyAtom() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

  static const int NFEAT = 40;   // 10 groups * 4 values
  static const int NGROUPS = 10;  // center + 9 neighbors
  static const int NPER = 4;       // distance, volume, nfaces, type

  // public so that compare_neigh (qsort callback, file-scope) can use the type
  struct NeighDist {
    int j;
    double rsq;
  };

 private:
  class NeighList *list;
  class Compute *c_voro;
  char *id_voro;
  char *model_path;   // optional TorchScript .pt path; null = feature-only
  int ntypegroups;    // 0 = use atom->type; >0 = use group membership as type 1,2,...
  int *type_groupbit; // groupbit for each type group (length ntypegroups)
  char **type_group_ids; // group IDs for type groups (freed in destructor)
  int nmax;
  double **feat;      // 40-dim feature buffer (always used to build input)
  double **class_out; // when model_path set: per-atom 0/1, array_atom points here
  void *torch_module; // torch::jit::script::Module* when LAMMPS_TORCH defined
  bool use_model;
  int get_feature_type(int atom_index, int *mask) const;
  void build_feature(int i, int jnum, int *jlist, double **x, int *mask, double **voro);
};

}    // namespace LAMMPS_NS

#endif
#endif
