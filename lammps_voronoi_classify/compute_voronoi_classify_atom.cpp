// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Voronoi-based 40-dim feature for binary classification.
   Feature = 10 groups * 4: group1 = center (dist=0, vol, nfaces, type);
   groups 2-10 = 9 nearest neighbors (dist, vol, nfaces, type) by distance.
   Uses compute voronoi/atom and neighbor list (no global top-K).
------------------------------------------------------------------------- */

#include "compute_voronoi_classify_atom.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "force.h"
#include "update.h"

#ifdef LAMMPS_TORCH
#include <torch/script.h>
#endif

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace LAMMPS_NS;

static int compare_neigh(const void *a, const void *b)
{
  const ComputeVoronoiClassifyAtom::NeighDist *pa =
      (const ComputeVoronoiClassifyAtom::NeighDist *) a;
  const ComputeVoronoiClassifyAtom::NeighDist *pb =
      (const ComputeVoronoiClassifyAtom::NeighDist *) b;
  if (pa->rsq < pb->rsq) return -1;
  if (pa->rsq > pb->rsq) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

ComputeVoronoiClassifyAtom::ComputeVoronoiClassifyAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), list(nullptr), c_voro(nullptr), id_voro(nullptr),
  model_path(nullptr), ntypegroups(0), type_groupbit(nullptr), type_group_ids(nullptr),
  nmax(0), feat(nullptr), class_out(nullptr),
  torch_module(nullptr), use_model(false)
{
  // LAMMPS 约定：arg[0]=ID, arg[1]=group, arg[2]=style, arg[3]=第一个参数（这里是 c_voronoi_id）
  if (narg < 4)
    error->all(FLERR, "Illegal compute voronoi/classify/atom command: "
                     "compute ID group voronoi/classify/atom c_voronoi_id [type g1 g2 ...] [model.pt]");

  id_voro = utils::strdup(arg[3]);
  int iarg = 4;

  if (iarg < narg && strcmp(arg[iarg], "type") == 0) {
    iarg++;
    int start = iarg;
    int n = 0;
    while (iarg < narg && strstr(arg[iarg], ".pt") == nullptr)
      n++, iarg++;
    if (n == 0)
      error->all(FLERR, "Compute voronoi/classify/atom: 'type' keyword requires at least one group ID");
    ntypegroups = n;
    type_group_ids = new char*[n];
    for (int k = 0; k < n; k++)
      type_group_ids[k] = utils::strdup(arg[start + k]);
  }

  if (iarg < narg && strstr(arg[iarg], ".pt") != nullptr) {
    use_model = true;
    model_path = utils::strdup(arg[iarg]);
    iarg++;
  }
  if (iarg < narg)
    error->all(FLERR, "Illegal compute voronoi/classify/atom command: unexpected argument");

  peratom_flag = 1;
  size_peratom_cols = use_model ? 1 : NFEAT;
}

/* ---------------------------------------------------------------------- */

ComputeVoronoiClassifyAtom::~ComputeVoronoiClassifyAtom()
{
  delete[] id_voro;
  delete[] model_path;
  if (type_group_ids) {
    for (int k = 0; k < ntypegroups; k++) delete[] type_group_ids[k];
    delete[] type_group_ids;
  }
  delete[] type_groupbit;
  memory->destroy(feat);
  memory->destroy(class_out);
#ifdef LAMMPS_TORCH
  if (torch_module)
    delete static_cast<torch::jit::script::Module *>(torch_module);
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeVoronoiClassifyAtom::init()
{
  c_voro = modify->get_compute_by_id(id_voro);
  if (!c_voro)
    error->all(FLERR, "Compute voronoi/classify/atom: compute ID {} not found", id_voro);
  if (!utils::strmatch(c_voro->style, "^voronoi/atom$"))
    error->all(FLERR, "Compute voronoi/classify/atom: {} is not compute voronoi/atom", id_voro);

  if (c_voro->size_peratom_cols < 2)
    error->all(FLERR, "Compute voronoi/classify/atom: voronoi/atom must provide at least 2 cols (vol, nfaces)");

  if (force->pair == nullptr)
    error->all(FLERR, "Compute voronoi/classify/atom requires a pair style");

  if (ntypegroups > 0) {
    type_groupbit = new int[ntypegroups];
    for (int k = 0; k < ntypegroups; k++) {
      int ig = group->find(type_group_ids[k]);
      if (ig < 0)
        error->all(FLERR, "Compute voronoi/classify/atom: type group {} not found", type_group_ids[k]);
      type_groupbit[k] = group->bitmask[ig];
    }
  }

  if (use_model) {
#ifdef LAMMPS_TORCH
    try {
      torch::jit::script::Module *m = new torch::jit::script::Module(torch::jit::load(model_path));
      torch_module = static_cast<void *>(m);
      (*m).eval();
    } catch (const c10::Error &e) {
      error->all(FLERR, "Compute voronoi/classify/atom: failed to load TorchScript model: {}",
                 e.msg());
    }
#else
    error->all(FLERR, "Compute voronoi/classify/atom: model file given but LAMMPS was not "
                      "built with LibTorch. Rebuild with -DLAMMPS_TORCH and link LibTorch.");
#endif
  }

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void ComputeVoronoiClassifyAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

int ComputeVoronoiClassifyAtom::get_feature_type(int atom_index, int *mask) const
{
  if (ntypegroups == 0)
    return atom->type[atom_index];
  for (int g = 0; g < ntypegroups; g++)
    if (mask[atom_index] & type_groupbit[g])
      return g + 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void ComputeVoronoiClassifyAtom::build_feature(int i, int jnum, int *jlist,
    double **x, int *mask, double **voro)
{
  double *f = feat[i];
  // group 1: center atom (distance = 0, volume, nfaces, type)
  f[0] = 0.0;
  f[1] = voro[i][0];
  f[2] = voro[i][1];
  f[3] = (double) get_feature_type(i, mask);

  if (jnum == 0) {
    for (int k = 1; k < NGROUPS; k++) {
      f[k*NPER + 0] = 0.0;
      f[k*NPER + 1] = 0.0;
      f[k*NPER + 2] = 0.0;
      f[k*NPER + 3] = 0.0;
    }
    return;
  }

  // collect neighbors with rsq and sort by distance
  NeighDist *nd = (NeighDist *) memory->smalloc(jnum * sizeof(NeighDist), "voronoi/classify/atom:nd");
  int n = 0;
  double xi[3] = { x[i][0], x[i][1], x[i][2] };

  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj] & 0x1FFFFFFF;  // NEIGHMASK
    double dx = x[j][0] - xi[0];
    double dy = x[j][1] - xi[1];
    double dz = x[j][2] - xi[2];
    double rsq = dx*dx + dy*dy + dz*dz;
    nd[n].j = j;
    nd[n].rsq = rsq;
    n++;
  }
  qsort(nd, n, sizeof(NeighDist), compare_neigh);

  // fill groups 2..10 from nearest 9 neighbors

  int k, g;
  for (g = 1; g < NGROUPS && (g-1) < n; g++) {
    k = g - 1;
    int j = nd[k].j;
    double r = std::sqrt(nd[k].rsq);
    f[g*NPER + 0] = r;
    f[g*NPER + 1] = voro[j][0];
    f[g*NPER + 2] = voro[j][1];
    f[g*NPER + 3] = (double) get_feature_type(j, mask);
  }
  for (; g < NGROUPS; g++) {
    f[g*NPER + 0] = 0.0;
    f[g*NPER + 1] = 0.0;
    f[g*NPER + 2] = 0.0;
    f[g*NPER + 3] = 0.0;
  }
  memory->sfree(nd);
}

/* ---------------------------------------------------------------------- */

void ComputeVoronoiClassifyAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // invoke voronoi/atom (fills volume, nfaces per atom)
  if (!(c_voro->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_voro->compute_peratom();
    c_voro->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  if (c_voro->comm_forward > 0)
    comm->forward_comm(c_voro);

  neighbor->build_one(list);

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  double **voro = c_voro->array_atom;
  if (!voro)
    error->all(FLERR, "Compute voronoi/classify/atom: voronoi/atom did not return array_atom");

  if (atom->nmax > nmax) {
    memory->destroy(feat);
    memory->destroy(class_out);
    nmax = atom->nmax;
    memory->create(feat, nmax, NFEAT, "voronoi/classify/atom:feat");
    if (use_model)
      memory->create(class_out, nmax, 1, "voronoi/classify/atom:class_out");
    array_atom = use_model ? class_out : feat;
  }

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if (mask[i] & groupbit) {
      build_feature(i, numneigh[i], firstneigh[i], x, mask, voro);
    } else {
      for (int c = 0; c < NFEAT; c++) feat[i][c] = 0.0;
    }
  }

  if (use_model) {
#ifdef LAMMPS_TORCH
    torch::jit::script::Module *m = static_cast<torch::jit::script::Module *>(torch_module);
    int nlocal = atom->nlocal;
    if (nlocal > 0) {
      try {
        torch::NoGradGuard no_grad;

        // 将 LAMMPS 的 double** feat 拷贝到一块连续的 float 缓冲区，再构造张量
        std::vector<float> buffer(nlocal * NFEAT);
        for (int i = 0; i < nlocal; i++) {
          for (int c = 0; c < NFEAT; c++) {
            buffer[i * NFEAT + c] = static_cast<float>(feat[i][c]);
          }
        }

        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
        auto input_t = torch::from_blob(
                           buffer.data(),
                           {nlocal, 1, NGROUPS, NPER},
                           options)
                           .clone();

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_t);
        torch::Tensor out = m->forward(inputs).toTensor();
        out = out.argmax(1);
        for (int i = 0; i < nlocal; i++)
          class_out[i][0] = static_cast<double>(out[i].item<int64_t>());
      } catch (const c10::Error &e) {
        error->one(FLERR, "Compute voronoi/classify/atom: model forward failed: {}", e.msg());
      }
    }
    for (int i = 0; i < nlocal; i++)
      if (!(mask[i] & groupbit))
        class_out[i][0] = 0.0;
#endif
  }
}

/* ---------------------------------------------------------------------- */

double ComputeVoronoiClassifyAtom::memory_usage()
{
  double bytes = (double) NFEAT * nmax * sizeof(double);
  if (use_model && class_out)
    bytes += (double) nmax * sizeof(double);
  return bytes;
}
