////////////////////////////////////////////
// File: qpbo_alpha_expand_pyx.h          //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////

// Includes
#include <type_traits>
#include <vector>

// Requires util/cpp
#include "Math/linalg_python.h"
using namespace linalg_python;

#include "qpbo_alpha_expand.h"

// Types
typedef double DTYPE_t;

// qpbo_alpha_expand
PyObject* qpbo_alpha_expand(PyArrayObject* npy_U,
                            PyObject* py_Ps,
                            PyObject* py_G,
                            PyObject* py_valid_labels,
                            PyArrayObject* npy_indices_in_valid_labels,
                            PyArrayObject* npy_l,
                            int max_num_iterations) {
  auto U = PyArrayObject_to_MatrixMap<DTYPE_t>(npy_U);
  auto _Ps = PyList_to_VectorOfMatrixMap<DTYPE_t>(py_Ps);
  auto G = ScipyCSRMatrix_to_CSRMatrixMap<int>(py_G);
  auto valid_labels = PyList_to_VectorOfVectorMap<int>(py_valid_labels);
  auto indices_in_valid_labels = PyArrayObject_to_VectorMap<int>(
    npy_indices_in_valid_labels);
  auto l = PyArrayObject_to_VectorMap<int>(npy_l);

  typedef std::decay<decltype(*_Ps[0])>::type Pairwise;
  std::vector<const Pairwise*> Ps;
  for (auto& P : _Ps) {
    Ps.push_back(P.get());
  }

  typedef std::decay<decltype(*valid_labels[0])>::type LabelVector;
  std::vector<const LabelVector*> valid;
  for (LabelVector::Index i = 0;
       i < indices_in_valid_labels->size();
       ++i) {
    valid.push_back(valid_labels[(*indices_in_valid_labels)[i]].get());
  }

  qpbo::ValidIfInSortedVector<LabelVector> label_validator(&valid);

  DTYPE_t e = qpbo::alpha_expand(*U, Ps, *G, label_validator, l.get(),
                                 max_num_iterations);
  return PyFloat_FromDouble(e);
}
