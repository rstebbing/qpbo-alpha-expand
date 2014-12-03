////////////////////////////////////////////
// File: qpbo_alpha_expand.h              //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////
#ifndef QPBO_ALPHA_EXPAND_H
#define QPBO_ALPHA_EXPAND_H

// Includes
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <memory> // For `unique_ptr`.
#include <vector>

#include "Eigen/Dense"

#include "QPBO.h"

// qpbo
namespace qpbo {

// AllLabelsAreValid
// A label validator which enables all labels for all nodes.
struct AllLabelsAreValid {
  template <typename Index>
  bool operator()(Index j, Index i) const {
    return true;
  }
};

// ValidIfInSortedVector
// A label validator which checks if label `j` is present in a vector of
// (sorted) available labels for node `i`.
// Importantly, the size of `valid` must be equal to the number of nodes in
// the problem, and the vectors in `valid` must be sorted. Both of these
// conditions are assumed and are the responsibility of the caller.
template <typename T>
class ValidIfInSortedVector {
 public:
  ValidIfInSortedVector(const std::vector<const T*>* valid)
    : valid_(valid)
  {}

  template <typename Index>
  bool operator()(Index j, Index i) const {
    auto& v = (*valid_)[i];
    return std::binary_search(v->data(), v->data() + v->size(), j);
  }

 private:
  const std::vector<const T*>* valid_;
};

// energy
// Return the energy of the current labelling `l`.
template <typename Unary,
          typename Pairwise,
          typename Graph,
          typename Labels>
auto energy(const Unary& U,
            const std::vector<const Pairwise*>& Ps,
            const Graph& G,
            const Labels& l) -> typename Unary::Scalar {
  typename Unary::Scalar e(0);

  for (typename Graph::Index i = 0; i < G.outerSize(); ++i) {
    e += U(l[i], i);

    for (typename Graph::InnerIterator it(G, i); it; ++it) {
      e += (*Ps[it.value() - 1])(l[it.col()], l[i]);
    }
  }

  return e;
}

// has_cols
// `has_cols<T>::value` is true if class `T` has a `cols` method and false
// otherwise.
template <typename T>
class has_cols {
  typedef char Yes;
  class No { char c[2]; };

  template <typename U> static Yes f(U* u, decltype(u->cols())* = nullptr);
  template <typename U> static No f(...);

 public:
  static const bool value = sizeof(f<T>(nullptr)) == sizeof(Yes);
};

// dimensions
// Sets `num_nodes` and `num_labels` using the `cols` and `rows` methods
// from `U`.
template <typename Unary>
inline void dimensions(const Unary& U,
                       typename Unary::Index* num_nodes,
                       typename Unary::Index* num_labels,
                       std::true_type) {
  *num_nodes = U.cols();
  *num_labels = U.rows();
}

// dimensions
// Sets `num_nodes` and `num_labels` using the `num_nodes` and `num_labels`
// methods from `U`.
template <typename Unary>
inline void dimensions(const Unary& U,
                       typename Unary::Index* num_nodes,
                       typename Unary::Index* num_labels,
                       std::false_type) {
  *num_nodes = U.num_nodes();
  *num_labels = U.num_labels();
}

// dimensions
// Set `num_nodes` and `num_labels` by calling the appropriate `dimensions`
// overload based on whether or not `Unary` has a `cols` method.
template <typename Unary>
inline void dimensions(const Unary& U,
                       typename Unary::Index* num_nodes,
                       typename Unary::Index* num_labels) {
  dimensions(U, num_nodes, num_labels,
             std::integral_constant<bool, has_cols<Unary>::value>());
}

// alpha_expand
//
// Inputs:
//
// `U` is the object which provides the retrieval (or evaluation) of the energy
// of assigning label `j` to node `i`.
//
// A minimum example implementation with the necessary types and methods is:
//
// class Unary {
//  public:
//   typedef ... Scalar;  // Energy data type.
//   typedef ... Index;   // Node and label index data type.
//
//   Scalar operator()(Index j, Index i) const {
//     // Return the energy for assigning label `j` to node `i`.
//     ...
//   }
//
//   Index num_nodes() const { // Return the number of nodes. ... }
//   Index num_labels() const { // Return the number of labels. ... }
//   // OR
//   Index cols() const { // Return the number of nodes. ... }
//   Index rows() const { // Return the number of labels. ... }
// };
//
// `Ps` is a vector of pointers to `Pairwise` objects. Each `Pairwise` object
// provides the retrieval (or evaluation) of the energy of adjacent nodes
// taking labels `i` and `j`. (Importantly, this is assumed to be symmetric.)
//
// A minimum example implementation with the necessary types and methods is:
//
// class Pairwise {
//  public:
//   typedef ... Scalar;  // Data type for pairwise energies.
//
//   Scalar operator()(Index j, Index i) const {
//     // Return the energy of adjacent nodes taking labels `j` and `i`.
//     // (Which is equal to the energy of adjacent nodes taking labels `i` and
//     // `j`.)
//     ...
//   }
// };
//
// `G` describes the connectivity of the nodes. If `G(i, j)` is non-zero, then
// between nodes `i` and `j`, `Ps[G(i, j) - 1]` is used to evaluate the
// pairwise energy.
//
// Importantly, the values of `G` are NOT checked to be valid 1-based indices
// into `Ps`; this is the responsibility of the caller.
//
// `G` should be an Eigen CSR matrix such as:
//   Eigen::Sparse<int, Eigen::RowMajor> or
//   Eigen::MappedSparseMatrix<int, Eigen::RowMajor>
// and of dimensions `(num_nodes, num_nodes)`.
//
// `label_validator` is a functor which determines whether label `j` can be
// assigned to node `i`.
//
// A minimal example is:
//
// struct AllLabelsAreValid {
//   template <typename Index>
//   bool operator()(Index j, Index i) const {
//     return true;
//   }
// };
//
// `labels` is an Eigen::Vector (or similar type) of size `num_nodes`. Element
// `i` of `labels` specifies the label for node `i`, and on return of
// `alpha_expand`, will be set to the optimal labelling determined by the QPBO
// alpha-expansion moves.
//
// Importantly, the initial values of `labels` are NOT checked to be valid
// labels; this is the responsibility of the caller.
//
// `max_num_iterations` is the maximum number of outer iterations to perform.
// For each outer iteration, alpha-expansion moves are performed for each
// possible label.
//
// Returns:
//
// `alpha_expand` returns the energy corresponding to the final labelling
// (which is stored in `labels`).
template <typename Unary,
          typename Pairwise,
          typename Graph,
          typename LabelValidator,
          typename Labels>
auto alpha_expand(const Unary& U,
                  const std::vector<const Pairwise*> & Ps,
                  const Graph& G,
                  const LabelValidator& label_validator,
                  Labels* labels,
                  int max_num_iterations) -> typename Unary::Scalar
{
  static_assert(std::is_same<typename Unary::Scalar,
                             typename Pairwise::Scalar>::value,
                "U and Ps must have the same scalar type.");
  static_assert(Graph::IsRowMajor,
                "G must be row major.");
  static_assert(std::is_integral<typename Graph::Scalar>::value,
                "G must have an integral type.");
  static_assert(std::is_integral<typename Labels::Scalar>::value,
                "Labels must have an integral type.");

  // `Scalar` can be floating-point or integral type.
  typedef typename Unary::Scalar Scalar;
  typedef typename Unary::Index Index;

  typedef Eigen::Matrix<Index, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXu;
  typedef Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXi;

  // Use `dimensions` to determine `num_nodes` and `num_labels` from `U`.
  Index num_nodes = 0, num_labels = 0;
  dimensions(U, &num_nodes, &num_labels);

  // `l` is a copy of `labels`, `p` are temporal labels, `ei` are effective
  // indices.
  VectorXu l(labels->template cast<Index>()), p(num_nodes);
  VectorXi ei(num_nodes);

  Scalar last_e = energy(U, Ps, G, l);

  if (max_num_iterations < 0) {
    max_num_iterations = std::numeric_limits<int>::max();
  }

  int i = 0;
  bool is_done = (i >= max_num_iterations);

  while (!is_done) {
    ++i;
    Scalar current_e = last_e;

    // For each `alpha`, solve a binary sub-problem. The binary sub-problem is:
    // Do nodes with labels not equal to `alpha` --- and which can take `alpha`
    // --- keep their current label or change to `alpha`?
    for (Index alpha = 0; alpha < num_labels; ++alpha) {
      ei.fill(-1);

      int num_effective_nodes = 0;

      for (Index j = 0; j < num_nodes; ++j) {
        if (l[j] == alpha || !label_validator(alpha, j)) {
          continue;
        }

        ei[j] = num_effective_nodes++;
      }

      if (num_effective_nodes == 0) {
        continue;
      }

      // Setup `QPBO` instance.
      // NOTE `G.nonZeros` is the upper limit on the number of edges in the
      // binary sub-problem.
      std::unique_ptr<QPBO<Scalar>> qpbo(
        new QPBO<Scalar>(num_effective_nodes, G.nonZeros()));

      qpbo->AddNode(num_effective_nodes);

      // Add unary and pairwise terms.
      for (Index j = 0; j < num_nodes; ++j) {
        // Skip nodes which don't contribute to this sub-problem.
        if (ei[j] < 0) {
          continue;
        }

        // `E0` is the energy of retaining the current label;
        // `E1` is the energy of node `j` changing to label `alpha`.
        Scalar E0 = U(l[j], j);
        Scalar E1 = U(alpha, j);

        for (typename Graph::InnerIterator it(
              G, static_cast<typename Graph::Index>(j)); it; ++it) {
          auto const & P = *Ps[it.value() - 1];
          Index k = it.col();

          // If node `k` does not take part in the binary sub-problem, then
          // "fold" the edge into the unary.
          if (ei[k] < 0) {
            E0 += P(l[k], l[j]);
            E1 += P(l[k], alpha);
          } else {
            // Otherwise, add the pairwise term.
            qpbo->AddPairwiseTerm(
              static_cast<typename QPBO<Scalar>::NodeId>(ei[j]),
              static_cast<typename QPBO<Scalar>::NodeId>(ei[k]),
              P(l[k], l[j]), P(alpha, l[j]), P(l[k], alpha), P(alpha, alpha));
          }
        }

        qpbo->AddUnaryTerm(ei[j], E0, E1);
      }

      // Solve.
      qpbo->MergeParallelEdges();

      qpbo->Solve();
      qpbo->ComputeWeakPersistencies();

      // `p` will be the updated label vector after the alpha-expansion move.
      p = l;
      int num_unlabelled = 0;

      // Update `p` with the results of the alpha-expansion.
      for (Index j = 0; j < num_nodes; ++j) {
        if (ei[j] < 0) {
          continue;
        }

        int xi = qpbo->GetLabel(ei[j]);
        if (xi == 1) {
          p[j] = alpha;
        } else {
          num_unlabelled += (xi < 0);
        }
      }

      // If the resulting expansion decreases the overall problem energy,
      // accept the new labelling.
      Scalar e = energy(U, Ps, G, p);

      if (e < current_e) {
        l = p;
        current_e = e;
      }
    }

    // Exit if there is no decrease in energy.
    is_done = (current_e >= last_e) || (i >= max_num_iterations);
    last_e = current_e;
  }

  // Save final labels and return the final energy.
  *labels = l.template cast<typename Labels::Scalar>();
  return last_e;
}

} // namespace qpbo

#endif // QPBO_ALPHA_EXPAND_H
