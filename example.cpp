////////////////////////////////////////////
// File: example.cpp                      //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////

// Includes
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "qpbo_alpha_expand.h"

// Data.
typedef Eigen::Map<const Eigen::MatrixXd> MapConstMatrixXd;

// Nodes (`X`).
const int kDim = 2;
const int kNumNodePoints = 2;
const double kNodePoints[kNumNodePoints * kDim] = {
  0.0, 0.0,
  1.0, 0.0
};
const MapConstMatrixXd kX(kNodePoints, kDim, kNumNodePoints);

// Data points (`Y`).
// Generated with the following Python code.
//   import numpy as np
//   np.random.seed(1337)
//   X = 0.10 * np.random.randn(32 * 2).reshape(32, 2)
//   X[16:] += (1.0, 0.0)
//   np.savetxt('X.txt', X, fmt='%+.6e')
const int kNumDataPoints = 32;
const double kDataPoints[kNumDataPoints * kDim] = {
  -7.031873e-02, -4.902824e-02,
  -3.218143e-02, -1.755079e-01,
  +2.066645e-02, -2.011265e-01,
  -5.572507e-02, +3.372170e-02,
  +1.548836e-01, -1.370737e-01,
  +1.425291e-01, -2.794639e-02,
  -5.596279e-02, +1.186383e-01,
  +1.698519e-01, -1.691220e-01,
  -6.995228e-02, +5.829628e-02,
  +9.782226e-02, -1.217372e-01,
  -1.329395e-01, -1.454742e-04,
  -1.314653e-01, -3.796117e-02,
  +1.265211e-01, +1.206677e-02,
  +1.479418e-02, -2.753726e-01,
  -3.568963e-02, +7.717837e-04,
  +1.478277e-01, -9.576146e-02,
  +1.132901e+00, -9.858496e-02,
  +1.047156e+00, -8.746529e-04,
  +1.036702e+00, +1.118555e-01,
  +9.991610e-01, +4.663154e-02,
  +1.126327e+00, -9.016547e-02,
  +8.971157e-01, +5.696784e-02,
  +1.064166e+00, +2.598119e-02,
  +1.119318e+00, -1.046300e-01,
  +1.013989e+00, -1.730656e-01,
  +9.869377e-01, -1.310260e-01,
  +7.828688e-01, -1.066181e-01,
  +9.966838e-01, +1.466396e-01,
  +1.087664e+00, +6.699896e-02,
  +1.069745e+00, -2.527854e-02,
  +1.056799e+00, +3.043879e-02,
  +8.999970e-01, -2.456418e-01
};
const MapConstMatrixXd kY(kDataPoints, kDim, kNumDataPoints);

// Functors.
// SquaredDistanceFunctor
class SquaredDistanceFunctor {
 public:
  typedef double Scalar;
  typedef MapConstMatrixXd::Index Index;

  SquaredDistanceFunctor(const MapConstMatrixXd* X0,
                         const MapConstMatrixXd* X1,
                         const Scalar& w=1.0)
    : X0_(X0), X1_(X1), w_(w)
  {}

  Scalar operator()(Index i, Index j) const {
    return w_ * (X0_->col(i) - X1_->col(j)).squaredNorm();
  }

 private:
  const MapConstMatrixXd* X0_;
  const MapConstMatrixXd* X1_;
  const Scalar w_;
};

// UnaryFunctor
class UnaryFunctor : public SquaredDistanceFunctor {
 public:
  UnaryFunctor(const MapConstMatrixXd* X,
               const MapConstMatrixXd* Y,
               const Scalar& w=1.0)
    : SquaredDistanceFunctor(Y, X, w),
      num_nodes_(X->cols()),
      num_labels_(Y->cols())
  {}

  Index num_nodes() const { return num_nodes_; }
  Index num_labels() const { return num_labels_; }

 private:
  const Index num_nodes_;
  const Index num_labels_;
};

// main
int main() {
  // `G` describes the connectivity of the problem.
  typedef Eigen::SparseMatrix<int, Eigen::RowMajor> Graph;
  Graph G(2, 2);
  G.insert(0, 1) = 1;

  // `l` is the label vector.
  Eigen::VectorXi l(2);

  // Solve the problem using `UnaryFunctor` and `SquaredDistanceFunctor`.
  {
    std::cout << "Unary: UnaryFunctor" << std::endl;
    std::cout << "Pairwise: SquaredDistanceFunctor" << std::endl;

    UnaryFunctor U(&kX, &kY, 2.0);
    SquaredDistanceFunctor P(&kY, &kY, 1.0);
    std::vector<const SquaredDistanceFunctor*> Ps;
    Ps.push_back(&P);

    l.setZero();
    std::cout << "  Initial `l`: " << l.transpose() << std::endl;

    qpbo::alpha_expand(U, Ps, G, qpbo::AllLabelsAreValid(), &l, 100);
    std::cout << "  Final `l`: " << l.transpose() << std::endl;
  }

  // Solve the problem by evaluating the unary and pairwise matrices.
  {
    std::cout << "Unary: Eigen::MatrixXd" << std::endl;
    std::cout << "Pairwise: Eigen::MatrixXd" << std::endl;

    typedef MapConstMatrixXd::Index Index;
    const Index num_nodes = kX.cols(), num_labels = kY.cols();
    Eigen::MatrixXd U(num_labels, num_nodes);
    for (Index i = 0; i < num_nodes; ++i) {
      for (Index j = 0; j < num_labels; ++j) {
        U(j, i) = 2.0 * (kY.col(j) - kX.col(i)).squaredNorm();
      }
    }

    Eigen::MatrixXd P(num_labels, num_labels);
    P.setZero();
    for (Index i = 0; i < num_labels; ++i) {
      for (Index j = i + 1; j < num_labels; ++j) {
        P(j, i) = P(i, j) = (kY.col(i) - kY.col(j)).squaredNorm();
      }
    }

    std::vector<const Eigen::MatrixXd*> Ps;
    Ps.push_back(&P);

    // Solve using `AllLabelsAreValid` for `label_validator` ...
    std::cout << "(AllLabelsAreValid)" << std::endl;
    l.setZero();
    std::cout << "  Initial `l`: " << l.transpose() << std::endl;

    qpbo::alpha_expand(U, Ps, G, qpbo::AllLabelsAreValid(), &l, 100);
    std::cout << "  Final `l`: " << l.transpose() << std::endl;

    // ... and again with `ValidIfInSortedVector`.
    Eigen::VectorXi sorted_labels(num_labels);
    sorted_labels.setLinSpaced(0, static_cast<int>(num_labels) - 1);

    std::vector<const Eigen::VectorXi*> valid;
    valid.push_back(&sorted_labels); // Node 0.
    valid.push_back(&sorted_labels); // Node 1.

    qpbo::ValidIfInSortedVector<Eigen::VectorXi> label_validator(&valid);

    std::cout << "(IfInSortedVector)" << std::endl;
    l.setZero();
    std::cout << "  Initial `l`: " << l.transpose() << std::endl;

    qpbo::alpha_expand(U, Ps, G, label_validator, &l, 100);
    std::cout << "  Final `l`: " << l.transpose() << std::endl;
  }

  return 0;
}
