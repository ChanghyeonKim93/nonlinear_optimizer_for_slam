#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_COST_FUNCTOR_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_COST_FUNCTOR_H_

#include "types.h"

#include "Eigen/Dense"
#include "ceres/ceres.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

class RelativePoseCostFunctor {
 public:
  RelativePoseCostFunctor(const Constraint& constraint, const double c1,
                          const double c2)
      : constraint_(constraint), c1_(c1), c2_(c2) {}

  template <typename T>
  bool operator()(const T* const trans_a, const T* const quat_a,
                  const T* const trans_b, const T* const quat_b,
                  T* residual_ptr) const {
    Eigen::Matrix<T, 3, 1> ta(trans_a[0], trans_a[1], trans_a[2]);
    Eigen::Quaternion<T> qa(quat_a[0], quat_a[1], quat_a[2], quat_a[3]);
    Eigen::Matrix<T, 3, 1> tb(trans_b[0], trans_b[1], trans_b[2]);
    Eigen::Quaternion<T> qb(quat_b[0], quat_b[1], quat_b[2], quat_b[3]);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual.template block<3, 1>(0, 0) =
        (tb - ta) -
        qa * constraint_.relative_pose_from_reference_to_query.translation()
                 .cast<T>();
    Eigen::Quaternion<T> delta_q =
        qb.conjugate() * qa *
        Eigen::Quaterniond(
            constraint_.relative_pose_from_reference_to_query.rotation())
            .template cast<T>();
    residual.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();
    return true;
  }

  static ceres::CostFunction* Create(const Constraint& constraint,
                                     const double c1, const double c2) {
    return new ceres::AutoDiffCostFunction<RelativePoseCostFunctor, 6, 3, 4, 3,
                                           4>(
        new RelativePoseCostFunctor(constraint, c1, c2));
  }

 private:
  const Constraint constraint_;
  const double c1_;
  const double c2_;
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_CERES_COST_FUNCTOR_H_