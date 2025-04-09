#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_COST_FUNCTOR_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_COST_FUNCTOR_H_

#include "types.h"

#include "Eigen/Dense"
#include "ceres/ceres.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

class RelativePoseCostFunctor {
 public:
  RelativePoseCostFunctor(const Constraint& constraint)
      : constraint_(constraint) {}

  template <typename T>
  bool operator()(const T* const reference_translation_ptr,
                  const T* const reference_orientation_ptr,
                  const T* const query_position_ptr,
                  const T* const query_orientation_ptr, T* residual_ptr) const {
    using CeresVec3 = Eigen::Matrix<T, 3, 1>;
    using CeresQuaternion = Eigen::Quaternion<T>;
    CeresVec3 reference_position(reference_translation_ptr[0],
                                 reference_translation_ptr[1],
                                 reference_translation_ptr[2]);
    CeresQuaternion reference_orientation(
        reference_orientation_ptr[0], reference_orientation_ptr[1],
        reference_orientation_ptr[2], reference_orientation_ptr[3]);
    CeresVec3 query_position(query_position_ptr[0], query_position_ptr[1],
                             query_position_ptr[2]);
    CeresQuaternion query_orientation(
        query_orientation_ptr[0], query_orientation_ptr[1],
        query_orientation_ptr[2], query_orientation_ptr[3]);

    CeresVec3 relative_translation_from_reference_to_query =
        constraint_.relative_pose_from_reference_to_query.translation()
            .cast<T>();
    CeresQuaternion relative_rotation_from_reference_to_query =
        Eigen::Quaterniond(
            constraint_.relative_pose_from_reference_to_query.rotation())
            .cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual.template block<3, 1>(0, 0) =
        (query_position - reference_position) -
        reference_orientation * relative_translation_from_reference_to_query;
    CeresQuaternion error_q = query_orientation.conjugate() *
                              reference_orientation *
                              relative_rotation_from_reference_to_query;
    residual.template block<3, 1>(3, 0) = T(2.0) * error_q.vec();
    return true;
  }

  static ceres::CostFunction* Create(const Constraint& constraint) {
    constexpr int kDimResidual = 6;
    constexpr int kDimTrans = 3;
    constexpr int kDimQuaternion = 4;
    return new ceres::AutoDiffCostFunction<
        RelativePoseCostFunctor, kDimResidual, kDimTrans, kDimQuaternion,
        kDimTrans, kDimQuaternion>(new RelativePoseCostFunctor(constraint));
  }

 private:
  const Constraint constraint_;
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_CERES_COST_FUNCTOR_H_