#include "../simd_helper_v2/simd_helper.h"

#include "cost_functors.h"
#include "pose_optimizer.h"

#include <memory>

using namespace nonlinear_optimizer::pose_optimizer;

int main(int argc, char** argv) {
  constexpr int kDimTranslation = 3;
  constexpr int kDimRotation = 3;
  constexpr int kDimPose = kDimTranslation + kDimRotation;

  LossFunction* loss_function = nullptr;

  loss_function = new ExponentialLossFunction(1.0, 0.1);
  double loss_output[3];
  loss_function->Evaluate(0.5, loss_output);

  Correspondence corr;
  corr.point.setRandom();
  corr.mean.setRandom();
  corr.sqrt_information.setIdentity();

  std::shared_ptr<CostFunction<kDimTranslation, kDimRotation>> cost_function =
      std::make_shared<MahalanobisDistanceCostFunctor>(corr);

  Eigen::Matrix<double, 3, kDimPose> jacobian_matrix;
  Eigen::Matrix<double, 3, 1> residual_vector;

  cost_function->Evaluate(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
                          jacobian_matrix.data(), residual_vector.data());

  std::cerr << "Point : " << corr.point.transpose() << std::endl;
  std::cerr << "jacobian_matrix:\n" << jacobian_matrix << std::endl;
  std::cerr << "residual_vector:\n" << residual_vector << std::endl;

  // Problem Test

  return 0;
}