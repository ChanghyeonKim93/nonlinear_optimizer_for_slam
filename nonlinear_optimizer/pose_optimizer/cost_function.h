#ifndef NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_COST_FUNCTION_H_
#define NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_COST_FUNCTION_H_

#include "../simd_helper_v2/simd_helper.h"

using Pose = Eigen::Isometry3d;

namespace nonlinear_optimizer {
namespace pose_optimizer {

class LossFunction {
 public:
  virtual ~LossFunction() {}

  virtual void Evaluate(const double& squared_residual, double output[3]) = 0;
};

template <int kDimTranslation, int kDimRotation>
class CostFunction {
 protected:
  static constexpr int kDimPose = kDimTranslation + kDimRotation;
  using RotationMatrix = Eigen::Matrix<double, kDimRotation, kDimRotation>;
  using TranslationVector = Eigen::Matrix<double, kDimTranslation, 1>;
  using HessianMatrix = Eigen::Matrix<double, kDimPose, kDimPose>;
  using GradientVector = Eigen::Matrix<double, kDimPose, 1>;

 public:
  virtual ~CostFunction() {}

  virtual bool Evaluate(const RotationMatrix& rotation_matrix,
                        const TranslationVector& translation,
                        double* jacobian_matrix_ptr,
                        double* residual_vector_ptr) = 0;

  int GetDimResidual() const { return dim_residual_; }

  int GetDimPose() const { return kDimPose; }

  int GetDimTranslation() const { return kDimTranslation; }

  int GetDimRotation() const { return kDimRotation; }

 protected:
  void SetDimResidual(const int dim_residual) { dim_residual_ = dim_residual; }

 private:
  int dim_residual_{-1};
};

template <int kDimResidual, int kDimTranslation, int kDimRotation>
class SizedCostFunction : public CostFunction<kDimTranslation, kDimRotation> {
 public:
  SizedCostFunction() { this->SetDimResidual(kDimResidual); }

  virtual ~SizedCostFunction() {}
};

template <int kDimTranslation, int kDimRotation>
class ResidualBlock {
 protected:
  static constexpr int kDimPose = kDimTranslation + kDimRotation;
  using RotationMatrix = Eigen::Matrix<double, kDimRotation, kDimRotation>;
  using TranslationVector = Eigen::Matrix<double, kDimTranslation, 1>;
  using HessianMatrix = Eigen::Matrix<double, kDimPose, kDimPose>;
  using GradientVector = Eigen::Matrix<double, kDimPose, 1>;

 public:
  ResidualBlock(CostFunction<kDimTranslation, kDimRotation>* cost_function,
                LossFunction* loss_function = nullptr)
      : cost_function_(cost_function), loss_function_(loss_function) {}

  bool Evaluate(const RotationMatrix& rotation_matrix,
                const TranslationVector& translation,
                HessianMatrix* local_hessian_ptr,
                GradientVector* local_gradient_ptr, double* cost) {
    static std::vector<double> jacobian(kDimPose * kDimPose);
    static std::vector<double> residual(kDimPose * kDimPose);
    if (local_hessian_ptr == nullptr || local_gradient_ptr == nullptr ||
        cost == nullptr) {
      return false;
    }

    const int dim_residual = cost_function_->GetDimResidual();
    jacobian.resize(dim_residual * kDimPose, 0.0);
    residual.resize(dim_residual, 0.0);
    if (!cost_function_->Evaluate(rotation_matrix, translation, jacobian.data(),
                                  residual.data())) {
      return false;
    }

    local_hessian_ptr->setZero();
    local_gradient_ptr->setZero();
    ComputeUpperTriangularHessian(jacobian.data(), dim_residual,
                                  local_hessian_ptr);
    const double squared_residual = ComputeSquaredResidual(residual.data());
    if (loss_function_) {
      *cost = squared_residual;
    } else {
      double loss_output[3];
      loss_function_->Evaluate(squared_residual, loss_output);
      *cost = squared_residual;
      const double weight = loss_output[1];
      MultiplyWeight(weight, local_hessian_ptr, local_gradient_ptr);
    }

    return true;
  }

 private:
  void ComputeUpperTriangularHessian(const double* jacobian_data_ptr,
                                     const int dim_residual,
                                     HessianMatrix* hessian_matrix) {
    const double* jac_col_i = 0;
    for (int i = 0; i < kDimPose; ++i, jac_col_i += dim_residual) {
      const double* jac_col_j = jac_col_i;
      for (int j = i; j < kDimPose; ++j, jac_col_j += dim_residual)
        for (int k = 0; k < dim_residual; ++k)
          (*hessian_matrix)(i, j) += jac_col_i[k] * jac_col_j[k];
    }
  }

  void MultiplyWeight(const double weight, HessianMatrix* hessian_matrix,
                      GradientVector* gradient_vector) {
    for (int i = 0; i < kDimPose; ++i) {
      (*gradient_vector)(i) *= weight;
      for (int j = i; j < kDimPose; ++j) (*hessian_matrix)(i, j) *= weight;
    }
  }

  double ComputeSquaredResidual(const double* residual_data_ptr,
                                const int dim_residual) {
    double squared_residual = 0.0;
    for (int i = 0; i < dim_residual; ++i)
      squared_residual += residual_data_ptr[i] * residual_data_ptr[i];
    return squared_residual;
  }

  CostFunction<kDimTranslation, kDimRotation>* cost_function_{nullptr};
  LossFunction* loss_function_{nullptr};
};

}  // namespace pose_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_COST_FUNCTION_H_