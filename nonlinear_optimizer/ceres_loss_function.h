#ifndef NONLINEAR_OPTIMIZER_CERES_LOSS_FUNCTION_H_
#define NONLINEAR_OPTIMIZER_CERES_LOSS_FUNCTION_H_

#include <cmath>
#include <stdexcept>
#include "ceres/ceres.h"

namespace nonlinear_optimizer {

class CERES_EXPORT ExponentialLoss : public ceres::LossFunction {
 public:
  explicit ExponentialLoss(double c1, double c2)
      : c1_(c1), c2_(c2), two_c1c2_(2.0 * c1_ * c2_) {}

  virtual void Evaluate(double squared_residual, double output[3]) const {
    const double exp_term = std::exp(-c2_ * squared_residual);
    output[0] = c1_ - c1_ * exp_term;
    output[1] = two_c1c2_ * exp_term;
    output[2] = -2.0 * c2_ * output[1];
  }

 private:
  const double c1_;
  const double c2_;
  const double two_c1c2_;
};

}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_CERES_LOSS_FUNCTION_H_