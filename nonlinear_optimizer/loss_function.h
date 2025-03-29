#ifndef NONLINEAR_OPTIOMIZER_LOSS_FUNCTION_H_
#define NONLINEAR_OPTIOMIZER_LOSS_FUNCTION_H_

#include <cmath>
#include <stdexcept>

namespace nonlinear_optimizer {

class LossFunction {
 public:
  LossFunction() {}

  virtual void Evaluate(const double squared_residual, double* output) = 0;
};

class ExponentialLossFunction : public LossFunction {
 public:
  ExponentialLossFunction(const double c1, const double c2)
      : c1_{c1}, c2_{c2}, two_c1c2_{2.0 * c1_ * c2_} {
    if (c1_ < 0.0) throw std::out_of_range("`c1_` should be positive number.");
    if (c2_ < 0.0) throw std::out_of_range("`c2_` should be positive number.");
  }

  void Evaluate(const double squared_residual, double output[2]) final {
    const double exp_term = std::exp(-c2_ * squared_residual);
    output[0] = c1_ - c1_ * exp_term;
    output[1] = two_c1c2_ * exp_term;
    output[2] = -2.0 * c2_ * output[1];
  }

 private:
  double c1_{0.0};
  double c2_{0.0};
  double two_c1c2_{0.0};
};

class HuberLossFunction : public LossFunction {
 public:
  HuberLossFunction(const double threshold)
      : threshold_{threshold}, squared_threshold_{threshold_ * threshold_} {
    if (threshold_ <= 0.0)
      throw std::out_of_range("threshold value should be larger than zero.");
  }

  void Evaluate(const double squared_residual, double output[2]) final {
    if (squared_residual > squared_threshold_) {
      const double residual = std::sqrt(squared_residual);
      output[0] = 2.0 * threshold_ * residual - squared_threshold_;
      output[1] = threshold_ / residual;
    } else {
      output[0] = squared_residual;
      output[1] = 1.0;
    }
  }

 private:
  double threshold_{0.0};
  const double squared_threshold_;
};

}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIOMIZER_LOSS_FUNCTION_H_