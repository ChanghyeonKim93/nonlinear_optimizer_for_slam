#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_ARM_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_ARM_H_

#include <iostream>

#if defined(__ARM_ARCH) || defined(__aarch64__)

// ARM-based CPU
// Required CMake flag: -o2 -ftree-vectorize -mtune=cortex-a72 (your cpu arch.
// model)

// Reference: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
#include "arm_neon.h"

#define _SIMD_DATA_STEP_FLOAT 4
#define _SIMD_FLOAT float32x4_t
#define _SIMD_SET1 vdupq_n_f32  // == vmovq_n_f32
#define _SIMD_LOAD vld1q_f32    // float32x4_t vld1q_f32(float32_t const *ptr)
#define _SIMD_RCP vrecpeq_f32   // float32x4_t vrecpeq_f32(float32x4_t a)
#define _SIMD_DIV \
  vdivq_f32  // float32x4_t vdivq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_MUL \
  vmulq_f32  // float32x4_t vmulq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_ADD \
  vaddq_f32  // float32x4_t vaddq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_SUB \
  vsubq_f32  // float32x4_t vsubq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_STORE \
  vst1q_f32  // void vst1q_f32(float32_t *ptr, float32x4_t val)
#define _SIMD_ROUND_FLOAT vrndaq_f32  // float32x4_t vrndq_f32(float32x4_t a)
#define _SIMD_SQRT_FLOAT vsqrtq_f32   // float32x4_t vsqrtq_f32(float32x4_t a)

namespace nonlinear_optimizer {
namespace simd {

class ScalarF {
 public:
  ScalarF() { data_ = vmovq_n_f32(0.0f); }

  explicit ScalarF(const float scalar) { data_ = vmovq_n_f32(scalar); }

  explicit ScalarF(const float n1, const float n2, const float n3,
                   const float n4) {
    data_ = vdupq_n_f32(0.0f);
    data_ = vsetq_lane_f32(n1, data_, 0);
    data_ = vsetq_lane_f32(n2, data_, 1);
    data_ = vsetq_lane_f32(n3, data_, 2);
    data_ = vsetq_lane_f32(n4, data_, 3);
  }

  explicit ScalarF(const float* rhs) { data_ = vld1q_f32(rhs); }

  ScalarF(const float32x4_t& rhs) { data_ = rhs; }

  ScalarF(const ScalarF& rhs) { data_ = rhs.data_; }

  // Note: vcgtq (a > b), vcgeq (a >= b), vcltq (a < b), vcleq (a <= b),
  // vceqq (a == b), vcneq (a != b)
  ScalarF operator<(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcltq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator<(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcltq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcleq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcleq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcgtq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcgtq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcgeq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcgeq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF& operator=(const ScalarF& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  ScalarF operator+(const float rhs) const {
    return ScalarF(vaddq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator+(const ScalarF& rhs) const {
    return ScalarF(vaddq_f32(data_, rhs.data_));
  }

  ScalarF operator-() const {
    return ScalarF(vsubq_f32(vdupq_n_f32(0.0f), data_));
  }

  ScalarF operator-(const ScalarF& rhs) const {
    return ScalarF(vsubq_f32(data_, rhs.data_));
  }

  ScalarF operator-(const float rhs) const {
    return ScalarF(vsubq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator*(const float rhs) const {
    return ScalarF(vmulq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator*(const ScalarF& rhs) const {
    return ScalarF(vmulq_f32(data_, rhs.data_));
  }

  ScalarF operator/(const float rhs) const {
    return ScalarF(vdivq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator/(const ScalarF& rhs) const {
    return ScalarF(vdivq_f32(data_, rhs.data_));
  }

  ScalarF& operator+=(const ScalarF& rhs) {
    data_ = vaddq_f32(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator-=(const ScalarF& rhs) {
    data_ = vsubq_f32(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator*=(const ScalarF& rhs) {
    data_ = vmulq_f32(data_, rhs.data_);
    return *this;
  }

  void StoreData(float* data) const { vst1q_f32(data, data_); }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const ScalarF& scalar) {
    float multi_scalars[_SIMD_DATA_STEP_FLOAT];
    scalar.StoreData(multi_scalars);
    std::cout << "[";
    for (int i = 0; i < _SIMD_DATA_STEP_FLOAT; ++i) {
      std::cout << "[" << multi_scalars[i] << "]";
      if (i != _SIMD_DATA_STEP_FLOAT - 1) std::cout << ",\n";
    }
    std::cout << "]" << std::endl;
    return outputStream;
  }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_FLOAT; }

 private:
  float32x4_t data_;
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif
#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_ARM_H_