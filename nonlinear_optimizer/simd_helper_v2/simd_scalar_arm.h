#ifndef SIMD_HELPER_V2_SIMD_SCALAR_ARM_H_
#define SIMD_HELPER_V2_SIMD_SCALAR_ARM_H_

#if defined(__ARM_ARCH) || defined(__aarch64__)

#include <iostream>
#include "arm_neon.h"

namespace {

float32x4_t __one{vmovq_n_f32(1.0f)};
float32x4_t __minus_one{vmovq_n_f32(-1.0f)};
float32x4_t __zero{vmovq_n_f32(0.0f)};
uint32_t __DATA_STRIDE{4};

}  // namespace

namespace simd {

class Scalar {
 public:
  // static member methods
  static inline size_t GetDataStride() { return __DATA_STRIDE; }

  static inline Scalar Zeros() { return Scalar(0.0f); }

  static inline Scalar Ones() { return Scalar(1.0f); }

 public:
  // Initialization & Assignment operations
  Scalar() { data_ = __zero; }

  Scalar(const float scalar) { data_ = vmovq_n_f32(scalar); }

  Scalar(const float n1, const float n2, const float n3, const float n4) {
    data_ = __zero;
    data_ = vsetq_lane_f32(n1, data_, 0);
    data_ = vsetq_lane_f32(n2, data_, 1);
    data_ = vsetq_lane_f32(n3, data_, 2);
    data_ = vsetq_lane_f32(n4, data_, 3);
  }

  Scalar(const float* rhs) { data_ = vld1q_f32(rhs); }

  Scalar(const Scalar& rhs) { data_ = rhs.data_; }

  Scalar(const float32x4_t& rhs) { data_ = rhs; }

  Scalar& operator=(const float rhs) {
    data_ = vmovq_n_f32(rhs);
    return *this;
  }

  Scalar& operator=(const Scalar& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  // Comparison operations
  Scalar operator<(const float scalar) const {
    return vbslq_f32(vcltq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
  }

  Scalar operator<=(const float scalar) const {
    return vbslq_f32(vcleq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
  }

  Scalar operator>(const float scalar) const {
    return vbslq_f32(vcgtq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
  }

  Scalar operator>=(const float scalar) const {
    return vbslq_f32(vcgeq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
  }

  Scalar operator<(const Scalar& rhs) const {
    return vbslq_f32(vcltq_f32(data_, rhs.data_), __one, __zero);
  }

  Scalar operator<=(const Scalar& rhs) const {
    return vbslq_f32(vcleq_f32(data_, rhs.data_), __one, __zero);
  }

  Scalar operator>(const Scalar& rhs) const {
    return vbslq_f32(vcgtq_f32(data_, rhs.data_), __one, __zero);
  }

  Scalar operator>=(const Scalar& rhs) const {
    return vbslq_f32(vcgeq_f32(data_, rhs.data_), __one, __zero);
  }

  // Arithmetic operations
  Scalar operator+() const { return Scalar(data_); }

  Scalar operator-() const { return Scalar(vsubq_f32(__zero, data_)); }

  Scalar operator+(const float rhs) const {
    return Scalar(vaddq_f32(data_, vdupq_n_f32(rhs)));
  }

  Scalar operator-(const float rhs) const {
    return Scalar(vsubq_f32(data_, vdupq_n_f32(rhs)));
  }

  Scalar operator*(const float rhs) const {
    return Scalar(vmulq_f32(data_, vdupq_n_f32(rhs)));
  }

  Scalar operator/(const float rhs) const {
    return Scalar(vdivq_f32(data_, vdupq_n_f32(rhs)));
  }

  Scalar operator+(const Scalar& rhs) const {
    return Scalar(vaddq_f32(data_, rhs.data_));
  }

  Scalar operator-(const Scalar& rhs) const {
    return Scalar(vsubq_f32(data_, rhs.data_));
  }

  Scalar operator*(const Scalar& rhs) const {
    return Scalar(vmulq_f32(data_, rhs.data_));
  }

  Scalar operator/(const Scalar& rhs) const {
    return Scalar(vdivq_f32(data_, rhs.data_));
  }

  // Compound assignment operations
  Scalar& operator+=(const Scalar& rhs) {
    data_ = vaddq_f32(data_, rhs.data_);
    return *this;
  }

  Scalar& operator-=(const Scalar& rhs) {
    data_ = vsubq_f32(data_, rhs.data_);
    return *this;
  }

  Scalar& operator*=(const Scalar& rhs) {
    data_ = vmulq_f32(data_, rhs.data_);
    return *this;
  }

  // Some useful operations
  Scalar sqrt() const { return Scalar(vsqrtq_f32(data_)); }

  // Store SIMD data to normal memory
  void StoreData(float* normal_memory) const {
    vst1q_f32(normal_memory, data_);
  }

  // Debug functions
  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const Scalar& scalar) {
    float multi_scalars[__DATA_STRIDE];
    scalar.StoreData(multi_scalars);
    std::cout << "[";
    for (int i = 0; i < __DATA_STRIDE; ++i) {
      std::cout << "[" << multi_scalars[i] << "]";
      if (i != __DATA_STRIDE - 1) std::cout << ",\n";
    }
    std::cout << "]" << std::endl;
    return outputStream;
  }

 private:
  float32x4_t data_;
};

}  // namespace simd

#endif  // defined(__ARM_ARCH) || defined(__aarch64__)
#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_AMD_H_