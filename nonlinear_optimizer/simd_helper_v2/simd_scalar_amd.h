#ifndef SIMD_HELPER_V2_SIMD_SCALAR_AMD_H_
#define SIMD_HELPER_V2_SIMD_SCALAR_AMD_H_

#if defined(__amd64__) || defined(__x86_64__)

#include <iostream>
#include "immintrin.h"

namespace {

__m256 __one{_mm256_set1_ps(1.0f)};
__m256 __minus_one{_mm256_set1_ps(-1.0f)};
__m256 __zero{_mm256_set1_ps(0.0f)};
uint32_t __DATA_STRIDE{8};

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

  Scalar(const float scalar) { data_ = _mm256_set1_ps(scalar); }

  Scalar(const float n1, const float n2, const float n3, const float n4,
         const float n5, const float n6, const float n7, const float n8) {
    data_ = _mm256_set_ps(n8, n7, n6, n5, n4, n3, n2, n1);
  }

  Scalar(const float* rhs) { data_ = _mm256_load_ps(rhs); }

  Scalar(const Scalar& rhs) { data_ = rhs.data_; }

  Scalar(const __m256& rhs) { data_ = rhs; }

  Scalar& operator=(const float rhs) {
    data_ = _mm256_set1_ps(rhs);
    return *this;
  }

  Scalar& operator=(const Scalar& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  // Comparison operations
  Scalar operator<(const float scalar) const {
    return _mm256_and_ps(
        _mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_LT_OS), __one);
  }

  Scalar operator<=(const float scalar) const {
    return _mm256_and_ps(
        _mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_LE_OS), __one);
  }

  Scalar operator>(const float scalar) const {
    return _mm256_and_ps(
        _mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_GT_OS), __one);
  }

  Scalar operator>=(const float scalar) const {
    return _mm256_and_ps(
        _mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_GE_OS), __one);
  }

  Scalar operator<(const Scalar& rhs) const {
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LT_OS), __one);
  }

  Scalar operator<=(const Scalar& rhs) const {
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LE_OS), __one);
  }

  Scalar operator>(const Scalar& rhs) const {
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GT_OS), __one);
  }

  Scalar operator>=(const Scalar& rhs) const {
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GE_OS), __one);
  }

  // Arithmetic operations
  Scalar operator-() const { return Scalar(_mm256_sub_ps(__zero, data_)); }

  Scalar operator+(const float rhs) const {
    return Scalar(_mm256_add_ps(data_, _mm256_set1_ps(rhs)));
  }

  Scalar operator-(const float rhs) const {
    return Scalar(_mm256_sub_ps(data_, _mm256_set1_ps(rhs)));
  }

  Scalar operator*(const float rhs) const {
    return Scalar(_mm256_mul_ps(data_, _mm256_set1_ps(rhs)));
  }

  Scalar operator/(const float rhs) const {
    return Scalar(_mm256_div_ps(data_, _mm256_set1_ps(rhs)));
  }

  Scalar operator+(const Scalar& rhs) const {
    return Scalar(_mm256_add_ps(data_, rhs.data_));
  }

  Scalar operator-(const Scalar& rhs) const {
    return Scalar(_mm256_sub_ps(data_, rhs.data_));
  }

  Scalar operator*(const Scalar& rhs) const {
    return Scalar(_mm256_mul_ps(data_, rhs.data_));
  }

  Scalar operator/(const Scalar& rhs) const {
    return Scalar(_mm256_div_ps(data_, rhs.data_));
  }

  // Compound assignment operations
  Scalar& operator+=(const Scalar& rhs) {
    data_ = _mm256_add_ps(data_, rhs.data_);
    return *this;
  }

  Scalar& operator-=(const Scalar& rhs) {
    data_ = _mm256_sub_ps(data_, rhs.data_);
    return *this;
  }

  Scalar& operator*=(const Scalar& rhs) {
    data_ = _mm256_mul_ps(data_, rhs.data_);
    return *this;
  }

  // Some useful operations
  Scalar sqrt() const { return Scalar(_mm256_sqrt_ps(data_)); }

  Scalar sign() const {
    __m256 is_positive =
        _mm256_cmp_ps(data_, __zero, _CMP_GE_OS);  // data_ >= 0.0
    __m256 result = _mm256_blendv_ps(__minus_one, __one, is_positive);
    return Scalar(result);
  }

  Scalar abs() const {
    // Use bitwise AND to clear the sign bit
    __m256 result = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), data_);
    return Scalar(result);
  }

  // Store SIMD data to normal memory
  void StoreData(float* normal_memory) const {
    _mm256_store_ps(normal_memory, data_);
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
  __m256 data_;
};

}  // namespace simd

#endif  // defined(__amd64__) || defined(__x86_64__)
#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_AMD_H_