#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_AMD_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_AMD_H_

#include <iostream>

#if defined(__amd64__) || defined(__x86_64__)

#include "immintrin.h"

// AMD CPU (Intel, AMD)
#define _SIMD_DATA_STEP_FLOAT 8
#define _SIMD_FLOAT __m256
#define _SIMD_SET1 _mm256_set1_ps
#define _SIMD_LOAD _mm256_load_ps
#define _SIMD_ADD _mm256_add_ps
#define _SIMD_SUB _mm256_sub_ps
#define _SIMD_MUL _mm256_mul_ps
#define _SIMD_DIV _mm256_div_ps
#define _SIMD_RCP _mm256_rcp_ps
#define _SIMD_STORE _mm256_store_ps

#define _SIMD_DATA_STEP_DOUBLE 4
#define _SIMD_DOUBLE __m256d
#define _SIMD_SET1_D _mm256_set1_pd
#define _SIMD_LOAD_D _mm256_load_pd
#define _SIMD_ADD_D _mm256_add_pd
#define _SIMD_SUB_D _mm256_sub_pd
#define _SIMD_MUL_D _mm256_mul_pd
#define _SIMD_RCP_D _mm256_div_pd
#define _SIMD_STORE_D _mm256_store_pd

namespace nonlinear_optimizer {
namespace simd {

class ScalarF {
 public:
  ScalarF() { data_ = _mm256_setzero_ps(); }

  explicit ScalarF(const float scalar) { data_ = _mm256_set1_ps(scalar); }

  explicit ScalarF(const float n1, const float n2, const float n3,
                   const float n4, const float n5, const float n6,
                   const float n7, const float n8) {
    data_ = _mm256_set_ps(n8, n7, n6, n5, n4, n3, n2, n1);
  }

  explicit ScalarF(const float* rhs) { data_ = _mm256_load_ps(rhs); }

  ScalarF(const __m256& rhs) { data_ = rhs; }

  ScalarF(const ScalarF& rhs) { data_ = rhs.data_; }

  ScalarF operator<(const float scalar) const {
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_LT_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const float scalar) const {
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_LE_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>(const float scalar) const {
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_GT_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const float scalar) const {
    // Convert mask to 0.0 or 1.0
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_GE_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator<(const ScalarF& rhs) const {
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LT_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const ScalarF& rhs) const {
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LE_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>(const ScalarF& rhs) const {
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GT_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const ScalarF& rhs) const {
    // Convert mask to 0.0 or 1.0
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GE_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF& operator=(const ScalarF& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  ScalarF operator+(const float rhs) const {
    return ScalarF(_mm256_add_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator-() const {
    return ScalarF(_mm256_sub_ps(_mm256_set1_ps(0.0f), data_));
  }

  ScalarF operator-(const float rhs) const {
    return ScalarF(_mm256_sub_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator*(const float rhs) const {
    return ScalarF(_mm256_mul_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator/(const float rhs) const {
    return ScalarF(_mm256_div_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator+(const ScalarF& rhs) const {
    return ScalarF(_mm256_add_ps(data_, rhs.data_));
  }

  ScalarF operator-(const ScalarF& rhs) const {
    return ScalarF(_mm256_sub_ps(data_, rhs.data_));
  }

  ScalarF operator*(const ScalarF& rhs) const {
    return ScalarF(_mm256_mul_ps(data_, rhs.data_));
  }

  ScalarF operator/(const ScalarF& rhs) const {
    return ScalarF(_mm256_div_ps(data_, rhs.data_));
  }

  ScalarF& operator+=(const ScalarF& rhs) {
    data_ = _mm256_add_ps(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator-=(const ScalarF& rhs) {
    data_ = _mm256_sub_ps(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator*=(const ScalarF& rhs) {
    data_ = _mm256_mul_ps(data_, rhs.data_);
    return *this;
  }

  void StoreData(float* data) const { _mm256_store_ps(data, data_); }

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
  __m256 data_;
};

class ScalarD {
 public:
  ScalarD() { data_ = _mm256_setzero_pd(); }
  explicit ScalarD(const double scalar) { data_ = _mm256_set1_pd(scalar); }
  explicit ScalarD(const double n1, const double n2, const double n3,
                   const double n4) {
    data_ = _mm256_set_pd(n4, n3, n2, n1);
  }
  explicit ScalarD(const double* rhs) { data_ = _mm256_load_pd(rhs); }
  ScalarD(const __m256d& rhs) { data_ = rhs; }
  ScalarD(const ScalarD& rhs) { data_ = rhs.data_; }

  ScalarD operator<(const double scalar) const {
    ScalarD comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_LT_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator<=(const double scalar) const {
    ScalarD comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_LE_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator>(const double scalar) const {
    ScalarD comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_GT_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator>=(const double scalar) const {
    // Convert mask to 0.0 or 1.0
    ScalarD comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_GE_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator<(const ScalarD& rhs) const {
    ScalarD comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_LT_OS),
                                    _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator<=(const ScalarD& rhs) const {
    ScalarD comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_LE_OS),
                                    _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator>(const ScalarD& rhs) const {
    ScalarD comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_GT_OS),
                                    _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD operator>=(const ScalarD& rhs) const {
    // Convert mask to 0.0 or 1.0
    ScalarD comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_GE_OS),
                                    _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  ScalarD& operator=(const ScalarD& rhs) {
    data_ = rhs.data_;
    return *this;
  }
  ScalarD operator+(const double rhs) const {
    return ScalarD(_mm256_add_pd(data_, _mm256_set1_pd(rhs)));
  }
  ScalarD operator-() const {
    return ScalarD(_mm256_sub_pd(_mm256_set1_pd(0.0), data_));
  }
  ScalarD operator-(const double rhs) const {
    return ScalarD(_mm256_sub_pd(data_, _mm256_set1_pd(rhs)));
  }
  ScalarD operator*(const double rhs) const {
    return ScalarD(_mm256_mul_pd(data_, _mm256_set1_pd(rhs)));
  }
  ScalarD operator/(const double rhs) const {
    return ScalarD(_mm256_div_pd(data_, _mm256_set1_pd(rhs)));
  }
  ScalarD operator+(const ScalarD& rhs) const {
    return ScalarD(_mm256_add_pd(data_, rhs.data_));
  }
  ScalarD operator-(const ScalarD& rhs) const {
    return ScalarD(_mm256_sub_pd(data_, rhs.data_));
  }
  ScalarD operator*(const ScalarD& rhs) const {
    return ScalarD(_mm256_mul_pd(data_, rhs.data_));
  }
  ScalarD operator/(const ScalarD& rhs) const {
    return ScalarD(_mm256_div_pd(data_, rhs.data_));
  }
  ScalarD& operator+=(const ScalarD& rhs) {
    data_ = _mm256_add_pd(data_, rhs.data_);
    return *this;
  }
  ScalarD& operator-=(const ScalarD& rhs) {
    data_ = _mm256_sub_pd(data_, rhs.data_);
    return *this;
  }
  ScalarD& operator*=(const ScalarD& rhs) {
    data_ = _mm256_mul_pd(data_, rhs.data_);
    return *this;
  }

  void StoreData(double* data) const { _mm256_store_pd(data, data_); }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const ScalarD& scalar) {
    double multi_scalars[4];
    scalar.StoreData(multi_scalars);
    std::cout << "[["
              << "[" << multi_scalars[0] << "],\n"
              << "[" << multi_scalars[1] << "],\n"
              << "[" << multi_scalars[2] << "],\n"
              << "[" << multi_scalars[3] << "]]" << std::endl;
    return outputStream;
  }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_DOUBLE; }

 private:
  __m256d data_;
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif
#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_AMD_H_