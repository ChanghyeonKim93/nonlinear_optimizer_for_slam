#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_H_

#include "immintrin.h"

#include "Eigen/Dense"

#define _SIMD_DATA_STEP_DOUBLE 4
#define _SIMD_DATA_STEP_FLOAT 8
#define _SIMD_FLOAT __m256
#define _SIMD_DOUBLE __m256d
#define _SIMD_SET1 _mm256_set1_ps
#define _SIMD_LOAD _mm256_load_ps
#define _SIMD_RCP _mm256_rcp_ps
#define _SIMD_DIV _mm256_div_ps
#define _SIMD_MUL _mm256_mul_ps
#define _SIMD_ADD _mm256_add_ps
#define _SIMD_SUB _mm256_sub_ps
#define _SIMD_STORE _mm256_store_ps
#define _SIMD_SET1_D _mm256_set1_pd
#define _SIMD_LOAD_D _mm256_load_pd
#define _SIMD_RCP_D _mm256_div_pd
#define _SIMD_MUL_D _mm256_mul_pd
#define _SIMD_ADD_D _mm256_add_pd
#define _SIMD_SUB_D _mm256_sub_pd
#define _SIMD_STORE_D _mm256_store_pd

#define ALIGN_BYTES 32
// AVX2 (512 bits = 64 Bytes), AVX (256 bits = 32 Bytes), SSE4.2 (128 bits = 16
// Bytes)
/** \internal Like malloc, but the returned pointer is guaranteed to be 32-byte
 * aligned. Fast, but wastes 32 additional bytes of memory. Does not throw any
 * exception.
 *
 * (256 bits) two LSB addresses of 32 bytes-aligned : 00, 20, 40, 60, 80, A0,
 * C0, E0 (128 bits) two LSB addresses of 16 bytes-aligned : 00, 10, 20, 30, 40,
 * 50, 60, 70, 80, 90, A0, B0, C0, D0, E0, F0
 */

namespace nonlinear_optimizer {

inline void* custom_aligned_malloc(std::size_t size) {
  void* original = std::malloc(size + ALIGN_BYTES);  // size+ALIGN_BYTES
  if (original == 0)
    return nullptr;  // if allocation is failed, return nullptr;
  void* aligned =
      reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) &
                               ~(std::size_t(ALIGN_BYTES - 1))) +
                              ALIGN_BYTES);
  *(reinterpret_cast<void**>(aligned) - 1) = original;
  return aligned;
}

/** \internal Frees memory allocated with handmade_aligned_malloc */
inline void custom_aligned_free(void* ptr) {
  if (ptr) std::free(*(reinterpret_cast<void**>(ptr) - 1));
}

class SimdFloat {
 public:
  SimdFloat() { data_ = _mm256_setzero_ps(); }
  explicit SimdFloat(const float scalar) { data_ = _mm256_set1_ps(scalar); }
  explicit SimdFloat(const float n1, const float n2, const float n3,
                     const float n4, const float n5, const float n6,
                     const float n7, const float n8) {
    data_ = _mm256_set_ps(n8, n7, n6, n5, n4, n3, n2, n1);
  }
  explicit SimdFloat(const float* rhs) { data_ = _mm256_load_ps(rhs); }
  SimdFloat(const __m256& rhs) { data_ = rhs; }
  SimdFloat(const SimdFloat& rhs) { data_ = rhs.data_; }
  SimdFloat operator=(const SimdFloat& rhs) { return SimdFloat(rhs.data_); }
  SimdFloat operator+(const float rhs) const {
    return SimdFloat(_mm256_add_ps(data_, _mm256_set1_ps(rhs)));
  }
  SimdFloat operator-(const float rhs) const {
    return SimdFloat(_mm256_sub_ps(data_, _mm256_set1_ps(rhs)));
  }
  SimdFloat operator*(const float rhs) const {
    return SimdFloat(_mm256_mul_ps(data_, _mm256_set1_ps(rhs)));
  }
  SimdFloat operator/(const float rhs) const {
    return SimdFloat(_mm256_div_ps(data_, _mm256_set1_ps(rhs)));
  }
  SimdFloat operator+(const SimdFloat& rhs) const {
    return SimdFloat(_mm256_add_ps(data_, rhs.data_));
  }
  SimdFloat operator-(const SimdFloat& rhs) const {
    return SimdFloat(_mm256_sub_ps(data_, rhs.data_));
  }
  SimdFloat operator*(const SimdFloat& rhs) const {
    return SimdFloat(_mm256_mul_ps(data_, rhs.data_));
  }
  SimdFloat operator/(const SimdFloat& rhs) const {
    return SimdFloat(_mm256_div_ps(data_, rhs.data_));
  }

  SimdFloat& operator+=(const SimdFloat& rhs) {
    data_ = _mm256_add_ps(data_, rhs.data_);
    return *this;
  }
  SimdFloat& operator-=(const SimdFloat& rhs) {
    data_ = _mm256_sub_ps(data_, rhs.data_);
    return *this;
  }
  SimdFloat& operator*=(const SimdFloat& rhs) {
    data_ = _mm256_mul_ps(data_, rhs.data_);
    return *this;
  }

  void StoreData(float* data) const { _mm256_store_ps(data, data_); }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_FLOAT; }

 private:
  __m256 data_;
};

class SimdDouble {
 public:
  SimdDouble() { data_ = _mm256_setzero_pd(); }
  explicit SimdDouble(const double scalar) { data_ = _mm256_set1_pd(scalar); }
  explicit SimdDouble(const double n1, const double n2, const double n3,
                      const double n4) {
    data_ = _mm256_set_pd(n4, n3, n2, n1);
  }
  explicit SimdDouble(const double* rhs) { data_ = _mm256_load_pd(rhs); }
  SimdDouble(const __m256d& rhs) { data_ = rhs; }
  SimdDouble(const SimdDouble& rhs) { data_ = rhs.data_; }
  SimdDouble operator=(const SimdDouble& rhs) { return SimdDouble(rhs.data_); }
  SimdDouble operator+(const double rhs) const {
    return SimdDouble(_mm256_add_pd(data_, _mm256_set1_pd(rhs)));
  }
  SimdDouble operator-(const double rhs) const {
    return SimdDouble(_mm256_sub_pd(data_, _mm256_set1_pd(rhs)));
  }
  SimdDouble operator*(const double rhs) const {
    return SimdDouble(_mm256_mul_pd(data_, _mm256_set1_pd(rhs)));
  }
  SimdDouble operator/(const double rhs) const {
    return SimdDouble(_mm256_div_pd(data_, _mm256_set1_pd(rhs)));
  }
  SimdDouble operator+(const SimdDouble& rhs) const {
    return SimdDouble(_mm256_add_pd(data_, rhs.data_));
  }
  SimdDouble operator-(const SimdDouble& rhs) const {
    return SimdDouble(_mm256_sub_pd(data_, rhs.data_));
  }
  SimdDouble operator*(const SimdDouble& rhs) const {
    return SimdDouble(_mm256_mul_pd(data_, rhs.data_));
  }
  SimdDouble operator/(const SimdDouble& rhs) const {
    return SimdDouble(_mm256_div_pd(data_, rhs.data_));
  }

  SimdDouble& operator+=(const SimdDouble& rhs) {
    data_ = _mm256_add_pd(data_, rhs.data_);
    return *this;
  }
  SimdDouble& operator-=(const SimdDouble& rhs) {
    data_ = _mm256_sub_pd(data_, rhs.data_);
    return *this;
  }
  SimdDouble& operator*=(const SimdDouble& rhs) {
    data_ = _mm256_mul_pd(data_, rhs.data_);
    return *this;
  }

  void StoreData(double* data) const { _mm256_store_pd(data, data_); }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_DOUBLE; }

 private:
  __m256d data_;
};

}  // namespace nonlinear_optimizer

#endif