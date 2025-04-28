#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_H_

#include <iostream>

#include "Eigen/Dense"

#include "nonlinear_optimizer/simd_helper/simd_helper_amd.h"
#include "nonlinear_optimizer/simd_helper/simd_helper_arm.h"

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
namespace simd {

template <typename DataType>
inline DataType* GetAlignedMemory(const size_t num_data) {
  return reinterpret_cast<DataType*>(
      std::aligned_alloc(ALIGN_BYTES, num_data * sizeof(DataType)));
}

template <typename DataType>
inline void FreeAlignedMemory(DataType* ptr) {
  if (ptr != nullptr) std::free(reinterpret_cast<void*>(ptr));
}

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_H_