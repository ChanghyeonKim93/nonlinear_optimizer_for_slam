#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_ARM_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_ARM_H_

#include <iostream>

#if defined(__ARM_ARCH) || defined(__aarch64__)

// ARM-based CPU
// Required CMake flag: -o2 -ftree-vectorize -mtune=cortex-a72 (your cpu arch.
// model)

// Reference: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
#include "arm_neon.h"

#define _SIMD_DATA_STEP_FLOAT 4
#define _SIMD_FLOAT float32x4_t
#define _SIMD_SET1 vmovq_n_f32
#define _SIMD_LOAD vld1q_f32   // float32x4_t vld1q_f32(float32_t const *ptr)
#define _SIMD_RCP vrecpeq_f32  // float32x4_t vrecpeq_f32(float32x4_t a)
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

#endif

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_ARM_H_