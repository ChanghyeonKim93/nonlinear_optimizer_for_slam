project(utility_tests)
message("  -- [SUBPROJECT]: " ${PROJECT_NAME})

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -pthread")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message("-- Detected architecture: x86_64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -msse -mavx -mavx2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("-- Detected architecture: ARM64 (aarch64)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -ftree-vectorize -march=armv8-a+simd -mtune=cortex-a72")
else()
  message("-- Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

include_directories(../)

add_executable(
  ${PROJECT_NAME}_simd_helper_v2_test
  simd_helper_v2_test.cc
)
target_include_directories(
  ${PROJECT_NAME}_simd_helper_v2_test 
  PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS} ${GTEST_INCLUDE_DIR}
)
target_link_libraries(
  ${PROJECT_NAME}_simd_helper_v2_test
  PUBLIC ${GTEST_BOTH_LIBRARIES}
         Threads::Threads
)
add_test(NAME ${PROJECT_NAME}_simd_helper_v2_test
         COMMAND ${PROJECT_NAME}_simd_helper_v2_test
)