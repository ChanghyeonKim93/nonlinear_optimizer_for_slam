project(pose_optimizer)

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

include_directories("../")

find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

set(SRC ${SRC} )

add_executable(
  test test.cc ${SRC}
)
target_include_directories(
  test
  PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS}
)
