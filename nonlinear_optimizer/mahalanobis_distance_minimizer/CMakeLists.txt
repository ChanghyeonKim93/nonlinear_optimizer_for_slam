project(${CMAKE_PROJECT_NAME}_mahalanobis_distance_minimizer)
message("  -- [SUBPROJECT]: " ${PROJECT_NAME})

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)

set(UTILITY_SRC ${CMAKE_SOURCE_DIR}/nonlinear_optimizer/time_checker.cc)

# tests
add_executable(
  ${PROJECT_NAME}_tests_simple_optimization_test
  tests/simple_optimization_test.cc mahalanobis_distance_minimizer_ceres.cc mahalanobis_distance_minimizer_analytic.cc
  ${UTILITY_SRC})
target_include_directories(
  ${PROJECT_NAME}_tests_simple_optimization_test
  PUBLIC ${CMAKE_SOURCE_DIR} # Include the parent CMake file's directory
         ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${PC_FLANN_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_tests_simple_optimization_test
                      PRIVATE ${CERES_LIBRARIES} ${FLANN_LIBRARIES})