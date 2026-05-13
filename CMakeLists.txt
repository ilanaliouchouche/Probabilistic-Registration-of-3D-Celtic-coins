cmake_minimum_required(VERSION 3.20)

project(GeneralizedICP LANGUAGES CXX)

if(APPLE)
  enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  nanoflann
  URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.9.0.tar.gz
)
FetchContent_GetProperties(nanoflann)
if(NOT nanoflann_POPULATED)
  FetchContent_Populate(nanoflann)
endif()

add_library(gicp src/gicp.cpp)

target_include_directories(gicp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${nanoflann_SOURCE_DIR}/include
)

target_link_libraries(gicp PUBLIC Eigen3::Eigen)

if(APPLE)
  add_library(gicp_metal src/gicp_metal.mm)
else()
  add_library(gicp_metal src/gicp_metal_stub.cpp)
endif()

target_include_directories(gicp_metal PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${nanoflann_SOURCE_DIR}/include
)

target_link_libraries(gicp_metal PUBLIC gicp Eigen3::Eigen)

if(APPLE)
  target_link_libraries(gicp_metal PUBLIC
    "-framework Foundation"
    "-framework Metal"
    "-framework CoreGraphics"
  )
  target_compile_definitions(gicp_metal PRIVATE
    GICP_METAL_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/metal/gicp_vgicp.metal"
  )
endif()

add_library(icp src/icp.cpp)
target_include_directories(icp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${nanoflann_SOURCE_DIR}/include
)
target_link_libraries(icp PUBLIC Eigen3::Eigen)

add_executable(coin_benchmark examples/coin_benchmark.cpp)
target_link_libraries(coin_benchmark PRIVATE icp gicp)

add_executable(riedones3d_noise examples/noise_robustness.cpp)
target_link_libraries(riedones3d_noise PRIVATE icp gicp)

add_executable(gicp_icp_coin examples/gicp_icp_coin.cpp)
target_link_libraries(gicp_icp_coin PRIVATE icp gicp)

add_executable(gicp_metal_demo examples/gicp_metal_demo.cpp)
target_link_libraries(gicp_metal_demo PRIVATE gicp_metal)
