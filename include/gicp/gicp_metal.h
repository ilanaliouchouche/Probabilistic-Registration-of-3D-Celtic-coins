#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "gicp.h"

namespace gicp {
namespace metal {

enum class NeighborSearchMethod {
    DIRECT1,
    DIRECT7,
    DIRECT27,
    DIRECT_RADIUS,
};

enum class CovarianceMode {
    KNN,
    VOXEL_APPROX,
};

struct Config {
    int max_iter = 20;
    int k_neighbors = 20;
    double epsilon = 1e-3;

    // Optional voxel downsampling before registration (<=0 disables).
    double downsample_resolution = 0.0;

    // Voxelized correspondences (VGICP-style)
    double voxel_resolution = 1.0;
    NeighborSearchMethod neighbor_method = NeighborSearchMethod::DIRECT27;
    double neighbor_radius = 1.0;
    double max_correspondence_distance = 1.0;

    // Covariance estimation mode.
    CovarianceMode covariance_mode = CovarianceMode::VOXEL_APPROX;
    double covariance_voxel_resolution = 0.0;  // <=0 => use voxel_resolution

    // If true, fallback to CPU GICP when Metal is unavailable.
    bool allow_cpu_fallback = true;
};

struct Result {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    int iterations = 0;
    double cov_ms = 0.0;
    double total_ms = 0.0;
    double objective = 0.0;

    bool used_gpu = false;
    std::string backend = "cpu";
};

Result registerPointClouds(const PointCloud& source,
                           const PointCloud& target,
                           const Config& cfg = Config());

}  // namespace metal
}  // namespace gicp
