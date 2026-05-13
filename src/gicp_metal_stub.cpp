#include "gicp/gicp_metal.h"

#include "gicp.h"

#include <chrono>

namespace gicp {
namespace metal {

Result registerPointClouds(const PointCloud& source,
                           const PointCloud& target,
                           const Config& cfg) {
    Result out;
    out.used_gpu = false;
    out.backend = "cpu_stub";

    const auto cov_start = std::chrono::steady_clock::now();
    auto [sourceCovs, targetCovs] = gicp::computeCovariances(
        source,
        target,
        cfg.k_neighbors,
        cfg.epsilon);
    const auto cov_end = std::chrono::steady_clock::now();
    out.cov_ms = std::chrono::duration<double, std::milli>(cov_end - cov_start).count();

    gicp::KDTree targetTree(3, std::cref(target));
    targetTree.index_->buildIndex();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    const auto t0 = std::chrono::steady_clock::now();
    int iter = 0;
    for (; iter < cfg.max_iter; ++iter) {
        auto [H, g] = gicp::accumulateSystem(
            source,
            target,
            sourceCovs,
            targetCovs,
            &targetTree,
            R,
            t);
        Eigen::Matrix<double, 6, 1> delta = gicp::solveDeltaXiDense(H, g);
        gicp::updateTransform(R, t, delta);
        if (delta.norm() < 1e-6) {
            ++iter;
            break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    out.R = R;
    out.t = t;
    out.iterations = iter;
    out.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

}  // namespace metal
}  // namespace gicp

