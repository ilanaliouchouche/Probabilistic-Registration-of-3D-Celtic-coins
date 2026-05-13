#include "gicp/gicp_metal.h"
#include "gicp.h"

#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace {

using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;

PointCloud makeCloud(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    PointCloud cloud(n, 3);
    for (int i = 0; i < n; ++i) {
        const double x = ud(rng);
        const double y = ud(rng);
        const double z = 0.15 * std::sin(2.5 * x) + 0.1 * std::cos(3.0 * y);
        cloud(i, 0) = x;
        cloud(i, 1) = y;
        cloud(i, 2) = z;
    }
    return cloud;
}

PointCloud transformCloud(const PointCloud& cloud,
                          const Eigen::Matrix3d& R,
                          const Eigen::Vector3d& t) {
    PointCloud out = cloud * R.transpose();
    out.rowwise() += t.transpose();
    return out;
}

struct CpuResult {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    int iterations = 0;
    double cov_ms = 0.0;
    double total_ms = 0.0;
};

CpuResult runCpuGicp(const PointCloud& source,
                     const PointCloud& target,
                     int max_iter,
                     int k_neighbors,
                     double epsilon) {
    CpuResult out;

    const auto cov_t0 = std::chrono::steady_clock::now();
    auto [sourceCovs, targetCovs] = gicp::computeCovariances(source, target, k_neighbors, epsilon);
    const auto cov_t1 = std::chrono::steady_clock::now();
    out.cov_ms = std::chrono::duration<double, std::milli>(cov_t1 - cov_t0).count();

    gicp::KDTree targetTree(3, std::cref(target));
    targetTree.index_->buildIndex();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    const auto t0 = std::chrono::steady_clock::now();
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        auto [H, g] = gicp::accumulateSystem(source, target, sourceCovs, targetCovs, &targetTree, R, t);
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

double rotationErrorDeg(const Eigen::Matrix3d& R_est, const Eigen::Matrix3d& R_ref) {
    const Eigen::Matrix3d dR = R_est * R_ref.transpose();
    const double tr = dR.trace();
    const double c = std::clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(c) * 180.0 / M_PI;
}

struct Args {
    int trials = 25;
    int points = 12000;
    int seed = 42;
    double max_angle_deg = 20.0;
    double fixed_angle_deg = -1.0;
    double max_translation = 0.08;
    double voxel_resolution = 0.08;
    double downsample_resolution = 0.0;
    double max_correspondence_distance = 1.0;
    double covariance_voxel_resolution = 0.08;
    std::string covariance_mode = "voxel";
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto readVal = [&](const std::string& key) -> std::string {
            auto eq = s.find('=');
            if (eq != std::string::npos) return s.substr(eq + 1);
            if (i + 1 < argc) return std::string(argv[++i]);
            throw std::runtime_error("Missing value for " + key);
        };

        if (s.rfind("--trials", 0) == 0) {
            a.trials = std::max(1, std::stoi(readVal("--trials")));
        } else if (s.rfind("--points", 0) == 0) {
            a.points = std::max(100, std::stoi(readVal("--points")));
        } else if (s.rfind("--seed", 0) == 0) {
            a.seed = std::stoi(readVal("--seed"));
        } else if (s.rfind("--max_angle_deg", 0) == 0) {
            a.max_angle_deg = std::max(0.0, std::stod(readVal("--max_angle_deg")));
        } else if (s.rfind("--fixed_angle_deg", 0) == 0) {
            a.fixed_angle_deg = std::stod(readVal("--fixed_angle_deg"));
        } else if (s.rfind("--max_translation", 0) == 0) {
            a.max_translation = std::max(0.0, std::stod(readVal("--max_translation")));
        } else if (s == "--help") {
            std::cout
                << "Usage: ./build/gicp_metal_demo [options]\n"
                << "  --trials <int>            Number of random trials (default: 25)\n"
                << "  --points <int>            Points per cloud (default: 12000)\n"
                << "  --seed <int>              RNG seed (default: 42)\n"
                << "  --max_angle_deg <double>  Max abs random rotation angle (default: 20)\n"
                << "  --fixed_angle_deg <double> If >=0, use this exact angle magnitude each trial (default: -1)\n"
                << "  --max_translation <double> Max abs random translation per axis (default: 0.08)\n"
                << "  --voxel_resolution <double> Voxel resolution for VGICP (default: 0.08)\n"
                << "  --downsample_resolution <double> Optional voxel downsample, <=0 disables (default: 0.0)\n"
                << "  --max_correspondence_distance <double> Distance gate in meters, <=0 disables (default: 1.0)\n"
                << "  --covariance_voxel_resolution <double> Voxel size for voxel covariance mode (default: 0.08)\n"
                << "  --covariance_mode <voxel|knn> Covariance estimation mode (default: voxel)\n";
            std::exit(0);
        } else if (s.rfind("--voxel_resolution", 0) == 0) {
            a.voxel_resolution = std::max(1e-6, std::stod(readVal("--voxel_resolution")));
        } else if (s.rfind("--downsample_resolution", 0) == 0) {
            a.downsample_resolution = std::stod(readVal("--downsample_resolution"));
        } else if (s.rfind("--max_correspondence_distance", 0) == 0) {
            a.max_correspondence_distance = std::stod(readVal("--max_correspondence_distance"));
        } else if (s.rfind("--covariance_voxel_resolution", 0) == 0) {
            a.covariance_voxel_resolution = std::max(1e-6, std::stod(readVal("--covariance_voxel_resolution")));
        } else if (s.rfind("--covariance_mode", 0) == 0) {
            a.covariance_mode = readVal("--covariance_mode");
        } else {
            throw std::runtime_error("Unknown argument: " + s);
        }
    }
    return a;
}

Eigen::Vector3d sampleUnitVec(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::Vector3d v;
    do {
        v = Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    } while (v.norm() < 1e-12);
    return v.normalized();
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    std::mt19937 rng(args.seed);

    gicp::metal::Config cfg;
    cfg.max_iter = 30;
    cfg.k_neighbors = 20;
    cfg.epsilon = 1e-3;
    cfg.downsample_resolution = args.downsample_resolution;
    cfg.voxel_resolution = args.voxel_resolution;
    cfg.neighbor_method = gicp::metal::NeighborSearchMethod::DIRECT27;
    cfg.max_correspondence_distance = args.max_correspondence_distance;
    cfg.covariance_voxel_resolution = args.covariance_voxel_resolution;
    cfg.covariance_mode = (args.covariance_mode == "knn")
                        ? gicp::metal::CovarianceMode::KNN
                        : gicp::metal::CovarianceMode::VOXEL_APPROX;
    cfg.allow_cpu_fallback = true;

    int success = 0;
    int used_gpu = 0;
    double mean_rot_err = 0.0;
    double mean_trans_err = 0.0;
    double mean_cpu_total = 0.0;
    double mean_gpu_total = 0.0;
    double mean_cpu_solve = 0.0;
    double mean_gpu_solve = 0.0;

    std::uniform_real_distribution<double> a_dist(-args.max_angle_deg, args.max_angle_deg);
    std::uniform_real_distribution<double> t_dist(-args.max_translation, args.max_translation);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    for (int trial = 0; trial < args.trials; ++trial) {
        const PointCloud target = makeCloud(args.points, rng);

        const Eigen::Vector3d axis = sampleUnitVec(rng);
        double angle_deg = a_dist(rng);
        if (args.fixed_angle_deg >= 0.0) {
            angle_deg = (sign_dist(rng) == 0 ? -1.0 : 1.0) * args.fixed_angle_deg;
        }
        const double angle_rad = angle_deg * M_PI / 180.0;
        const Eigen::AngleAxisd aa(angle_rad, axis);
        const Eigen::Matrix3d R_gt = aa.toRotationMatrix();
        const Eigen::Vector3d t_gt(t_dist(rng), t_dist(rng), t_dist(rng));
        const PointCloud source = transformCloud(target, R_gt, t_gt);

        const CpuResult cpu = runCpuGicp(source, target, cfg.max_iter, cfg.k_neighbors, cfg.epsilon);
        const gicp::metal::Result gpu = gicp::metal::registerPointClouds(source, target, cfg);

        const Eigen::Matrix3d R_expected = R_gt.transpose();
        const Eigen::Vector3d t_expected = -(R_gt.transpose() * t_gt);

        const double rot_err_deg = rotationErrorDeg(gpu.R, R_expected);
        const double trans_err = (gpu.t - t_expected).norm();

        const bool ok = rot_err_deg < 0.2 && trans_err < 1e-3;
        if (ok) ++success;
        if (gpu.used_gpu) ++used_gpu;

        mean_rot_err += rot_err_deg;
        mean_trans_err += trans_err;
        mean_cpu_solve += cpu.total_ms;
        mean_gpu_solve += gpu.total_ms;
        mean_cpu_total += (cpu.cov_ms + cpu.total_ms);
        mean_gpu_total += (gpu.cov_ms + gpu.total_ms);
    }

    const double inv_n = 1.0 / static_cast<double>(args.trials);
    mean_rot_err *= inv_n;
    mean_trans_err *= inv_n;
    mean_cpu_solve *= inv_n;
    mean_gpu_solve *= inv_n;
    mean_cpu_total *= inv_n;
    mean_gpu_total *= inv_n;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Trials=" << args.trials
              << " points=" << args.points
              << " seed=" << args.seed
              << " max_angle_deg=" << args.max_angle_deg
              << " fixed_angle_deg=" << args.fixed_angle_deg
              << " max_translation=" << args.max_translation
              << " voxel_resolution=" << args.voxel_resolution
              << " downsample_resolution=" << args.downsample_resolution
              << " cov_mode=" << args.covariance_mode
              << "\n";
    std::cout << "GPU backend active in " << used_gpu << "/" << args.trials << " trials\n";
    std::cout << "Convergence success (rot<0.2 deg && trans<1e-3): " << success << "/" << args.trials
              << " (" << (100.0 * static_cast<double>(success) * inv_n) << "%)\n";
    std::cout << "Mean pose error: rot_deg=" << mean_rot_err
              << " trans_norm=" << mean_trans_err << "\n";
    std::cout << "Mean speedup solve-only: x" << (mean_cpu_solve / std::max(mean_gpu_solve, 1e-9))
              << " | end-to-end: x" << (mean_cpu_total / std::max(mean_gpu_total, 1e-9))
              << "\n";

    return 0;
}
