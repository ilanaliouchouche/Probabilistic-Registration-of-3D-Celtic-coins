#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gicp/gicp_metal.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gicp.h"

#ifndef GICP_METAL_SHADER_PATH
#define GICP_METAL_SHADER_PATH "metal/gicp_vgicp.metal"
#endif

namespace gicp {
namespace metal {
namespace {

struct PackedPoint {
    float x;
    float y;
    float z;
    float w;
};

struct PackedOffset {
    int x;
    int y;
    int z;
    int w;
};

struct PackedVoxelBucket {
    int x;
    int y;
    int z;
    int index;
};

struct PackedParams {
    uint32_t source_count;
    uint32_t voxel_count;
    uint32_t bucket_count;
    uint32_t offset_count;

    float voxel_resolution;
    uint32_t bucket_mask;
    float max_corr_dist_sq;
    uint32_t _pad0;

    float r00, r01, r02, tx;
    float r10, r11, r12, ty;
    float r20, r21, r22, tz;
};

struct VoxelKey {
    int x;
    int y;
    int z;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.x)) * 73856093ull;
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.y)) * 19349663ull;
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.z)) * 83492791ull;
        return static_cast<std::size_t>(h);
    }
};

struct VoxelAccumCov {
    Eigen::Vector3d sum_mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_cov = Eigen::Matrix3d::Zero();
    std::uint32_t count = 0;
};

struct VoxelAccumMoments {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();
    std::uint32_t count = 0;
};

struct VoxelizedTarget {
    std::vector<PackedVoxelBucket> buckets;
    std::vector<PackedPoint> means;
    std::vector<float> covs;
    std::vector<std::uint32_t> counts;
    std::uint32_t bucket_mask = 0;
};

inline std::uint32_t nextPow2(std::uint32_t v) {
    if (v <= 1u) return 1u;
    --v;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

inline VoxelKey voxelCoord(const Eigen::Vector3d& p, double res) {
    return VoxelKey{
        static_cast<int>(std::floor(p.x() / res)),
        static_cast<int>(std::floor(p.y() / res)),
        static_cast<int>(std::floor(p.z() / res)),
    };
}

inline std::uint32_t hashCoord(const VoxelKey& c) {
    std::uint32_t x = static_cast<std::uint32_t>(c.x);
    std::uint32_t y = static_cast<std::uint32_t>(c.y);
    std::uint32_t z = static_cast<std::uint32_t>(c.z);
    return x * 73856093u ^ y * 19349663u ^ z * 83492791u;
}

std::vector<PackedOffset> buildOffsets(NeighborSearchMethod method, double radius) {
    std::vector<PackedOffset> out;
    switch (method) {
        case NeighborSearchMethod::DIRECT1:
            out.push_back({0, 0, 0, 0});
            break;
        case NeighborSearchMethod::DIRECT7:
            out = {
                {0, 0, 0, 0}, {1, 0, 0, 0}, {-1, 0, 0, 0},
                {0, 1, 0, 0}, {0, -1, 0, 0}, {0, 0, 1, 0}, {0, 0, -1, 0},
            };
            break;
        case NeighborSearchMethod::DIRECT27:
            out.reserve(27);
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    for (int z = -1; z <= 1; ++z) {
                        out.push_back({x, y, z, 0});
                    }
                }
            }
            break;
        case NeighborSearchMethod::DIRECT_RADIUS: {
            int r = static_cast<int>(std::ceil(radius));
            for (int x = -r; x <= r; ++x) {
                for (int y = -r; y <= r; ++y) {
                    for (int z = -r; z <= r; ++z) {
                        const double d = std::sqrt(static_cast<double>(x * x + y * y + z * z));
                        if (d <= radius + 1e-9) {
                            out.push_back({x, y, z, 0});
                        }
                    }
                }
            }
            break;
        }
    }
    return out;
}

Eigen::Matrix3d regularizeCovariance(const Eigen::Matrix3d& cov_raw, double epsilon) {
    Eigen::Matrix3d cov = cov_raw;
    cov = 0.5 * (cov + cov.transpose());
    cov += 1e-9 * Eigen::Matrix3d::Identity();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    if (es.info() != Eigen::Success) {
        return Eigen::Matrix3d::Identity() * epsilon;
    }

    Eigen::Vector3d evals;
    evals << epsilon, 1.0, 1.0;
    return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}

PointCloud voxelDownsampleCentroid(const PointCloud& cloud, double resolution) {
    if (resolution <= 0.0 || cloud.rows() == 0) {
        return cloud;
    }

    std::unordered_map<VoxelKey, std::pair<Eigen::Vector3d, std::uint32_t>, VoxelKeyHash> vox;
    vox.reserve(static_cast<std::size_t>(cloud.rows()));

    for (Eigen::Index i = 0; i < cloud.rows(); ++i) {
        const Eigen::Vector3d p = cloud.row(i).transpose();
        const VoxelKey key = voxelCoord(p, resolution);
        auto& acc = vox[key];
        acc.first += p;
        acc.second += 1u;
    }

    PointCloud down(static_cast<Eigen::Index>(vox.size()), 3);
    Eigen::Index idx = 0;
    for (const auto& kv : vox) {
        const Eigen::Vector3d mean = kv.second.first / static_cast<double>(kv.second.second);
        down.row(idx++) = mean.transpose();
    }
    return down;
}

std::unordered_map<VoxelKey, VoxelAccumMoments, VoxelKeyHash>
buildVoxelMoments(const PointCloud& cloud, double voxel_resolution) {
    std::unordered_map<VoxelKey, VoxelAccumMoments, VoxelKeyHash> vox;
    vox.reserve(static_cast<std::size_t>(cloud.rows()));

    for (Eigen::Index i = 0; i < cloud.rows(); ++i) {
        const Eigen::Vector3d p = cloud.row(i).transpose();
        const VoxelKey key = voxelCoord(p, voxel_resolution);
        auto& v = vox[key];
        v.sum += p;
        v.sum_outer += p * p.transpose();
        v.count += 1u;
    }

    return vox;
}

CovarianceList buildVoxelApproxPointCovariances(const PointCloud& cloud,
                                                double voxel_resolution,
                                                double epsilon) {
    CovarianceList covs(static_cast<std::size_t>(cloud.rows()), Eigen::Matrix3d::Identity() * epsilon);
    if (cloud.rows() == 0) {
        return covs;
    }

    auto moments = buildVoxelMoments(cloud, voxel_resolution);
    std::unordered_map<VoxelKey, Eigen::Matrix3d, VoxelKeyHash> reg_covs;
    reg_covs.reserve(moments.size());

    for (const auto& kv : moments) {
        const auto& m = kv.second;
        if (m.count < 3u) {
            reg_covs.emplace(kv.first, Eigen::Matrix3d::Identity() * epsilon);
            continue;
        }
        const double inv = 1.0 / static_cast<double>(m.count);
        const Eigen::Vector3d mean = m.sum * inv;
        Eigen::Matrix3d cov = m.sum_outer * inv - mean * mean.transpose();
        reg_covs.emplace(kv.first, regularizeCovariance(cov, epsilon));
    }

    for (Eigen::Index i = 0; i < cloud.rows(); ++i) {
        const VoxelKey key = voxelCoord(cloud.row(i).transpose(), voxel_resolution);
        auto it = reg_covs.find(key);
        if (it != reg_covs.end()) {
            covs[static_cast<std::size_t>(i)] = it->second;
        }
    }

    return covs;
}

VoxelizedTarget buildVoxelizedTargetFromPointCovs(const PointCloud& target,
                                                  const CovarianceList& targetCovs,
                                                  double voxel_resolution) {
    if (target.rows() <= 0) {
        return VoxelizedTarget{};
    }

    std::unordered_map<VoxelKey, VoxelAccumCov, VoxelKeyHash> vox;
    vox.reserve(static_cast<std::size_t>(target.rows()));

    for (Eigen::Index i = 0; i < target.rows(); ++i) {
        Eigen::Vector3d p = target.row(i).transpose();
        VoxelKey key = voxelCoord(p, voxel_resolution);
        auto& acc = vox[key];
        acc.sum_mean += p;
        acc.sum_cov += targetCovs[static_cast<std::size_t>(i)];
        acc.count += 1u;
    }

    VoxelizedTarget out;
    out.means.reserve(vox.size());
    out.covs.reserve(vox.size() * 9);
    out.counts.reserve(vox.size());

    std::vector<std::pair<VoxelKey, std::uint32_t>> key_to_idx;
    key_to_idx.reserve(vox.size());

    std::uint32_t idx = 0;
    for (const auto& kv : vox) {
        const VoxelAccumCov& v = kv.second;
        const double inv = 1.0 / static_cast<double>(v.count);

        Eigen::Vector3d mean = v.sum_mean * inv;
        Eigen::Matrix3d cov = v.sum_cov * inv;

        out.means.push_back(PackedPoint{
            static_cast<float>(mean.x()),
            static_cast<float>(mean.y()),
            static_cast<float>(mean.z()),
            0.0f,
        });

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out.covs.push_back(static_cast<float>(cov(r, c)));
            }
        }

        out.counts.push_back(v.count);
        key_to_idx.emplace_back(kv.first, idx);
        ++idx;
    }

    std::uint32_t bucket_count = nextPow2(std::max<std::uint32_t>(1024u, static_cast<std::uint32_t>(vox.size() * 2u)));
    out.bucket_mask = bucket_count - 1u;

    out.buckets.assign(bucket_count, PackedVoxelBucket{
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        -1,
    });

    for (const auto& kv : key_to_idx) {
        const VoxelKey key = kv.first;
        const std::uint32_t vi = kv.second;
        std::uint32_t h = hashCoord(key);

        bool placed = false;
        for (std::uint32_t scan = 0; scan < bucket_count; ++scan) {
            const std::uint32_t slot = (h + scan) & out.bucket_mask;
            if (out.buckets[slot].index < 0) {
                out.buckets[slot] = PackedVoxelBucket{key.x, key.y, key.z, static_cast<int>(vi)};
                placed = true;
                break;
            }
        }

        if (!placed) {
            throw std::runtime_error("Voxel hash table overflow. Increase bucket capacity.");
        }
    }

    return out;
}

std::vector<PackedPoint> packPointCloud(const PointCloud& cloud) {
    std::vector<PackedPoint> out;
    out.reserve(static_cast<std::size_t>(cloud.rows()));
    for (Eigen::Index i = 0; i < cloud.rows(); ++i) {
        out.push_back(PackedPoint{
            static_cast<float>(cloud(i, 0)),
            static_cast<float>(cloud(i, 1)),
            static_cast<float>(cloud(i, 2)),
            0.0f,
        });
    }
    return out;
}

std::vector<float> packCovariances(const CovarianceList& covs) {
    std::vector<float> out;
    out.reserve(covs.size() * 9);
    for (const auto& c : covs) {
        for (int r = 0; r < 3; ++r) {
            for (int col = 0; col < 3; ++col) {
                out.push_back(static_cast<float>(c(r, col)));
            }
        }
    }
    return out;
}

void fillSymmetricH(const std::array<double, 21>& h_upper,
                    Eigen::Matrix<double, 6, 6>& H) {
    H.setZero();
    int idx = 0;
    for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) {
            H(r, c) = h_upper[static_cast<std::size_t>(idx++)];
        }
    }
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < r; ++c) {
            H(r, c) = H(c, r);
        }
    }
}

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;
    bool ready = false;
    std::string error;
};

std::string nsErrorToString(NSError* err) {
    if (!err) {
        return "unknown";
    }
    const char* msg = [[err localizedDescription] UTF8String];
    return msg ? std::string(msg) : std::string("unknown");
}

const MetalContext& getMetalContext() {
    static MetalContext ctx;
    static std::once_flag once;

    std::call_once(once, [&]() {
        ctx.device = MTLCreateSystemDefaultDevice();
        if (!ctx.device) {
            ctx.error = "no Metal device available";
            return;
        }

        NSError* err = nil;
        NSString* shaderPath = [NSString stringWithUTF8String:GICP_METAL_SHADER_PATH];
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&err];
        if (!shaderSource) {
            ctx.error = "failed to read shader file: " + std::string(GICP_METAL_SHADER_PATH) + " error=" + nsErrorToString(err);
            return;
        }

        id<MTLLibrary> library = [ctx.device newLibraryWithSource:shaderSource options:nil error:&err];
        if (!library) {
            ctx.error = "shader compile failed: " + nsErrorToString(err);
            return;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"vgicp_accumulate"];
        if (!fn) {
            ctx.error = "missing kernel function vgicp_accumulate";
            return;
        }

        ctx.pso = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!ctx.pso) {
            ctx.error = "pipeline creation failed: " + nsErrorToString(err);
            return;
        }

        ctx.queue = [ctx.device newCommandQueue];
        if (!ctx.queue) {
            ctx.error = "command queue creation failed";
            return;
        }

        ctx.ready = true;
    });

    return ctx;
}

bool runGpuIteration(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pso,
                     id<MTLBuffer> sourcePointsBuf,
                     id<MTLBuffer> sourceCovsBuf,
                     id<MTLBuffer> bucketBuf,
                     id<MTLBuffer> voxelMeansBuf,
                     id<MTLBuffer> voxelCovsBuf,
                     id<MTLBuffer> voxelCountsBuf,
                     id<MTLBuffer> offsetsBuf,
                     id<MTLBuffer> paramsBuf,
                     id<MTLBuffer> partialBuf,
                     std::size_t source_count,
                     std::size_t partial_count,
                     Eigen::Matrix<double, 6, 6>& H,
                     Eigen::Matrix<double, 6, 1>& g,
                     double& objective,
                     double& matches) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        return false;
    }

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        return false;
    }

    [enc setComputePipelineState:pso];
    [enc setBuffer:sourcePointsBuf offset:0 atIndex:0];
    [enc setBuffer:sourceCovsBuf offset:0 atIndex:1];
    [enc setBuffer:bucketBuf offset:0 atIndex:2];
    [enc setBuffer:voxelMeansBuf offset:0 atIndex:3];
    [enc setBuffer:voxelCovsBuf offset:0 atIndex:4];
    [enc setBuffer:voxelCountsBuf offset:0 atIndex:5];
    [enc setBuffer:offsetsBuf offset:0 atIndex:6];
    [enc setBuffer:paramsBuf offset:0 atIndex:7];
    [enc setBuffer:partialBuf offset:0 atIndex:8];

    const NSUInteger tg_size = 256;
    const NSUInteger groups = (source_count + tg_size - 1) / tg_size;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status != MTLCommandBufferStatusCompleted) {
        return false;
    }

    const float* partial = static_cast<const float*>([partialBuf contents]);
    if (!partial) {
        return false;
    }

    std::array<double, 21> h_sum{};
    std::array<double, 6> g_sum{};
    double err_sum = 0.0;
    double cnt_sum = 0.0;

    for (std::size_t i = 0; i < partial_count; ++i) {
        const std::size_t base = i * 29;
        for (int k = 0; k < 21; ++k) h_sum[static_cast<std::size_t>(k)] += partial[base + static_cast<std::size_t>(k)];
        for (int k = 0; k < 6; ++k) g_sum[static_cast<std::size_t>(k)] += partial[base + 21 + static_cast<std::size_t>(k)];
        err_sum += partial[base + 27];
        cnt_sum += partial[base + 28];
    }

    fillSymmetricH(h_sum, H);
    for (int i = 0; i < 6; ++i) g(i) = g_sum[static_cast<std::size_t>(i)];
    objective = err_sum;
    matches = cnt_sum;

    return true;
}

Result runCpuWithPrecomputedCovariances(const PointCloud& source,
                                        const PointCloud& target,
                                        const CovarianceList& sourceCovs,
                                        const CovarianceList& targetCovs,
                                        int max_iter,
                                        double cov_ms_override) {
    Result out;
    out.backend = "cpu_fallback";
    out.used_gpu = false;

    gicp::KDTree targetTree(3, std::cref(target));
    targetTree.index_->buildIndex();

    out.cov_ms = cov_ms_override;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    const auto t0 = std::chrono::steady_clock::now();
    int iter = 0;
    for (; iter < max_iter; ++iter) {
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

}  // namespace

Result registerPointClouds(const PointCloud& source,
                           const PointCloud& target,
                           const Config& cfg) {
    Result out;
    out.backend = "metal";
    out.used_gpu = false;

    if (source.rows() == 0 || target.rows() == 0) {
        return out;
    }

    PointCloud work_source = source;
    PointCloud work_target = target;
    if (cfg.downsample_resolution > 0.0) {
        work_source = voxelDownsampleCentroid(source, cfg.downsample_resolution);
        work_target = voxelDownsampleCentroid(target, cfg.downsample_resolution);
    }

    if (work_source.rows() == 0 || work_target.rows() == 0) {
        return out;
    }

    CovarianceList sourceCovs;
    CovarianceList targetCovs;
    const auto cov_t0 = std::chrono::steady_clock::now();
    if (cfg.covariance_mode == CovarianceMode::KNN) {
        auto pair = gicp::computeCovariances(
            work_source,
            work_target,
            cfg.k_neighbors,
            cfg.epsilon);
        sourceCovs = std::move(pair.first);
        targetCovs = std::move(pair.second);
    } else {
        const double cov_voxel = (cfg.covariance_voxel_resolution > 0.0)
                               ? cfg.covariance_voxel_resolution
                               : cfg.voxel_resolution;
        sourceCovs = buildVoxelApproxPointCovariances(work_source, cov_voxel, cfg.epsilon);
        targetCovs = buildVoxelApproxPointCovariances(work_target, cov_voxel, cfg.epsilon);
    }
    const auto cov_t1 = std::chrono::steady_clock::now();
    out.cov_ms = std::chrono::duration<double, std::milli>(cov_t1 - cov_t0).count();

    try {
        const MetalContext& metal = getMetalContext();
        if (!metal.ready) {
            std::cerr << "[gicp_metal] " << metal.error << ", fallback to CPU\n";
            if (cfg.allow_cpu_fallback) {
                return runCpuWithPrecomputedCovariances(
                    work_source,
                    work_target,
                    sourceCovs,
                    targetCovs,
                    cfg.max_iter,
                    out.cov_ms);
            }
            return out;
        }

        const auto sourcePacked = packPointCloud(work_source);
        const auto sourceCovPacked = packCovariances(sourceCovs);
        const auto voxelTarget = buildVoxelizedTargetFromPointCovs(work_target, targetCovs, cfg.voxel_resolution);
        const auto offsets = buildOffsets(cfg.neighbor_method, cfg.neighbor_radius);

        if (voxelTarget.means.empty() || offsets.empty()) {
            if (cfg.allow_cpu_fallback) {
                return runCpuWithPrecomputedCovariances(
                    work_source,
                    work_target,
                    sourceCovs,
                    targetCovs,
                    cfg.max_iter,
                    out.cov_ms);
            }
            return out;
        }

        const NSUInteger opt = MTLResourceStorageModeShared;

        id<MTLBuffer> sourcePointsBuf = [metal.device newBufferWithBytes:sourcePacked.data()
                                                            length:sourcePacked.size() * sizeof(PackedPoint)
                                                           options:opt];
        id<MTLBuffer> sourceCovsBuf = [metal.device newBufferWithBytes:sourceCovPacked.data()
                                                          length:sourceCovPacked.size() * sizeof(float)
                                                         options:opt];
        id<MTLBuffer> bucketBuf = [metal.device newBufferWithBytes:voxelTarget.buckets.data()
                                                      length:voxelTarget.buckets.size() * sizeof(PackedVoxelBucket)
                                                     options:opt];
        id<MTLBuffer> voxelMeansBuf = [metal.device newBufferWithBytes:voxelTarget.means.data()
                                                          length:voxelTarget.means.size() * sizeof(PackedPoint)
                                                         options:opt];
        id<MTLBuffer> voxelCovsBuf = [metal.device newBufferWithBytes:voxelTarget.covs.data()
                                                         length:voxelTarget.covs.size() * sizeof(float)
                                                        options:opt];
        id<MTLBuffer> voxelCountsBuf = [metal.device newBufferWithBytes:voxelTarget.counts.data()
                                                           length:voxelTarget.counts.size() * sizeof(std::uint32_t)
                                                          options:opt];
        id<MTLBuffer> offsetsBuf = [metal.device newBufferWithBytes:offsets.data()
                                                       length:offsets.size() * sizeof(PackedOffset)
                                                      options:opt];

        const std::size_t sourceCount = sourcePacked.size();
        const std::size_t threadgroupSize = 256;
        const std::size_t partialCount = (sourceCount + threadgroupSize - 1) / threadgroupSize;

        id<MTLBuffer> paramsBuf = [metal.device newBufferWithLength:sizeof(PackedParams) options:opt];
        id<MTLBuffer> partialBuf = [metal.device newBufferWithLength:partialCount * 29 * sizeof(float) options:opt];

        if (!sourcePointsBuf || !sourceCovsBuf || !bucketBuf || !voxelMeansBuf || !voxelCovsBuf || !voxelCountsBuf || !offsetsBuf || !paramsBuf || !partialBuf) {
            std::cerr << "[gicp_metal] buffer allocation failed\n";
            if (cfg.allow_cpu_fallback) {
                return runCpuWithPrecomputedCovariances(
                    work_source,
                    work_target,
                    sourceCovs,
                    targetCovs,
                    cfg.max_iter,
                    out.cov_ms);
            }
            return out;
        }

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();

        const auto t0 = std::chrono::steady_clock::now();

        int iter = 0;
        for (; iter < cfg.max_iter; ++iter) {
            PackedParams p{};
            p.source_count = static_cast<std::uint32_t>(sourceCount);
            p.voxel_count = static_cast<std::uint32_t>(voxelTarget.means.size());
            p.bucket_count = static_cast<std::uint32_t>(voxelTarget.buckets.size());
            p.offset_count = static_cast<std::uint32_t>(offsets.size());
            p.voxel_resolution = static_cast<float>(cfg.voxel_resolution);
            p.bucket_mask = voxelTarget.bucket_mask;
            p.max_corr_dist_sq = (cfg.max_correspondence_distance > 0.0)
                               ? static_cast<float>(cfg.max_correspondence_distance * cfg.max_correspondence_distance)
                               : -1.0f;
            p._pad0 = 0u;

            p.r00 = static_cast<float>(R(0, 0)); p.r01 = static_cast<float>(R(0, 1)); p.r02 = static_cast<float>(R(0, 2)); p.tx = static_cast<float>(t(0));
            p.r10 = static_cast<float>(R(1, 0)); p.r11 = static_cast<float>(R(1, 1)); p.r12 = static_cast<float>(R(1, 2)); p.ty = static_cast<float>(t(1));
            p.r20 = static_cast<float>(R(2, 0)); p.r21 = static_cast<float>(R(2, 1)); p.r22 = static_cast<float>(R(2, 2)); p.tz = static_cast<float>(t(2));

            std::memcpy([paramsBuf contents], &p, sizeof(PackedParams));
            std::memset([partialBuf contents], 0, partialCount * 29 * sizeof(float));

            Eigen::Matrix<double, 6, 6> H;
            Eigen::Matrix<double, 6, 1> g;
            double objective = 0.0;
            double matches = 0.0;

            const bool ok = runGpuIteration(
                metal.queue,
                metal.pso,
                sourcePointsBuf,
                sourceCovsBuf,
                bucketBuf,
                voxelMeansBuf,
                voxelCovsBuf,
                voxelCountsBuf,
                offsetsBuf,
                paramsBuf,
                partialBuf,
                sourceCount,
                partialCount,
                H,
                g,
                objective,
                matches);

            if (!ok || matches <= 0.0) {
                std::cerr << "[gicp_metal] gpu iteration failed (ok=" << (ok ? "true" : "false")
                          << ", matches=" << matches << "), fallback to CPU\n";
                if (cfg.allow_cpu_fallback) {
                    return runCpuWithPrecomputedCovariances(
                        work_source,
                        work_target,
                        sourceCovs,
                        targetCovs,
                        cfg.max_iter,
                        out.cov_ms);
                }
                break;
            }

            Eigen::Matrix<double, 6, 1> delta = gicp::solveDeltaXiDense(H, g);
            gicp::updateTransform(R, t, delta);

            out.objective = objective;
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
        out.used_gpu = true;
        out.backend = "metal_vgicp";

        return out;
    } catch (const std::exception&) {
        std::cerr << "[gicp_metal] exception during GPU path, fallback to CPU\n";
        if (cfg.allow_cpu_fallback) {
            return runCpuWithPrecomputedCovariances(
                work_source,
                work_target,
                sourceCovs,
                targetCovs,
                cfg.max_iter,
                out.cov_ms);
        }
        return out;
    }
}

}  // namespace metal
}  // namespace gicp

#else

#include "gicp_metal_stub.cpp"

#endif
