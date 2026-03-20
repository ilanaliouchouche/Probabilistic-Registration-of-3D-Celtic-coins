#include "icp.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <nanoflann.hpp>

#include <iostream>
#include <memory>
#include <vector>

namespace {

std::unique_ptr<icp::KDTree> buildIndex(const icp::PointCloud& cloud)
{
    auto tree = std::make_unique<icp::KDTree>(3, std::cref(cloud));
    tree->index_->buildIndex();
    return tree;
}

}  // namespace

namespace icp {

AccumulatedSystem accumulateSystem(
    const PointCloud& source,
    const PointCloud& target,
    KDTree* targetTree,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t)
{
    AccumulatedSystem sys;
    const int N = static_cast<int>(source.rows());
    if (N <= 0)
    {
        return sys;
    }

    Eigen::Vector3d sum_source = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_target = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_source_target = Eigen::Matrix3d::Zero();

    size_t nn_index = 0;
    double nn_dist = 0.0;
    nanoflann::KNNResultSet<double> resultSet(1);

    for (int i = 0; i < N; ++i)
    {
        const Eigen::Vector3d p = source.row(i).transpose();
        const Eigen::Vector3d Rp = R * p + t;

        const double query_pt[3] = {Rp(0), Rp(1), Rp(2)};
        resultSet.init(&nn_index, &nn_dist);
        targetTree->index_->findNeighbors(
            resultSet,
            query_pt,
            nanoflann::SearchParameters());

        const Eigen::Vector3d q = target.row(nn_index).transpose();

        sum_source += Rp;
        sum_target += q;
        sum_source_target.noalias() += Rp * q.transpose();
    }

    sys.count = N;
    sys.source_mean = sum_source / static_cast<double>(N);
    sys.target_mean = sum_target / static_cast<double>(N);
    sys.H = sum_source_target -
            static_cast<double>(N) * sys.source_mean * sys.target_mean.transpose();

    return sys;
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d>
solveTransformSVD(
    const Eigen::Matrix3d& H,
    const Eigen::Vector3d& source_mean,
    const Eigen::Vector3d& target_mean)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((V * U.transpose()).determinant() < 0.0)
    {
        D(2, 2) = -1.0;
    }

    const Eigen::Matrix3d dR = V * D * U.transpose();
    const Eigen::Vector3d dt = target_mean - dR * source_mean;

    return {dR, dt};
}

void updateTransform(
    Eigen::Matrix3d& R,
    Eigen::Vector3d& t,
    const Eigen::Matrix3d& dR,
    const Eigen::Vector3d& dt)
{
    t = dR * t + dt;
    R = dR * R;
}

}  // namespace icp
