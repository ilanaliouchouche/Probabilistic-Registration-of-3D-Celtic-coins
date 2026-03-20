#pragma once

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <utility>
#include <vector>

namespace icp {

using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<PointCloud, 3, nanoflann::metric_L2>;

struct AccumulatedSystem
{
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d source_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_mean = Eigen::Vector3d::Zero();
    int count = 0;
};

AccumulatedSystem accumulateSystem(const PointCloud& source,
                                   const PointCloud& target,
                                   KDTree* targetTree,
                                   const Eigen::Matrix3d& R,
                                   const Eigen::Vector3d& t);

std::pair<Eigen::Matrix3d, Eigen::Vector3d>
solveTransformSVD(const Eigen::Matrix3d& H,
                  const Eigen::Vector3d& source_mean,
                  const Eigen::Vector3d& target_mean);

void updateTransform(Eigen::Matrix3d& R,
                     Eigen::Vector3d& t,
                     const Eigen::Matrix3d& dR,
                     const Eigen::Vector3d& dt);

}
