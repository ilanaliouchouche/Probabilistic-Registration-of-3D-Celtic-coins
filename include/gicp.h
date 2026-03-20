#pragma once

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <utility>
#include <vector>

namespace gicp {

using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using CovarianceList = std::vector<Eigen::Matrix3d>;
using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<PointCloud, 3, nanoflann::metric_L2>;

std::pair<CovarianceList, CovarianceList>
computeCovariances(const PointCloud& source,
                   const PointCloud& target,
                   int kNeighbors = 20,
                   double epsilon = 1e-3);

std::pair<Eigen::Matrix<double,6,6>, Eigen::Matrix<double,6,1>>
accumulateSystem(const PointCloud& source,
                 const PointCloud& target,
                 const CovarianceList& sourceCovs,
                 const CovarianceList& targetCovs,
                 KDTree* targetTree,
                 const Eigen::Matrix3d& R,
                 const Eigen::Vector3d& t);

Eigen::Matrix<double,6,1>
solveDeltaXiDense(const Eigen::Matrix<double,6,6>& H,
                  const Eigen::Matrix<double,6,1>& g);

void updateTransform(Eigen::Matrix3d& R,
                     Eigen::Vector3d& t,
                     const Eigen::Matrix<double,6,1>& delta_xi);

}
