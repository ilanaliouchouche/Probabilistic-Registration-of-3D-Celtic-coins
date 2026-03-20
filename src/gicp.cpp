#include "gicp.h"

#include <Eigen/Dense>
#include <nanoflann.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace {

std::unique_ptr<gicp::KDTree> buildIndex(const gicp::PointCloud& cloud)
{
    auto tree = std::make_unique<gicp::KDTree>(3, std::cref(cloud));
    tree->index_->buildIndex();
    return tree;
}

Eigen::Matrix3d computePointCovariance(
    const gicp::PointCloud& cloud,
    gicp::KDTree* tree,
    int queryIndex,
    int k,
    double epsilon)
{
    int kk = k;
    const int point_count = static_cast<int>(cloud.rows());
    if (kk > point_count) kk = point_count;
    if (kk <= 0)
    {
        return Eigen::Matrix3d::Identity() * epsilon;
    }

    std::vector<size_t> indices(kk);
    std::vector<double> dists(kk);

    nanoflann::KNNResultSet<double> resultSet(kk);
    resultSet.init(indices.data(), dists.data());

    const double query_pt[3] = {
        cloud(queryIndex,0),
        cloud(queryIndex,1),
        cloud(queryIndex,2)
    };

    tree->index_->findNeighbors(
        resultSet,
        query_pt,
        nanoflann::SearchParameters());

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();

    for(int i=0;i<kk;i++)
        mean += cloud.row(indices[i]).transpose();

    mean /= double(kk);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    for(int i=0;i<kk;i++){
        Eigen::Vector3d diff =
            cloud.row(indices[i]).transpose() - mean;
        cov += diff * diff.transpose();
    }

    cov /= double(kk);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);

    Eigen::Matrix3d R = es.eigenvectors();

    Eigen::Vector3d evals;
    evals << epsilon, 1.0, 1.0;

    return R * evals.asDiagonal() * R.transpose();
}

Eigen::Matrix3d computePointWeightMatrix(
    const Eigen::Matrix3d& sourceCov,
    const Eigen::Matrix3d& targetCov,
    const Eigen::Matrix3d& R)
{
    return (targetCov + R * sourceCov * R.transpose()).inverse();
}

Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<
        0,      -v(2),  v(1),
        v(2),    0,    -v(0),
       -v(1),   v(0),   0;
    return S;
}

Eigen::Matrix<double,3,6> computePointJacobian(
    const Eigen::Vector3d& sourcePoint,
    const Eigen::Matrix3d& R)
{
    Eigen::Matrix<double,3,6> J;

    Eigen::Vector3d Rp = R * sourcePoint;

    J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    J.block<3,3>(0,3) = -skew(Rp);

    return J;
}

}  // namespace

namespace gicp {

std::pair<CovarianceList,CovarianceList>
computeCovariances(
    const PointCloud& source,
    const PointCloud& target,
    int kNeighbors,
    double epsilon)
{
    auto sourceTree = buildIndex(source);
    auto targetTree = buildIndex(target);

    CovarianceList sourceCovs(source.rows());
    CovarianceList targetCovs(target.rows());

    for(int i=0;i<source.rows();i++){
        sourceCovs[i] =
            computePointCovariance(
                source,
                sourceTree.get(),
                i,
                kNeighbors,
                epsilon);
    }

    for(int i=0;i<target.rows();i++){
        targetCovs[i] =
            computePointCovariance(
                target,
                targetTree.get(),
                i,
                kNeighbors,
                epsilon);
    }

    return {sourceCovs,targetCovs};
}

std::pair<Eigen::Matrix<double,6,6>, Eigen::Matrix<double,6,1>>
accumulateSystem(
    const PointCloud& source,
    const PointCloud& target,
    const CovarianceList& sourceCovs,
    const CovarianceList& targetCovs,
    KDTree* targetTree,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t)
{
    Eigen::Matrix<double,6,6> H =
        Eigen::Matrix<double,6,6>::Zero();

    Eigen::Matrix<double,6,1> g =
        Eigen::Matrix<double,6,1>::Zero();

    const int N = source.rows();

    size_t nn_index;
    double nn_dist;

    nanoflann::KNNResultSet<double> resultSet(1);

    for(int i=0;i<N;i++)
    {
        Eigen::Vector3d p =
            source.row(i).transpose();

        Eigen::Vector3d Rp =
            R * p + t;

        const double query_pt[3] =
        {
            Rp(0),
            Rp(1),
            Rp(2)
        };

        resultSet.init(&nn_index,&nn_dist);

        targetTree->index_->findNeighbors(
            resultSet,
            query_pt,
            nanoflann::SearchParameters());

        Eigen::Vector3d q =
            target.row(nn_index).transpose();

        Eigen::Vector3d error =
            q - Rp;

        Eigen::Matrix3d W =
            computePointWeightMatrix(
                sourceCovs[i],
                targetCovs[nn_index],
                R);

        Eigen::Matrix<double,3,6> J =
            computePointJacobian(
                p,
                R);

        H.noalias() +=
            J.transpose() * W * J;

        g.noalias() +=
            J.transpose() * W * error;
    }

    return {H,g};
}


Eigen::Matrix<double,6,1>
solveDeltaXiDense(
    const Eigen::Matrix<double,6,6>& H,
    const Eigen::Matrix<double,6,1>& g)
{
    Eigen::LDLT<Eigen::Matrix<double,6,6>> ldlt(H);
    if(ldlt.info() != Eigen::Success)
    {
        std::cerr << "LDLT decomposition failed\n";
        return Eigen::Matrix<double,6,1>::Zero();
    }

    Eigen::Matrix<double,6,1> delta_xi = ldlt.solve(g);
    if(ldlt.info() != Eigen::Success)
        std::cerr << "LDLT solve failed\n";

    return delta_xi;
}

void updateTransform(
    Eigen::Matrix3d& R,
    Eigen::Vector3d& t,
    const Eigen::Matrix<double,6,1>& delta_xi)
{
    Eigen::Vector3d dt =
        delta_xi.head<3>();

    Eigen::Vector3d omega =
        delta_xi.tail<3>();

    double theta = omega.norm();
    Eigen::Matrix3d Omega =
        skew(omega);

    Eigen::Matrix3d dR =
        Eigen::Matrix3d::Identity();

    if(theta > 1e-12)
    {
        dR =
            Eigen::Matrix3d::Identity()
            + (std::sin(theta)/theta) * Omega
            + ((1.0 - std::cos(theta)) / (theta*theta)) * Omega * Omega;
    }
    else
    {
        dR =
            Eigen::Matrix3d::Identity()
            + Omega;
    }

    t = dR * t + dt;
    R = dR * R;
}

}
