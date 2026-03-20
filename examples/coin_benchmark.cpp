#include "gicp.h"
#include "icp.h"

#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<PointCloud, 3, nanoflann::metric_L2>;

struct Config
{
    std::string dataset_root = "Riedones3D_Registration_Benchmark/test";
    int max_iter = 20;
    int k_neighbors = 20;
    double epsilon = 1e-3;
    double sample_ratio = 1.0;
    std::string log_dir = "logs";
};

struct PairJob
{
    std::string die;
    std::string label;
    std::string pair_id;
    std::string source;
    std::string target;
    double overlap = 0.0;
    Eigen::Matrix3d R_gt = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_gt = Eigen::Vector3d::Zero();
    std::filesystem::path txt_path;
};

struct MethodResult
{
    std::string method;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    int iterations = 0;
    double total_ms = 0.0;
    double cov_ms = 0.0;
    double sre = 0.0;
    double sse = 0.0;
};

struct ProgressBar
{
    explicit ProgressBar(std::size_t total, std::size_t width = 30)
        : total_(total), width_(width) {}

    void update(std::size_t current)
    {
        if (total_ == 0)
        {
            return;
        }
        const double ratio = static_cast<double>(current) / static_cast<double>(total_);
        const std::size_t filled = static_cast<std::size_t>(ratio * static_cast<double>(width_));
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < width_; ++i)
        {
            oss << (i < filled ? "#" : "-");
        }
        oss << "] " << current << "/" << total_ << " (";
        oss << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%)";
        std::cout << "\r" << oss.str() << std::flush;
        if (current == total_)
        {
            std::cout << "\n";
        }
    }

private:
    std::size_t total_ = 0;
    std::size_t width_ = 30;
};

std::string nowTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

void printUsage(const char* prog)
{
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --dataset_root <path>  Test dataset root (default: Riedones3D_Registration_Benchmark/test)\n"
        << "  --max_iter <int>       Max iterations (default: 20)\n"
        << "  --k_neighbors <int>    k for GICP covariances (default: 20)\n"
        << "  --epsilon <double>     Covariance epsilon (default: 1e-3)\n"
        << "  --sample_ratio <0-1>   Fraction of points to keep (default: 1.0)\n"
        << "  --log_dir <path>       Base log directory (default: logs)\n"
        << "  --help                 Show this help\n";
}

bool parseArgs(int argc, char** argv, Config& cfg)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            printUsage(argv[0]);
            return false;
        }

        auto getValue = [&](const std::string& a) -> std::string {
            auto eq = a.find('=');
            if (eq != std::string::npos)
            {
                return a.substr(eq + 1);
            }
            if (i + 1 < argc)
            {
                return argv[++i];
            }
            throw std::runtime_error("Missing value for " + a);
        };

        if (arg.rfind("--dataset_root", 0) == 0)
        {
            cfg.dataset_root = getValue(arg);
        }
        else if (arg.rfind("--max_iter", 0) == 0)
        {
            cfg.max_iter = std::stoi(getValue(arg));
        }
        else if (arg.rfind("--k_neighbors", 0) == 0)
        {
            cfg.k_neighbors = std::stoi(getValue(arg));
        }
        else if (arg.rfind("--epsilon", 0) == 0)
        {
            cfg.epsilon = std::stod(getValue(arg));
        }
        else if (arg.rfind("--sample_ratio", 0) == 0)
        {
            cfg.sample_ratio = std::stod(getValue(arg));
        }
        else if (arg.rfind("--log_dir", 0) == 0)
        {
            cfg.log_dir = getValue(arg);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return true;
}

PointCloud loadPlyXYZ(const std::string& path, double sample_ratio, std::mt19937& rng)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open PLY: " + path);
    }

    std::string line;
    enum class PlyFormat
    {
        Ascii,
        BinaryLittle,
        BinaryBig
    };
    PlyFormat format = PlyFormat::Ascii;
    std::size_t vertex_count = 0;
    bool in_vertex_element = false;

    struct PlyProperty
    {
        std::string type;
        std::string name;
    };
    std::vector<PlyProperty> properties;

    while (std::getline(in, line))
    {
        if (line.rfind("format ascii", 0) == 0)
        {
            format = PlyFormat::Ascii;
        }
        else if (line.rfind("format binary_little_endian", 0) == 0)
        {
            format = PlyFormat::BinaryLittle;
        }
        else if (line.rfind("format binary_big_endian", 0) == 0)
        {
            format = PlyFormat::BinaryBig;
        }
        else if (line.rfind("element vertex", 0) == 0)
        {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> vertex_count;
            in_vertex_element = true;
            properties.clear();
        }
        else if (line.rfind("element ", 0) == 0)
        {
            in_vertex_element = false;
        }
        else if (line.rfind("property list", 0) == 0)
        {
            if (in_vertex_element)
            {
                throw std::runtime_error("PLY list properties are not supported: " + path);
            }
        }
        else if (line.rfind("property ", 0) == 0)
        {
            if (in_vertex_element)
            {
                std::istringstream iss(line);
                std::string prop_token;
                PlyProperty prop;
                iss >> prop_token >> prop.type >> prop.name;
                properties.push_back(prop);
            }
        }
        else if (line == "end_header")
        {
            break;
        }
    }

    if (vertex_count == 0)
    {
        throw std::runtime_error("No vertices found in PLY header.");
    }
    if (format == PlyFormat::BinaryBig)
    {
        throw std::runtime_error("Binary big-endian PLY not supported: " + path);
    }
    if (properties.empty())
    {
        throw std::runtime_error("No vertex properties found in PLY header.");
    }

    std::vector<Eigen::Vector3d> points;
    points.reserve(static_cast<std::size_t>(std::floor(static_cast<double>(vertex_count) * sample_ratio)) + 1);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    bool has_fallback = false;
    Eigen::Vector3d fallback = Eigen::Vector3d::Zero();

    auto parseScalarAscii = [](const std::string& token) -> double {
        return std::stod(token);
    };

    auto typeSize = [](const std::string& type) -> std::size_t {
        if (type == "char" || type == "int8") return 1;
        if (type == "uchar" || type == "uint8") return 1;
        if (type == "short" || type == "int16") return 2;
        if (type == "ushort" || type == "uint16") return 2;
        if (type == "int" || type == "int32") return 4;
        if (type == "uint" || type == "uint32") return 4;
        if (type == "float" || type == "float32") return 4;
        if (type == "double" || type == "float64") return 8;
        throw std::runtime_error("Unsupported PLY property type: " + type);
    };

    auto readScalarBinary = [&](const std::string& type) -> double {
        if (type == "char" || type == "int8")
        {
            int8_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "uchar" || type == "uint8")
        {
            uint8_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "short" || type == "int16")
        {
            int16_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "ushort" || type == "uint16")
        {
            uint16_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "int" || type == "int32")
        {
            int32_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "uint" || type == "uint32")
        {
            uint32_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "float" || type == "float32")
        {
            float v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return static_cast<double>(v);
        }
        if (type == "double" || type == "float64")
        {
            double v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        }
        throw std::runtime_error("Unsupported PLY property type: " + type);
    };

    auto pushIfSelected = [&](const Eigen::Vector3d& p) {
        if (!has_fallback)
        {
            fallback = p;
            has_fallback = true;
        }
        if (sample_ratio >= 1.0 || dist(rng) <= sample_ratio)
        {
            points.push_back(p);
        }
    };

    if (format == PlyFormat::Ascii)
    {
        for (std::size_t i = 0; i < vertex_count; ++i)
        {
            if (!std::getline(in, line))
            {
                throw std::runtime_error("Failed to read ASCII vertex line.");
            }
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            tokens.reserve(properties.size());
            std::string token;
            while (iss >> token)
            {
                tokens.push_back(token);
            }
            if (tokens.size() < properties.size())
            {
                throw std::runtime_error("Malformed ASCII vertex line.");
            }
            double x = 0.0, y = 0.0, z = 0.0;
            bool has_x = false, has_y = false, has_z = false;
            for (std::size_t p = 0; p < properties.size(); ++p)
            {
                const auto& prop = properties[p];
                const double value = parseScalarAscii(tokens[p]);
                if (prop.name == "x")
                {
                    x = value;
                    has_x = true;
                }
                else if (prop.name == "y")
                {
                    y = value;
                    has_y = true;
                }
                else if (prop.name == "z")
                {
                    z = value;
                    has_z = true;
                }
            }
            if (!has_x || !has_y || !has_z)
            {
                throw std::runtime_error("PLY missing x/y/z properties.");
            }
            pushIfSelected(Eigen::Vector3d(x, y, z));
        }
    }
    else
    {
        for (std::size_t i = 0; i < vertex_count; ++i)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            bool has_x = false, has_y = false, has_z = false;
            for (const auto& prop : properties)
            {
                double value = readScalarBinary(prop.type);
                if (prop.name == "x")
                {
                    x = value;
                    has_x = true;
                }
                else if (prop.name == "y")
                {
                    y = value;
                    has_y = true;
                }
                else if (prop.name == "z")
                {
                    z = value;
                    has_z = true;
                }
            }
            if (!has_x || !has_y || !has_z)
            {
                throw std::runtime_error("PLY missing x/y/z properties.");
            }
            pushIfSelected(Eigen::Vector3d(x, y, z));
        }
    }

    if (points.empty() && has_fallback)
    {
        points.push_back(fallback);
    }
    if (points.empty())
    {
        throw std::runtime_error("No points loaded from PLY.");
    }

    PointCloud cloud(static_cast<Eigen::Index>(points.size()), 3);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        cloud.row(static_cast<Eigen::Index>(i)) = points[i].transpose();
    }

    return cloud;
}

PointCloud downsamplePointCloud(
    const PointCloud& cloud,
    double ratio,
    std::mt19937& rng)
{
    if (ratio >= 1.0)
    {
        return cloud;
    }
    if (ratio <= 0.0 || cloud.rows() == 0)
    {
        return PointCloud(0, 3);
    }

    const std::size_t total = static_cast<std::size_t>(cloud.rows());
    std::size_t keep = static_cast<std::size_t>(std::floor(ratio * static_cast<double>(total)));
    if (keep == 0)
    {
        keep = 1;
    }

    std::vector<std::size_t> indices(total);
    for (std::size_t i = 0; i < total; ++i)
    {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(keep);

    PointCloud sampled(static_cast<Eigen::Index>(keep), 3);
    for (std::size_t i = 0; i < keep; ++i)
    {
        sampled.row(static_cast<Eigen::Index>(i)) = cloud.row(static_cast<Eigen::Index>(indices[i]));
    }
    return sampled;
}

PointCloud transformPointCloud(
    const PointCloud& source,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t)
{
    PointCloud transformed = source * R.transpose();
    transformed.rowwise() += t.transpose();
    return transformed;
}

double computeSRE(
    const PointCloud& source,
    const Eigen::Matrix3d& R_est,
    const Eigen::Vector3d& t_est,
    const Eigen::Matrix3d& R_gt,
    const Eigen::Vector3d& t_gt)
{
    const int N = static_cast<int>(source.rows());
    if (N <= 0)
    {
        return 0.0;
    }

    const Eigen::Vector3d mean = source.colwise().mean();
    const Eigen::Vector3d gt_mean = R_gt * mean + t_gt;

    double sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        const Eigen::Vector3d x = source.row(i).transpose();
        const Eigen::Vector3d gt_x = R_gt * x + t_gt;
        const Eigen::Vector3d est_x = R_est * x + t_est;

        const double num = (gt_x - est_x).norm();
        const double den = (gt_x - gt_mean).norm();

        if (den > 1e-12)
        {
            sum += num / den;
        }
    }

    return sum / static_cast<double>(N);
}

double computeSumSquaredNN(
    const PointCloud& transformed_source,
    const PointCloud& target)
{
    KDTree tree(3, std::cref(target));
    tree.index_->buildIndex();

    const int N = static_cast<int>(transformed_source.rows());
    if (N <= 0)
    {
        return 0.0;
    }

    double sum = 0.0;
    size_t nn_index = 0;
    double nn_dist = 0.0;
    nanoflann::KNNResultSet<double> resultSet(1);

    for (int i = 0; i < N; ++i)
    {
        const double query_pt[3] = {
            transformed_source(i, 0),
            transformed_source(i, 1),
            transformed_source(i, 2)
        };
        resultSet.init(&nn_index, &nn_dist);
        tree.index_->findNeighbors(
            resultSet,
            query_pt,
            nanoflann::SearchParameters());
        sum += nn_dist;
    }

    return sum;
}

std::vector<PairJob> readPairsFromFile(
    const std::filesystem::path& txt_path,
    const std::string& die,
    const std::string& label)
{
    std::ifstream in(txt_path);
    if (!in)
    {
        throw std::runtime_error("Failed to open pair file: " + txt_path.string());
    }

    std::vector<PairJob> jobs;
    std::string header;
    if (!std::getline(in, header))
    {
        return jobs;
    }

    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::istringstream iss(line);
        PairJob job;
        job.die = die;
        job.label = label;
        job.txt_path = txt_path;
        double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
        if (!(iss >> job.pair_id >> job.source >> job.target >> job.overlap
                  >> t1 >> t2 >> t3 >> t4
                  >> t5 >> t6 >> t7 >> t8
                  >> t9 >> t10 >> t11 >> t12))
        {
            throw std::runtime_error("Malformed line in: " + txt_path.string());
        }
        job.R_gt << t1, t2, t3,
                    t5, t6, t7,
                    t9, t10, t11;
        job.t_gt << t4, t8, t12;
        jobs.push_back(job);
    }
    return jobs;
}

MethodResult runICP(
    const PointCloud& source,
    const PointCloud& target,
    int max_iter)
{
    MethodResult result;
    result.method = "icp";

    icp::KDTree targetTree(3, std::cref(target));
    targetTree.index_->buildIndex();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    auto start = std::chrono::steady_clock::now();

    int iter = 0;
    for (; iter < max_iter; ++iter)
    {
        icp::AccumulatedSystem sys = icp::accumulateSystem(
            source,
            target,
            &targetTree,
            R,
            t);

        auto [dR, dt] = icp::solveTransformSVD(
            sys.H,
            sys.source_mean,
            sys.target_mean);

        icp::updateTransform(R, t, dR, dt);

        const double angle = std::acos(
            std::min(1.0, std::max(-1.0, (dR.trace() - 1.0) * 0.5)));
        const double dx_norm = std::sqrt(dt.squaredNorm() + angle * angle);
        if (dx_norm < 1e-6)
        {
            ++iter;
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.iterations = iter;
    result.R = R;
    result.t = t;
    return result;
}

MethodResult runGICP(
    const PointCloud& source,
    const PointCloud& target,
    int max_iter,
    int k_neighbors,
    double epsilon)
{
    MethodResult result;
    result.method = "gicp";

    gicp::KDTree targetTree(3, std::cref(target));
    targetTree.index_->buildIndex();

    auto cov_start = std::chrono::steady_clock::now();
    auto [sourceCovs, targetCovs] = gicp::computeCovariances(
        source,
        target,
        k_neighbors,
        epsilon);
    auto cov_end = std::chrono::steady_clock::now();
    result.cov_ms = std::chrono::duration<double, std::milli>(cov_end - cov_start).count();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    auto start = std::chrono::steady_clock::now();

    int iter = 0;
    for (; iter < max_iter; ++iter)
    {
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

        if (delta.norm() < 1e-6)
        {
            ++iter;
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.iterations = iter;
    result.R = R;
    result.t = t;
    return result;
}

std::vector<std::filesystem::path> listDieDirs(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(root))
    {
        if (entry.is_directory())
        {
            dirs.push_back(entry.path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

}  // namespace

int main(int argc, char** argv)
{
    Config cfg;
    try
    {
        if (!parseArgs(argc, argv, cfg))
        {
            return 0;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    const std::filesystem::path dataset_root(cfg.dataset_root);
    if (!std::filesystem::exists(dataset_root))
    {
        std::cerr << "Dataset root not found: " << dataset_root.string() << "\n";
        return 1;
    }
    if (cfg.sample_ratio <= 0.0 || cfg.sample_ratio > 1.0)
    {
        std::cerr << "sample_ratio must be in (0, 1].\n";
        return 1;
    }

    const std::string run_stamp = nowTimestamp();
    const std::filesystem::path run_dir =
        std::filesystem::path(cfg.log_dir) / ("coins_" + run_stamp);
    std::filesystem::create_directories(run_dir);

    std::vector<PairJob> jobs;
    for (const auto& die_dir : listDieDirs(dataset_root))
    {
        const std::string die = die_dir.filename().string();
        const std::filesystem::path local_txt = dataset_root / (die + "_local.txt");
        const std::filesystem::path global_txt = dataset_root / (die + "_global.txt");

        if (std::filesystem::exists(local_txt))
        {
            auto local_jobs = readPairsFromFile(local_txt, die, "local");
            jobs.insert(jobs.end(), local_jobs.begin(), local_jobs.end());
        }
        if (std::filesystem::exists(global_txt))
        {
            auto global_jobs = readPairsFromFile(global_txt, die, "global");
            jobs.insert(jobs.end(), global_jobs.begin(), global_jobs.end());
        }
    }

    if (jobs.empty())
    {
        std::cerr << "No pair files found under: " << dataset_root.string() << "\n";
        return 1;
    }

    std::ofstream csv((run_dir / "results.csv").string());
    csv << "die,label,method,pair_id,source,target,overlap,iterations,total_ms,cov_ms,sre,sse_sum,";
    csv << "gt_t1,gt_t2,gt_t3,gt_t4,gt_t5,gt_t6,gt_t7,gt_t8,gt_t9,gt_t10,gt_t11,gt_t12,";
    csv << "est_t1,est_t2,est_t3,est_t4,est_t5,est_t6,est_t7,est_t8,est_t9,est_t10,est_t11,est_t12\n";

    std::unordered_map<std::string, PointCloud> cloud_cache;
    std::mt19937 rng(42);
    const std::size_t total_steps = jobs.size() * 2;
    ProgressBar bar(total_steps);
    std::size_t completed = 0;

    auto getCloud = [&](const std::filesystem::path& path) -> const PointCloud& {
        const std::string key = path.string() + "|r=" + std::to_string(cfg.sample_ratio);
        auto it = cloud_cache.find(key);
        if (it != cloud_cache.end())
        {
            return it->second;
        }
        PointCloud cloud = loadPlyXYZ(path.string(), cfg.sample_ratio, rng);
        auto inserted = cloud_cache.emplace(key, std::move(cloud));
        return inserted.first->second;
    };

    for (const auto& job : jobs)
    {
        const std::filesystem::path source_path = dataset_root / job.die / job.source;
        const std::filesystem::path target_path = dataset_root / job.die / job.target;

        if (!std::filesystem::exists(source_path) || !std::filesystem::exists(target_path))
        {
            throw std::runtime_error("Missing source/target PLY for die " + job.die);
        }

        PointCloud source = getCloud(source_path);
        PointCloud target = getCloud(target_path);
        if (cfg.sample_ratio < 1.0)
        {
            source = downsamplePointCloud(source, cfg.sample_ratio, rng);
            target = downsamplePointCloud(target, cfg.sample_ratio, rng);
        }

        MethodResult icp_res = runICP(source, target, cfg.max_iter);
        icp_res.sre = computeSRE(source, icp_res.R, icp_res.t, job.R_gt, job.t_gt);
        icp_res.sse = computeSumSquaredNN(transformPointCloud(source, icp_res.R, icp_res.t), target);

        csv << job.die << "," << job.label << "," << icp_res.method << ",";
        csv << job.pair_id << "," << job.source << "," << job.target << "," << job.overlap << ",";
        csv << icp_res.iterations << "," << icp_res.total_ms << "," << icp_res.cov_ms << ",";
        csv << icp_res.sre << "," << icp_res.sse << ",";
        csv << job.R_gt(0,0) << "," << job.R_gt(0,1) << "," << job.R_gt(0,2) << "," << job.t_gt(0) << ",";
        csv << job.R_gt(1,0) << "," << job.R_gt(1,1) << "," << job.R_gt(1,2) << "," << job.t_gt(1) << ",";
        csv << job.R_gt(2,0) << "," << job.R_gt(2,1) << "," << job.R_gt(2,2) << "," << job.t_gt(2) << ",";
        csv << icp_res.R(0,0) << "," << icp_res.R(0,1) << "," << icp_res.R(0,2) << "," << icp_res.t(0) << ",";
        csv << icp_res.R(1,0) << "," << icp_res.R(1,1) << "," << icp_res.R(1,2) << "," << icp_res.t(1) << ",";
        csv << icp_res.R(2,0) << "," << icp_res.R(2,1) << "," << icp_res.R(2,2) << "," << icp_res.t(2) << "\n";

        ++completed;
        bar.update(completed);

        MethodResult gicp_res = runGICP(source, target, cfg.max_iter, cfg.k_neighbors, cfg.epsilon);
        gicp_res.sre = computeSRE(source, gicp_res.R, gicp_res.t, job.R_gt, job.t_gt);
        gicp_res.sse = computeSumSquaredNN(transformPointCloud(source, gicp_res.R, gicp_res.t), target);

        csv << job.die << "," << job.label << "," << gicp_res.method << ",";
        csv << job.pair_id << "," << job.source << "," << job.target << "," << job.overlap << ",";
        csv << gicp_res.iterations << "," << gicp_res.total_ms << "," << gicp_res.cov_ms << ",";
        csv << gicp_res.sre << "," << gicp_res.sse << ",";
        csv << job.R_gt(0,0) << "," << job.R_gt(0,1) << "," << job.R_gt(0,2) << "," << job.t_gt(0) << ",";
        csv << job.R_gt(1,0) << "," << job.R_gt(1,1) << "," << job.R_gt(1,2) << "," << job.t_gt(1) << ",";
        csv << job.R_gt(2,0) << "," << job.R_gt(2,1) << "," << job.R_gt(2,2) << "," << job.t_gt(2) << ",";
        csv << gicp_res.R(0,0) << "," << gicp_res.R(0,1) << "," << gicp_res.R(0,2) << "," << gicp_res.t(0) << ",";
        csv << gicp_res.R(1,0) << "," << gicp_res.R(1,1) << "," << gicp_res.R(1,2) << "," << gicp_res.t(1) << ",";
        csv << gicp_res.R(2,0) << "," << gicp_res.R(2,1) << "," << gicp_res.R(2,2) << "," << gicp_res.t(2) << "\n";

        ++completed;
        bar.update(completed);
    }

    std::ofstream summary((run_dir / "summary.txt").string());
    summary << "dataset_root=" << dataset_root.string() << "\n";
    summary << "pairs=" << jobs.size() << "\n";
    summary << "methods=icp,gicp\n";
    summary << "max_iter=" << cfg.max_iter << "\n";
    summary << "k_neighbors=" << cfg.k_neighbors << "\n";
    summary << "epsilon=" << cfg.epsilon << "\n";
    summary << "sample_ratio=" << cfg.sample_ratio << "\n";

    std::cout << "Results written to: " << (run_dir / "results.csv").string() << "\n";
    return 0;
}
