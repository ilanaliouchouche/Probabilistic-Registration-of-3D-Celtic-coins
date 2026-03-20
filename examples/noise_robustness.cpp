#include "gicp.h"
#include "icp.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;

struct Config
{
    std::string dataset_root = "Riedones3D_Registration_Benchmark/test";
    int num_pieces = 5;
    int max_iter = 20;
    int k_neighbors = 20;
    double epsilon = 1e-3;
    double sample_ratio = 1.0;
    std::string jitter_list = "0,0.0005,0.001";
    std::string rot_deg_list = "0,5,10";
    std::string trans_list = "0,0.005,0.01";
    int seed = 42;
    std::string log_dir = "logs";
};

struct MethodResult
{
    std::string method;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    int iterations = 0;
    double total_ms = 0.0;
    double cov_ms = 0.0;
};

struct AggStats
{
    int count = 0;
    double sum = 0.0;
    double sumsq = 0.0;

    void add(double v)
    {
        ++count;
        sum += v;
        sumsq += v * v;
    }

    double mean() const
    {
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }

    double stddev() const
    {
        if (count <= 1)
        {
            return 0.0;
        }
        const double m = mean();
        const double var = std::max(0.0, sumsq / static_cast<double>(count) - m * m);
        return std::sqrt(var);
    }
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
        << "  --dataset_root <path>  Dataset root (default: Riedones3D_Registration_Benchmark/test)\n"
        << "  --num_pieces <int>     Number of pieces to aggregate (default: 5)\n"
        << "  --max_iter <int>       Max iterations (default: 20)\n"
        << "  --k_neighbors <int>    k for GICP covariances (default: 20)\n"
        << "  --epsilon <double>     Covariance epsilon (default: 1e-3)\n"
        << "  --sample_ratio <0-1>   Fraction of points to keep (default: 1.0)\n"
        << "  --jitter_list <csv>    Jitter sigma list (default: 0,0.0005,0.001)\n"
        << "  --rot_deg_list <csv>   Rotation deg list (default: 0,5,10)\n"
        << "  --trans_list <csv>     Translation norm list (default: 0,0.005,0.01)\n"
        << "  --seed <int>           RNG seed (default: 42)\n"
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
        else if (arg.rfind("--num_pieces", 0) == 0)
        {
            cfg.num_pieces = std::stoi(getValue(arg));
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
        else if (arg.rfind("--jitter_list", 0) == 0)
        {
            cfg.jitter_list = getValue(arg);
        }
        else if (arg.rfind("--rot_deg_list", 0) == 0)
        {
            cfg.rot_deg_list = getValue(arg);
        }
        else if (arg.rfind("--trans_list", 0) == 0)
        {
            cfg.trans_list = getValue(arg);
        }
        else if (arg.rfind("--seed", 0) == 0)
        {
            cfg.seed = std::stoi(getValue(arg));
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

std::string trim(const std::string& s)
{
    const auto begin = s.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos)
    {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(begin, end - begin + 1);
}

std::vector<double> parseList(const std::string& input)
{
    std::vector<double> values;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        token = trim(token);
        if (token.empty())
        {
            continue;
        }
        std::stringstream ts(token);
        double v = 0.0;
        if (!(ts >> v))
        {
            throw std::runtime_error("Failed to parse list entry: " + token);
        }
        values.push_back(v);
    }
    if (values.empty())
    {
        throw std::runtime_error("Empty list provided: " + input);
    }
    return values;
}

PointCloud loadPlyXYZ(const std::string& path, double sample_ratio, std::mt19937& rng)
{
    std::ifstream in(path, std::ios::binary);
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

    auto keepPoint = [&](std::size_t idx) -> bool {
        if (sample_ratio >= 1.0)
        {
            return true;
        }
        std::bernoulli_distribution keep(sample_ratio);
        return keep(rng);
    };

    std::vector<Eigen::Vector3d> points;
    points.reserve(vertex_count);

    auto parseAscii = [&]() {
        for (std::size_t i = 0; i < vertex_count; ++i)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            bool has_x = false, has_y = false, has_z = false;
            for (const auto& prop : properties)
            {
                double value = 0.0;
                if (!(in >> value))
                {
                    throw std::runtime_error("Failed reading ASCII PLY data: " + path);
                }
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
            if (keepPoint(i))
            {
                points.emplace_back(x, y, z);
            }
        }
    };

    auto readBinaryValue = [&](const std::string& type) -> double {
        if (type == "float" || type == "float32")
        {
            float v;
            in.read(reinterpret_cast<char*>(&v), sizeof(float));
            return static_cast<double>(v);
        }
        if (type == "double" || type == "float64")
        {
            double v;
            in.read(reinterpret_cast<char*>(&v), sizeof(double));
            return v;
        }
        if (type == "uchar" || type == "uint8")
        {
            std::uint8_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(std::uint8_t));
            return static_cast<double>(v);
        }
        if (type == "char" || type == "int8")
        {
            std::int8_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(std::int8_t));
            return static_cast<double>(v);
        }
        if (type == "ushort" || type == "uint16")
        {
            std::uint16_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(std::uint16_t));
            return static_cast<double>(v);
        }
        if (type == "short" || type == "int16")
        {
            std::int16_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(std::int16_t));
            return static_cast<double>(v);
        }
        if (type == "uint" || type == "uint32")
        {
            std::uint32_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(std::uint32_t));
            return static_cast<double>(v);
        }
        if (type == "int" || type == "int32")
        {
            std::int32_t v;
            in.read(reinterpret_cast<char*>(&v), sizeof(std::int32_t));
            return static_cast<double>(v);
        }
        throw std::runtime_error("Unsupported PLY property type: " + type);
    };

    auto parseBinary = [&]() {
        for (std::size_t i = 0; i < vertex_count; ++i)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            bool has_x = false, has_y = false, has_z = false;
            for (const auto& prop : properties)
            {
                double value = readBinaryValue(prop.type);
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
            if (keepPoint(i))
            {
                points.emplace_back(x, y, z);
            }
        }
    };

    if (format == PlyFormat::Ascii)
    {
        parseAscii();
    }
    else
    {
        parseBinary();
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

[[maybe_unused]] PointCloud downsamplePointCloud(const PointCloud& cloud, double ratio, std::mt19937& rng)
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

double computeRMSE(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t)
{
    const PointCloud transformed = transformPointCloud(source, R, t);
    const PointCloud diff = target - transformed;
    const double mse = diff.rowwise().squaredNorm().mean();
    return std::sqrt(mse);
}

MethodResult runICP(const PointCloud& source, const PointCloud& target, int max_iter)
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

std::vector<std::filesystem::path> listPlyFiles(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".ply")
        {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

Eigen::Matrix3d randomRotation(double angle_deg, std::mt19937& rng)
{
    if (angle_deg == 0.0)
    {
        return Eigen::Matrix3d::Identity();
    }

    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::Vector3d axis(normal(rng), normal(rng), normal(rng));
    if (axis.norm() < 1e-9)
    {
        axis = Eigen::Vector3d::UnitZ();
    }
    axis.normalize();

    const double pi = std::acos(-1.0);
    const double angle_rad = angle_deg * (pi / 180.0);
    Eigen::AngleAxisd aa(angle_rad, axis);
    return aa.toRotationMatrix();
}

Eigen::Vector3d randomTranslation(double norm, std::mt19937& rng)
{
    if (norm == 0.0)
    {
        return Eigen::Vector3d::Zero();
    }
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::Vector3d dir(normal(rng), normal(rng), normal(rng));
    if (dir.norm() < 1e-9)
    {
        dir = Eigen::Vector3d::UnitX();
    }
    dir.normalize();
    return dir * norm;
}

PointCloud applyNoise(
    const PointCloud& source,
    double noise_sigma,
    std::mt19937& rng)
{
    if (noise_sigma <= 0.0)
    {
        return source;
    }
    std::normal_distribution<double> noise(0.0, noise_sigma);
    PointCloud noisy = source;
    for (Eigen::Index i = 0; i < noisy.rows(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            noisy(i, j) += noise(rng);
        }
    }
    return noisy;
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

    if (cfg.num_pieces <= 0)
    {
        std::cerr << "num_pieces must be > 0.\n";
        return 1;
    }
    if (cfg.sample_ratio <= 0.0 || cfg.sample_ratio > 1.0)
    {
        std::cerr << "sample_ratio must be in (0, 1].\n";
        return 1;
    }

    const std::filesystem::path dataset_root(cfg.dataset_root);
    if (!std::filesystem::exists(dataset_root))
    {
        std::cerr << "Dataset root not found: " << dataset_root.string() << "\n";
        return 1;
    }

    std::vector<double> jitter_sigmas;
    std::vector<double> rot_degs;
    std::vector<double> trans_norms;
    try
    {
        jitter_sigmas = parseList(cfg.jitter_list);
        rot_degs = parseList(cfg.rot_deg_list);
        trans_norms = parseList(cfg.trans_list);
    }
    catch (const std::exception& e)
    {
        std::cerr << "List parse error: " << e.what() << "\n";
        return 1;
    }

    std::vector<std::filesystem::path> all_ply = listPlyFiles(dataset_root);
    if (all_ply.empty())
    {
        std::cerr << "No PLY files found under: " << dataset_root.string() << "\n";
        return 1;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));
    std::shuffle(all_ply.begin(), all_ply.end(), rng);
    if (static_cast<std::size_t>(cfg.num_pieces) < all_ply.size())
    {
        all_ply.resize(static_cast<std::size_t>(cfg.num_pieces));
    }

    const std::string run_stamp = nowTimestamp();
    const std::filesystem::path run_dir =
        std::filesystem::path(cfg.log_dir) / ("riedones3d_noise_" + run_stamp);
    std::filesystem::create_directories(run_dir);

    std::ofstream meta((run_dir / "metadata.txt").string());
    meta << "dataset_root=" << cfg.dataset_root << "\n";
    meta << "num_pieces=" << cfg.num_pieces << "\n";
    meta << "max_iter=" << cfg.max_iter << "\n";
    meta << "k_neighbors=" << cfg.k_neighbors << "\n";
    meta << "epsilon=" << cfg.epsilon << "\n";
    meta << "sample_ratio=" << cfg.sample_ratio << "\n";
    meta << "jitter_list=" << cfg.jitter_list << "\n";
    meta << "rot_deg_list=" << cfg.rot_deg_list << "\n";
    meta << "trans_list=" << cfg.trans_list << "\n";
    meta << "seed=" << cfg.seed << "\n";
    meta << "pieces=" << all_ply.size() << "\n";
    for (const auto& p : all_ply)
    {
        meta << "piece=" << p.string() << "\n";
    }

    std::ofstream csv((run_dir / "results.csv").string());
    csv << "piece,method,jitter_sigma,rot_deg,trans_norm,rmse,iterations,total_ms,cov_ms,";
    csv << "gt_r00,gt_r01,gt_r02,gt_r10,gt_r11,gt_r12,gt_r20,gt_r21,gt_r22,gt_tx,gt_ty,gt_tz\n";

    std::map<std::tuple<int, int, int, std::string>, AggStats> agg;

    std::unordered_map<std::string, PointCloud> cloud_cache;
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

    for (std::size_t pidx = 0; pidx < all_ply.size(); ++pidx)
    {
        const auto& ply_path = all_ply[pidx];
        PointCloud target = getCloud(ply_path);

        for (std::size_t j = 0; j < jitter_sigmas.size(); ++j)
        {
            for (std::size_t r = 0; r < rot_degs.size(); ++r)
            {
                for (std::size_t t = 0; t < trans_norms.size(); ++t)
                {
                    const double jitter_sigma = jitter_sigmas[j];
                    const double rot_deg = rot_degs[r];
                    const double trans_norm = trans_norms[t];

                    std::mt19937 local_rng(static_cast<std::mt19937::result_type>(
                        cfg.seed + static_cast<int>(pidx * 73856093u + j * 19349663u + r * 83492791u + t * 2654435761u)));

                    Eigen::Matrix3d R_gt = randomRotation(rot_deg, local_rng);
                    Eigen::Vector3d t_gt = randomTranslation(trans_norm, local_rng);

                    const Eigen::Matrix3d R_inv = R_gt.transpose();
                    const Eigen::Vector3d t_inv = -R_inv * t_gt;
                    PointCloud source = transformPointCloud(target, R_inv, t_inv);
                    source = applyNoise(source, jitter_sigma, local_rng);

                    MethodResult icp_res = runICP(source, target, cfg.max_iter);
                    const double icp_rmse = computeRMSE(source, target, icp_res.R, icp_res.t);
                    csv << ply_path.string() << "," << icp_res.method << ",";
                    csv << jitter_sigma << "," << rot_deg << "," << trans_norm << ",";
                    csv << icp_rmse << "," << icp_res.iterations << "," << icp_res.total_ms << "," << icp_res.cov_ms << ",";
                    csv << R_gt(0,0) << "," << R_gt(0,1) << "," << R_gt(0,2) << ",";
                    csv << R_gt(1,0) << "," << R_gt(1,1) << "," << R_gt(1,2) << ",";
                    csv << R_gt(2,0) << "," << R_gt(2,1) << "," << R_gt(2,2) << ",";
                    csv << t_gt(0) << "," << t_gt(1) << "," << t_gt(2) << "\n";

                    agg[std::make_tuple(static_cast<int>(j), static_cast<int>(r), static_cast<int>(t), icp_res.method)].add(icp_rmse);

                    MethodResult gicp_res = runGICP(source, target, cfg.max_iter, cfg.k_neighbors, cfg.epsilon);
                    const double gicp_rmse = computeRMSE(source, target, gicp_res.R, gicp_res.t);
                    csv << ply_path.string() << "," << gicp_res.method << ",";
                    csv << jitter_sigma << "," << rot_deg << "," << trans_norm << ",";
                    csv << gicp_rmse << "," << gicp_res.iterations << "," << gicp_res.total_ms << "," << gicp_res.cov_ms << ",";
                    csv << R_gt(0,0) << "," << R_gt(0,1) << "," << R_gt(0,2) << ",";
                    csv << R_gt(1,0) << "," << R_gt(1,1) << "," << R_gt(1,2) << ",";
                    csv << R_gt(2,0) << "," << R_gt(2,1) << "," << R_gt(2,2) << ",";
                    csv << t_gt(0) << "," << t_gt(1) << "," << t_gt(2) << "\n";

                    agg[std::make_tuple(static_cast<int>(j), static_cast<int>(r), static_cast<int>(t), gicp_res.method)].add(gicp_rmse);
                }
            }
        }
    }

    std::ofstream summary((run_dir / "summary.csv").string());
    summary << "jitter_sigma,rot_deg,trans_norm,method,count,rmse_mean,rmse_std\n";
    for (const auto& kv : agg)
    {
        const auto& key = kv.first;
        const auto& stats = kv.second;
        const int j = std::get<0>(key);
        const int r = std::get<1>(key);
        const int t = std::get<2>(key);
        const std::string& method = std::get<3>(key);
        summary << jitter_sigmas[static_cast<std::size_t>(j)] << ",";
        summary << rot_degs[static_cast<std::size_t>(r)] << ",";
        summary << trans_norms[static_cast<std::size_t>(t)] << ",";
        summary << method << "," << stats.count << ",";
        summary << stats.mean() << "," << stats.stddev() << "\n";
    }

    std::cout << "Done. Logs in: " << run_dir.string() << "\n";
    return 0;
}
