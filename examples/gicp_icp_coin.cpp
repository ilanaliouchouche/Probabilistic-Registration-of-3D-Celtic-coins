#include "gicp.h"
#include "icp.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using ColorCloud = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 3>;

struct CloudWithAttrs
{
    PointCloud points;
    PointCloud normals;
    ColorCloud colors;
    bool has_normals = false;
    bool has_colors = false;
};

struct Config
{
    std::string input_path = "Riedones3D_Registration_Benchmark/test/die_obverse_0001/L0001D.ply";
    double angle_deg = 10.0;
    double trans_norm = 0.01;
    double noise_sigma = 0.001;
    int max_iter = 20;
    int k_neighbors = 20;
    double epsilon = 1e-3;
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
        << "Usage: " << prog << " --input <ply> [options]\n"
        << "Options:\n"
        << "  --input <ply>          Input coin PLY (default: dataset sample)\n"
        << "  --angle_deg <double>   Rotation angle (deg) applied to source (default: 10)\n"
        << "  --trans_norm <double>  Translation norm applied to source (default: 0.01)\n"
        << "  --noise_sigma <double> Gaussian noise sigma (default: 0.001)\n"
        << "  --max_iter <int>       Max iterations (default: 20)\n"
        << "  --k_neighbors <int>    k for GICP covariances (default: 20)\n"
        << "  --epsilon <double>     Covariance epsilon (default: 1e-3)\n"
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

        if (arg.rfind("--input", 0) == 0)
        {
            cfg.input_path = getValue(arg);
        }
        else if (arg.rfind("--angle_deg", 0) == 0)
        {
            cfg.angle_deg = std::stod(getValue(arg));
        }
        else if (arg.rfind("--trans_norm", 0) == 0)
        {
            cfg.trans_norm = std::stod(getValue(arg));
        }
        else if (arg.rfind("--noise_sigma", 0) == 0)
        {
            cfg.noise_sigma = std::stod(getValue(arg));
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

CloudWithAttrs loadPlyWithAttrs(const std::string& path)
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

    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3i> colors;
    points.reserve(vertex_count);
    normals.reserve(vertex_count);
    colors.reserve(vertex_count);

    auto parseAscii = [&]() {
        for (std::size_t i = 0; i < vertex_count; ++i)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            double nx = 0.0, ny = 0.0, nz = 0.0;
            int r = 0, g = 0, b = 0;
            bool has_x = false, has_y = false, has_z = false;
            bool has_nx = false, has_ny = false, has_nz = false;
            bool has_r = false, has_g = false, has_b = false;
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
                else if (prop.name == "nx")
                {
                    nx = value;
                    has_nx = true;
                }
                else if (prop.name == "ny")
                {
                    ny = value;
                    has_ny = true;
                }
                else if (prop.name == "nz")
                {
                    nz = value;
                    has_nz = true;
                }
                else if (prop.name == "red")
                {
                    r = static_cast<int>(std::lround(value));
                    has_r = true;
                }
                else if (prop.name == "green")
                {
                    g = static_cast<int>(std::lround(value));
                    has_g = true;
                }
                else if (prop.name == "blue")
                {
                    b = static_cast<int>(std::lround(value));
                    has_b = true;
                }
            }
            if (!has_x || !has_y || !has_z)
            {
                throw std::runtime_error("PLY missing x/y/z properties.");
            }
            points.emplace_back(x, y, z);
            if (has_nx && has_ny && has_nz)
            {
                normals.emplace_back(nx, ny, nz);
            }
            if (has_r && has_g && has_b)
            {
                colors.emplace_back(r, g, b);
            }
        }
    };

    auto parseBinary = [&]() {
        for (std::size_t i = 0; i < vertex_count; ++i)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            double nx = 0.0, ny = 0.0, nz = 0.0;
            int r = 0, g = 0, b = 0;
            bool has_x = false, has_y = false, has_z = false;
            bool has_nx = false, has_ny = false, has_nz = false;
            bool has_r = false, has_g = false, has_b = false;
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
                else if (prop.name == "nx")
                {
                    nx = value;
                    has_nx = true;
                }
                else if (prop.name == "ny")
                {
                    ny = value;
                    has_ny = true;
                }
                else if (prop.name == "nz")
                {
                    nz = value;
                    has_nz = true;
                }
                else if (prop.name == "red")
                {
                    r = static_cast<int>(std::lround(value));
                    has_r = true;
                }
                else if (prop.name == "green")
                {
                    g = static_cast<int>(std::lround(value));
                    has_g = true;
                }
                else if (prop.name == "blue")
                {
                    b = static_cast<int>(std::lround(value));
                    has_b = true;
                }
            }
            if (!has_x || !has_y || !has_z)
            {
                throw std::runtime_error("PLY missing x/y/z properties.");
            }
            points.emplace_back(x, y, z);
            if (has_nx && has_ny && has_nz)
            {
                normals.emplace_back(nx, ny, nz);
            }
            if (has_r && has_g && has_b)
            {
                colors.emplace_back(r, g, b);
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

    CloudWithAttrs out;
    out.points.resize(static_cast<Eigen::Index>(points.size()), 3);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        out.points.row(static_cast<Eigen::Index>(i)) = points[i].transpose();
    }
    if (normals.size() == points.size())
    {
        out.has_normals = true;
        out.normals.resize(static_cast<Eigen::Index>(normals.size()), 3);
        for (std::size_t i = 0; i < normals.size(); ++i)
        {
            out.normals.row(static_cast<Eigen::Index>(i)) = normals[i].transpose();
        }
    }
    if (colors.size() == points.size())
    {
        out.has_colors = true;
        out.colors.resize(static_cast<Eigen::Index>(colors.size()), 3);
        for (std::size_t i = 0; i < colors.size(); ++i)
        {
            out.colors.row(static_cast<Eigen::Index>(i)) = colors[i].cast<std::uint8_t>();
        }
    }

    return out;
}

void writePlyXYZ(
    const std::string& path,
    const PointCloud& positions,
    const PointCloud* normals,
    const ColorCloud* colors,
    const std::vector<std::string>& header_comments)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to write PLY: " + path);
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    for (const auto& c : header_comments)
    {
        out << "comment " << c << "\n";
    }
    out << "element vertex " << positions.rows() << "\n";
    out << "property double x\n";
    out << "property double y\n";
    out << "property double z\n";
    if (normals)
    {
        out << "property double nx\n";
        out << "property double ny\n";
        out << "property double nz\n";
    }
    if (colors)
    {
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
    }
    out << "end_header\n";

    out << std::fixed << std::setprecision(12);
    for (Eigen::Index i = 0; i < positions.rows(); ++i)
    {
        out << positions(i, 0) << " " << positions(i, 1) << " " << positions(i, 2);
        if (normals)
        {
            out << " " << (*normals)(i, 0) << " " << (*normals)(i, 1) << " " << (*normals)(i, 2);
        }
        if (colors)
        {
            out << " " << static_cast<int>((*colors)(i, 0))
                << " " << static_cast<int>((*colors)(i, 1))
                << " " << static_cast<int>((*colors)(i, 2));
        }
        out << "\n";
    }
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

PointCloud transformNormals(
    const PointCloud& normals,
    const Eigen::Matrix3d& R)
{
    if (normals.rows() == 0)
    {
        return normals;
    }
    PointCloud transformed = normals * R.transpose();
    return transformed;
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

PointCloud applyNoise(const PointCloud& source, double noise_sigma, std::mt19937& rng)
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

    if (!std::filesystem::exists(cfg.input_path))
    {
        std::cerr << "Input PLY not found: " << cfg.input_path << "\n";
        return 1;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));

    CloudWithAttrs target_full = loadPlyWithAttrs(cfg.input_path);
    const PointCloud& target = target_full.points;
    Eigen::Matrix3d R_gt = randomRotation(cfg.angle_deg, rng);
    Eigen::Vector3d t_gt = randomTranslation(cfg.trans_norm, rng);

    const Eigen::Matrix3d R_inv = R_gt.transpose();
    const Eigen::Vector3d t_inv = -R_inv * t_gt;

    PointCloud source = transformPointCloud(target, R_inv, t_inv);
    source = applyNoise(source, cfg.noise_sigma, rng);

    MethodResult icp_res = runICP(source, target, cfg.max_iter);
    MethodResult gicp_res = runGICP(source, target, cfg.max_iter, cfg.k_neighbors, cfg.epsilon);

    PointCloud icp_aligned = transformPointCloud(source, icp_res.R, icp_res.t);
    PointCloud gicp_aligned = transformPointCloud(source, gicp_res.R, gicp_res.t);
    PointCloud gt_aligned = transformPointCloud(source, R_gt, t_gt);

    PointCloud source_normals;
    if (target_full.has_normals)
    {
        source_normals = transformNormals(target_full.normals, R_inv);
        if (cfg.noise_sigma > 0.0)
        {
            for (Eigen::Index i = 0; i < source_normals.rows(); ++i)
            {
                const double n = source_normals.row(i).norm();
                if (n > 1e-12)
                {
                    source_normals.row(i) /= n;
                }
            }
        }
    }

    PointCloud gt_normals;
    PointCloud icp_normals;
    PointCloud gicp_normals;
    if (target_full.has_normals)
    {
        gt_normals = transformNormals(source_normals, R_gt);
        icp_normals = transformNormals(source_normals, icp_res.R);
        gicp_normals = transformNormals(source_normals, gicp_res.R);
    }

    const std::string run_stamp = nowTimestamp();
    const std::filesystem::path run_dir =
        std::filesystem::path(cfg.log_dir) / ("gicp_icp_coin_" + run_stamp);
    std::filesystem::create_directories(run_dir);

    try
    {
        std::filesystem::copy_file(cfg.input_path, run_dir / "target.ply",
                                   std::filesystem::copy_options::overwrite_existing);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Warning: failed to copy target PLY: " << e.what() << "\n";
    }

    std::ofstream meta((run_dir / "metadata.txt").string());
    meta << "input=" << cfg.input_path << "\n";
    meta << "angle_deg=" << cfg.angle_deg << "\n";
    meta << "trans_norm=" << cfg.trans_norm << "\n";
    meta << "noise_sigma=" << cfg.noise_sigma << "\n";
    meta << "seed=" << cfg.seed << "\n";
    meta << "max_iter=" << cfg.max_iter << "\n";
    meta << "k_neighbors=" << cfg.k_neighbors << "\n";
    meta << "epsilon=" << cfg.epsilon << "\n";
    meta << "gt_r00=" << R_gt(0,0) << " gt_r01=" << R_gt(0,1) << " gt_r02=" << R_gt(0,2) << "\n";
    meta << "gt_r10=" << R_gt(1,0) << " gt_r11=" << R_gt(1,1) << " gt_r12=" << R_gt(1,2) << "\n";
    meta << "gt_r20=" << R_gt(2,0) << " gt_r21=" << R_gt(2,1) << " gt_r22=" << R_gt(2,2) << "\n";
    meta << "gt_t=" << t_gt.transpose() << "\n";
    meta << "icp_iter=" << icp_res.iterations << " icp_ms=" << icp_res.total_ms << "\n";
    meta << "gicp_iter=" << gicp_res.iterations << " gicp_ms=" << gicp_res.total_ms << " cov_ms=" << gicp_res.cov_ms << "\n";

    const std::vector<std::string> header_comments = {
        "input=" + cfg.input_path,
        "angle_deg=" + std::to_string(cfg.angle_deg),
        "trans_norm=" + std::to_string(cfg.trans_norm),
        "noise_sigma=" + std::to_string(cfg.noise_sigma),
        "seed=" + std::to_string(cfg.seed)
    };

    writePlyXYZ((run_dir / "source_init.ply").string(),
                source,
                target_full.has_normals ? &source_normals : nullptr,
                target_full.has_colors ? &target_full.colors : nullptr,
                header_comments);
    writePlyXYZ((run_dir / "gt_aligned.ply").string(),
                gt_aligned,
                target_full.has_normals ? &gt_normals : nullptr,
                target_full.has_colors ? &target_full.colors : nullptr,
                header_comments);
    writePlyXYZ((run_dir / "icp_aligned.ply").string(),
                icp_aligned,
                target_full.has_normals ? &icp_normals : nullptr,
                target_full.has_colors ? &target_full.colors : nullptr,
                header_comments);
    writePlyXYZ((run_dir / "gicp_aligned.ply").string(),
                gicp_aligned,
                target_full.has_normals ? &gicp_normals : nullptr,
                target_full.has_colors ? &target_full.colors : nullptr,
                header_comments);

    std::cout << "Done. Logs in: " << run_dir.string() << "\n";
    return 0;
}
