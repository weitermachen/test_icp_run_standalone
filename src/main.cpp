// Author: weitermachen
// Time: 2026-03-24

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "HVCloudPairICP.h"
#include "HVUtils.h"

/**
 * @brief 双路输出流缓冲区
 *
 * @description 将日志内容同时写入文件与控制台，供 LogManager 接管标准输出时复用。
 */
class DualStreambuf : public std::streambuf {
public:
    DualStreambuf(std::ostream& file, std::streambuf* console_buf)
        : file_(file), console_buf_(console_buf) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        file_.write(s, n);
        file_.flush();
        console_buf_->sputn(s, n);
        return n;
    }

    virtual int overflow(int c) override {
        if (c != EOF) {
            file_.put(c);
            file_.flush();
            console_buf_->sputc(c);
        }
        return c;
    }

private:
    std::ostream& file_;
    std::streambuf* console_buf_;
};

/**
 * @brief 日志生命周期管理器
 *
 * @description 负责创建 result/log.txt，并在对象生命周期内把 std::cout / std::cerr
 * 重定向到文件与控制台双写缓冲区，析构时自动恢复。
 */
class LogManager {
public:
    LogManager()
        : cout_backup_(nullptr)
        , cerr_backup_(nullptr)
        , dual_cout_(nullptr)
        , dual_cerr_(nullptr) {
        std::error_code ec;
        const std::filesystem::path result_dir = std::filesystem::path("result");
        std::filesystem::create_directories(result_dir, ec);

        if (ec) {
            return;
        }

        log_file_.open(result_dir / "log.txt", std::ios::out | std::ios::trunc);
        if (log_file_.is_open()) {
            cout_backup_ = std::cout.rdbuf();
            cerr_backup_ = std::cerr.rdbuf();
            dual_cout_ = new DualStreambuf(log_file_, cout_backup_);
            dual_cerr_ = new DualStreambuf(log_file_, cerr_backup_);
            std::cout.rdbuf(dual_cout_);
            std::cerr.rdbuf(dual_cerr_);
        }
    }
    
    ~LogManager() {
        if (log_file_.is_open()) {
            std::cout.rdbuf(cout_backup_);
            std::cerr.rdbuf(cerr_backup_);
            log_file_ << "\n=== Logging completed ===" << std::endl;
            log_file_.close();
            delete dual_cout_;
            delete dual_cerr_;
            std::cout << "Log saved to: result/log.txt" << std::endl;
        }
    }
    
    bool IsOpen() const { return log_file_.is_open(); }

private:
    std::ofstream log_file_;
    std::streambuf* cout_backup_;
    std::streambuf* cerr_backup_;
    DualStreambuf* dual_cout_;
    DualStreambuf* dual_cerr_;
};

namespace {

using PCLCloud = pcl::PointCloud<pcl::PointXYZ>;
using PCLCloudPtr = PCLCloud::Ptr;

const std::string kDefaultConfigFileName = "config/icp_run_config.json";
const std::string kDefaultLeftCloudPath = "D:/hans/qunyingxihan/calibration/pointcloud/left_1_sv_mixtif_sv - Cloud.txt";
const std::string kDefaultRightCloudPath = "D:/hans/qunyingxihan/calibration/pointcloud/right_1_sv_mixtif_sv - Cloud.txt";
const std::string kDefaultResultDir = "D:/hans/qunyingxihan/calibration/pointcloud/result";
const std::string kDefaultTransformedSourceResultName = "transformed_source_by_registration.ply";
const std::string kDefaultResultJsonName = "registration_result.json";
const std::string kDefaultInitialTransformJsonName = "initial_transform_for_icp.json";

/** @brief 体素滤波参数。 */
struct FilterParams {
    float leaf_size = 0.6f;
};

/** @brief 统计离群点滤波配置。 */
struct NoiseFilterConfig {
    bool enabled = false;
    int mean_k = 20;
    double stddev_mul_thresh = 1.0;
};

/** @brief 单阶段配准参数集合。 */
struct RegistrationParams {
    int method = 1;                         // 0: ICP, 1: GICP, 2: LP-ICP
    double voxel_size = 0.0;                // <=0 means disabled in HVCloudPairICP
    double max_correspondence_distance = 8.0;
    int max_iterations = 120;
    bool use_initial_guess = false;
    std::string initial_transform_json = "";
};

/** @brief 源点云镜像轴配置。 */
struct MirrorAxes {
    bool x = true;
    bool y = false;
    bool z = false;

    bool AnyEnabled() const {
        return x || y || z;
    }
};

/** @brief coarse/refine 两阶段执行开关。 */
struct PipelineConfig {
    bool run_coarse = true;
    bool run_refine = true;
};

/** @brief 配准评估配置。 */
struct EvaluationConfig {
    bool enabled = false;
    bool evaluate_coarse = true;
    bool evaluate_refine = true;
    std::size_t max_sample_points = 50000;
};

/** @brief 结果输出配置。 */
struct OutputConfig {
    std::string result_dir = kDefaultResultDir;
    std::string transformed_source_name = kDefaultTransformedSourceResultName;
    std::string result_json_name = kDefaultResultJsonName;
    std::string initial_transform_json_name = kDefaultInitialTransformJsonName;
};

/** @brief 多位姿融合优化配置。 */
struct FusionConfig {
    std::string candidate_result_dir = "result";
    std::string data_dir = "data";
    std::string output_json_name = "fused_registration_result.json";
    std::size_t max_sample_points_per_pair = 50000;
    std::size_t optimization_sample_points_per_pair = 8000;
    double trim_ratio = 0.1;
    double max_refine_rmse = -1.0;  // <=0 means disabled
    int optimization_max_iterations = 40;
    double optimization_rotation_step_deg = 0.5;
    double optimization_translation_step = 0.5;
    double optimization_min_rotation_step_deg = 0.005;
    double optimization_min_translation_step = 0.01;
};

/** @brief 最近邻评估统计结果。 */
struct DistanceMetrics {
    std::size_t sampled_points = 0;
    std::size_t sample_stride = 1;
    double mean_distance = 0.0;
    double rmse = 0.0;
    double max_distance = 0.0;
};

nlohmann::json RigidTransformToJson(const double rotation[3][3],
                                    const double translation[3]);
PCLCloudPtr MakeFiniteCloud(const PCLCloudPtr& input);
PCLCloudPtr LimitCloudPoints(const PCLCloudPtr& input, std::size_t max_points);
void MirrorCloudByNegatingX(PCLCloudPtr& cloud,
                            const MirrorAxes& axes,
                            bool save_mirrored_cloud = true);

/**
 * @brief 构建默认 refine 配准参数
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
RegistrationParams BuildDefaultRefineParams() {
    RegistrationParams params;
    params.method = 2;
    params.voxel_size = 0.3;
    params.max_correspondence_distance = 2.5;
    params.max_iterations = 140;
    params.use_initial_guess = true;
    params.initial_transform_json = "";
    return params;
}

/** @brief 程序完整运行配置。 */
struct AppConfig {
    std::string run_mode = "registration";  // registration | fusion
    std::string source_cloud_path = kDefaultLeftCloudPath;
    std::string target_cloud_path = kDefaultRightCloudPath;
    bool enable_txt_cache = true;
    FilterParams filter;
    NoiseFilterConfig noise_filter;
    RegistrationParams coarse_registration;
    RegistrationParams refine_registration = BuildDefaultRefineParams();
    MirrorAxes mirror_axes;
    PipelineConfig pipeline;
    EvaluationConfig evaluation;
    OutputConfig output;
    FusionConfig fusion;
};

/**
 * @brief 将字符串转换为小写
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/**
 * @brief 去除字符串首尾空白字符
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::string TrimWhitespace(const std::string& text) {
    size_t begin = 0;
    size_t end = text.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(begin, end - begin);
}

/**
 * @brief 从 JSON 对象中读取可选字段
 * @param object_json 输入 JSON 对象
 * @param key 目标字段名
 * @param output 字段存在时写入的输出变量
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
template <typename T>
void ReadOptionalValue(const nlohmann::json& object_json, const char* key, T& output) {
    auto it = object_json.find(key);
    if (it != object_json.end() && !it->is_null()) {
        output = it->get<T>();
    }
}

/**
 * @brief 判断字符是否为 TXT 点云分隔符
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool IsTxtSeparator(char c) {
    return c == ' ' || c == '\t' || c == ',' || c == ';';
}

/**
 * @brief 判断文件扩展名是否为支持的点云格式
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool IsSupportedCloudExtension(const std::string& ext) {
    return ext == ".txt" || ext == ".pcd" || ext == ".ply";
}

/**
 * @brief 忽略大小写判断字符串前缀
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool StartsWithCaseInsensitive(const std::string& text, const std::string& prefix) {
    if (text.size() < prefix.size()) {
        return false;
    }

    for (std::size_t i = 0; i < prefix.size(); ++i) {
        const char a = static_cast<char>(
            std::tolower(static_cast<unsigned char>(text[i])));
        const char b = static_cast<char>(
            std::tolower(static_cast<unsigned char>(prefix[i])));
        if (a != b) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 提取字符串中的数字字符
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::string ExtractDigits(const std::string& text) {
    std::string digits;
    for (char c : text) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            digits.push_back(c);
        }
    }
    return digits;
}

/**
 * @brief 规范化数字标识字符串
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::string NormalizeNumericToken(const std::string& token) {
    if (token.empty()) {
        return token;
    }

    std::size_t first_non_zero = 0;
    while (first_non_zero + 1 < token.size() && token[first_non_zero] == '0') {
        ++first_non_zero;
    }
    return token.substr(first_non_zero);
}

/**
 * @brief 提取指定前缀后的数字标识
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ExtractNumericTokenAfterPrefix(const std::string& file_name_lower,
                                    const std::string& prefix_lower,
                                    std::string& token_out) {
    token_out.clear();
    if (!StartsWithCaseInsensitive(file_name_lower, prefix_lower)) {
        return false;
    }

    std::size_t pos = prefix_lower.size();
    if (pos < file_name_lower.size() && file_name_lower[pos] == '_') {
        ++pos;
    }

    const std::size_t begin = pos;
    while (pos < file_name_lower.size() &&
           std::isdigit(static_cast<unsigned char>(file_name_lower[pos]))) {
        ++pos;
    }

    if (pos == begin) {
        return false;
    }

    token_out = file_name_lower.substr(begin, pos - begin);
    return true;
}

/**
 * @brief 判断点坐标是否全部为有限值
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool IsFinitePoint(const pcl::PointXYZ& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

const char* SkipLeadingTextSpaces(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
        ++p;
    }
    return p;
}

const char* SkipTxtSeparators(const char* p) {
    while (*p != '\0' && IsTxtSeparator(*p)) {
        ++p;
    }
    return p;
}

/**
 * @brief 从文本指针位置解析浮点数
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ParseFloatToken(const char*& p, float& value) {
    p = SkipTxtSeparators(p);
    if (*p == '\0') {
        return false;
    }

    errno = 0;
    char* end_ptr = nullptr;
    const float parsed = std::strtof(p, &end_ptr);
    if (end_ptr == p || errno == ERANGE || !std::isfinite(parsed)) {
        return false;
    }

    value = parsed;
    p = end_ptr;
    return true;
}

/**
 * @brief 构建 TXT 点云缓存文件路径
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::filesystem::path BuildTxtCachePath(const std::filesystem::path& txt_path) {
    std::filesystem::path cache_path = txt_path;
    cache_path += ".cache.pcd";
    return cache_path;
}

/**
 * @brief 解析一行 TXT 点云坐标
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool TryParseTxtPointLine(const std::string& line, pcl::PointXYZ& point) {
    const char* p = SkipLeadingTextSpaces(line.c_str());
    if (*p == '\0') {
        return false;
    }

    if (*p == '#' || (*p == '/' && *(p + 1) == '/')) {
        return false;
    }

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    if (!ParseFloatToken(p, x) || !ParseFloatToken(p, y) || !ParseFloatToken(p, z)) {
        return false;
    }

    point.x = x;
    point.y = y;
    point.z = z;
    return true;
}

/**
 * @brief 基于基础目录解析路径
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::string ResolvePathFromBase(const std::string& raw_path,
                                const std::filesystem::path& base_dir) {
    if (raw_path.empty()) {
        return raw_path;
    }

    const std::filesystem::path path_value(raw_path);
    if (path_value.is_absolute()) {
        return path_value.lexically_normal().string();
    }

    // Prefer project-root working directory; fallback to config directory for compatibility.
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec && !cwd.empty()) {
        const std::filesystem::path cwd_candidate = (cwd / path_value).lexically_normal();
        if (std::filesystem::exists(cwd_candidate, ec) && !ec) {
            return cwd_candidate.string();
        }

        ec.clear();
        const std::filesystem::path config_candidate = (base_dir / path_value).lexically_normal();
        if (std::filesystem::exists(config_candidate, ec) && !ec) {
            return config_candidate.string();
        }

        return cwd_candidate.string();
    }

    return (base_dir / path_value).lexically_normal().string();
}

/**
 * @brief 从字符串解析镜像轴配置
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ParseMirrorAxesFromString(const std::string& axes_text,
                               MirrorAxes& axes,
                               std::string& error) {
    MirrorAxes parsed_axes;
    parsed_axes.x = false;
    parsed_axes.y = false;
    parsed_axes.z = false;

    for (const char raw_char : axes_text) {
        const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(raw_char)));
        if (c == 'x') {
            parsed_axes.x = true;
        } else if (c == 'y') {
            parsed_axes.y = true;
        } else if (c == 'z') {
            parsed_axes.z = true;
        } else if (std::isspace(static_cast<unsigned char>(c)) ||
                   c == ',' || c == ';' || c == '|' || c == '+' || c == '-') {
            continue;
        } else {
            error = "Invalid mirror axis token: " + std::string(1, raw_char);
            return false;
        }
    }

    axes = parsed_axes;
    return true;
}

/**
 * @brief 解析镜像轴 JSON 配置
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ParseMirrorAxesConfig(const nlohmann::json& mirror_axes_json,
                           MirrorAxes& axes,
                           std::string& error) {
    if (mirror_axes_json.is_string()) {
        return ParseMirrorAxesFromString(mirror_axes_json.get<std::string>(), axes, error);
    }

    if (mirror_axes_json.is_array()) {
        MirrorAxes parsed_axes;
        parsed_axes.x = false;
        parsed_axes.y = false;
        parsed_axes.z = false;

        for (const auto& item : mirror_axes_json) {
            if (!item.is_string()) {
                error = "mirror_axes array must contain axis strings x/y/z.";
                return false;
            }
            std::string token_error;
            MirrorAxes token_axes;
            if (!ParseMirrorAxesFromString(item.get<std::string>(), token_axes, token_error)) {
                error = token_error;
                return false;
            }
            parsed_axes.x = parsed_axes.x || token_axes.x;
            parsed_axes.y = parsed_axes.y || token_axes.y;
            parsed_axes.z = parsed_axes.z || token_axes.z;
        }

        axes = parsed_axes;
        return true;
    }

    if (mirror_axes_json.is_object()) {
        MirrorAxes parsed_axes;
        parsed_axes.x = false;
        parsed_axes.y = false;
        parsed_axes.z = false;

        ReadOptionalValue(mirror_axes_json, "x", parsed_axes.x);
        ReadOptionalValue(mirror_axes_json, "y", parsed_axes.y);
        ReadOptionalValue(mirror_axes_json, "z", parsed_axes.z);

        axes = parsed_axes;
        return true;
    }

    error = "mirror_axes must be string, array, or object.";
    return false;
}

/**
 * @brief 应用评估阶段标记
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ApplyStageToken(const std::string& token,
                     bool& evaluate_coarse,
                     bool& evaluate_refine,
                     std::string& error) {
    const std::string lowered = ToLower(token);
    if (lowered.empty()) {
        return true;
    }

    if (lowered == "coarse" || lowered == "stage1" || lowered == "stage-1") {
        evaluate_coarse = true;
        return true;
    }
    if (lowered == "refine" || lowered == "stage2" || lowered == "stage-2") {
        evaluate_refine = true;
        return true;
    }
    if (lowered == "both" || lowered == "all") {
        evaluate_coarse = true;
        evaluate_refine = true;
        return true;
    }

    error = "Unsupported evaluation stage token: " + token;
    return false;
}

/**
 * @brief 解析评估阶段配置
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ParseEvaluationStages(const nlohmann::json& stages_json,
                           EvaluationConfig& evaluation,
                           std::string& error) {
    bool evaluate_coarse = false;
    bool evaluate_refine = false;

    if (stages_json.is_string()) {
        std::string text = stages_json.get<std::string>();
        for (char& c : text) {
            if (c == ',' || c == ';' || c == '|' || c == '+') {
                c = ' ';
            }
        }

        std::istringstream stream(text);
        std::string token;
        while (stream >> token) {
            if (!ApplyStageToken(token, evaluate_coarse, evaluate_refine, error)) {
                return false;
            }
        }
    } else if (stages_json.is_array()) {
        for (const auto& item : stages_json) {
            if (!item.is_string()) {
                error = "evaluation.stages array must contain strings.";
                return false;
            }

            if (!ApplyStageToken(item.get<std::string>(), evaluate_coarse, evaluate_refine, error)) {
                return false;
            }
        }
    } else {
        error = "evaluation.stages must be a string or array.";
        return false;
    }

    if (!evaluate_coarse && !evaluate_refine) {
        error = "evaluation.stages resolved to empty stage set.";
        return false;
    }

    evaluation.evaluate_coarse = evaluate_coarse;
    evaluation.evaluate_refine = evaluate_refine;
    return true;
}

/**
 * @brief 读取初始位姿 JSON 参数
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ReadInitialTransformJsonParam(const nlohmann::json& params_json,
                                   const char* key,
                                   std::string& value,
                                   std::string& error) {
    auto it = params_json.find(key);
    if (it == params_json.end() || it->is_null()) {
        return true;
    }

    if (it->is_string()) {
        value = it->get<std::string>();
        return true;
    }

    if (it->is_object()) {
        value = it->dump();
        return true;
    }

    error = std::string("registration field '") + key + "' must be string or object.";
    return false;
}

/**
 * @brief 读取配准参数配置
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ReadRegistrationParams(const nlohmann::json& params_json,
                            RegistrationParams& params,
                            std::string& error) {
    ReadOptionalValue(params_json, "method", params.method);
    ReadOptionalValue(params_json, "voxel_size", params.voxel_size);
    ReadOptionalValue(params_json, "max_correspondence_distance", params.max_correspondence_distance);
    ReadOptionalValue(params_json, "max_iterations", params.max_iterations);
    ReadOptionalValue(params_json, "use_initial_guess", params.use_initial_guess);
    return ReadInitialTransformJsonParam(params_json,
                                         "initial_transform_json",
                                         params.initial_transform_json,
                                         error);
}

/**
 * @brief 判断目录是否为工程根目录
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool IsProjectRootDirectory(const std::filesystem::path& dir) {
    std::error_code ec;
    const bool has_cmake = std::filesystem::exists(dir / "CMakeLists.txt", ec);
    if (ec || !has_cmake) {
        return false;
    }

    ec.clear();
    const bool has_default_config = std::filesystem::exists(
        dir / std::filesystem::path(kDefaultConfigFileName),
        ec);
    return !ec && has_default_config;
}

/**
 * @brief 解析工程根目录路径
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::filesystem::path ResolveProjectRootPath(int argc, char** argv) {
    std::vector<std::filesystem::path> start_points;
    std::error_code ec;

    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec && !cwd.empty()) {
        start_points.push_back(cwd);
    }

    ec.clear();
    if (argc > 0 && argv != nullptr && argv[0] != nullptr) {
        const std::filesystem::path exe_path =
            std::filesystem::absolute(std::filesystem::path(argv[0]), ec);
        if (!ec && !exe_path.empty()) {
            start_points.push_back(exe_path.parent_path());
        }
    }

    for (const auto& start : start_points) {
        std::filesystem::path cursor = start;
        for (int depth = 0; depth < 8 && !cursor.empty(); ++depth) {
            if (IsProjectRootDirectory(cursor)) {
                return cursor;
            }

            const std::filesystem::path parent = cursor.parent_path();
            if (parent == cursor) {
                break;
            }
            cursor = parent;
        }
    }

    return std::filesystem::path();
}

/**
 * @brief 将工作目录切换到工程根目录
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void EnsureWorkingDirectoryAtProjectRoot(int argc, char** argv) {
    const std::filesystem::path root_path = ResolveProjectRootPath(argc, argv);
    if (root_path.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::current_path(root_path, ec);
}

/**
 * @brief 解析默认配置文件路径
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::filesystem::path ResolveDefaultConfigPath(int argc, char** argv) {
    const std::filesystem::path default_relative_path =
        std::filesystem::path(kDefaultConfigFileName).lexically_normal();

    std::vector<std::filesystem::path> search_bases;
    std::error_code ec;

    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec && !cwd.empty()) {
        search_bases.push_back(cwd);
    }
    ec.clear();

    if (argc > 0 && argv != nullptr && argv[0] != nullptr) {
        const std::filesystem::path exe_path =
            std::filesystem::absolute(std::filesystem::path(argv[0]), ec);
        if (!ec && !exe_path.empty()) {
            std::filesystem::path base = exe_path.parent_path();
            for (int depth = 0; depth < 6 && !base.empty(); ++depth) {
                search_bases.push_back(base);
                const std::filesystem::path parent = base.parent_path();
                if (parent == base) {
                    break;
                }
                base = parent;
            }
        }
    }

    for (const auto& base : search_bases) {
        const std::filesystem::path candidate = (base / default_relative_path).lexically_normal();
        if (std::filesystem::exists(candidate, ec) && !ec) {
            return candidate;
        }
        ec.clear();
    }

    if (!search_bases.empty()) {
        return (search_bases.front() / default_relative_path).lexically_normal();
    }
    return default_relative_path;
}

/**
 * @brief 解析运行时配置文件路径
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::filesystem::path ResolveConfigPath(int argc, char** argv) {
    if (argc > 1 && argv[1] != nullptr) {
        const std::string arg_path = argv[1];
        if (!arg_path.empty()) {
            return std::filesystem::path(arg_path);
        }
    }

    const std::filesystem::path default_config_path = ResolveDefaultConfigPath(argc, argv);
    std::cout << "Input config .json path (press Enter to use default):" << std::endl
              << "  " << default_config_path.string() << std::endl
              << "> " << std::flush;

    std::string input_path;
    if (!std::getline(std::cin, input_path)) {
        return default_config_path;
    }

    input_path = TrimWhitespace(input_path);
    if (input_path.empty()) {
        return default_config_path;
    }

    return std::filesystem::path(input_path);
}

/**
 * @brief 从 JSON 文件加载应用配置
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool LoadConfigFromJsonFile(const std::filesystem::path& config_path,
                            AppConfig& config,
                            std::string& error) {
    std::ifstream config_stream(config_path.string(), std::ios::in);
    if (!config_stream.is_open()) {
        error = "Failed to open config file: " + config_path.string();
        return false;
    }

    nlohmann::json root = nlohmann::json::parse(config_stream, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        error = "Failed to parse config json: " + config_path.string();
        return false;
    }

    const std::filesystem::path base_dir = config_path.parent_path();

    try {
        // 基础运行模式与输入输出路径优先从配置中读取，再结合配置文件目录解析相对路径。
        ReadOptionalValue(root, "run_mode", config.run_mode);

        if (root.contains("input") && root["input"].is_object()) {
            const auto& input_json = root["input"];

            std::string source_path = config.source_cloud_path;
            std::string target_path = config.target_cloud_path;
            ReadOptionalValue(input_json, "source_cloud", source_path);
            ReadOptionalValue(input_json, "target_cloud", target_path);
            config.source_cloud_path = ResolvePathFromBase(source_path, base_dir);
            config.target_cloud_path = ResolvePathFromBase(target_path, base_dir);
        }

        if (root.contains("preprocess") && root["preprocess"].is_object()) {
            // 预处理配置集中管理 TXT 缓存、降采样、去噪和镜像等前置操作。
            const auto& preprocess_json = root["preprocess"];
            ReadOptionalValue(preprocess_json, "enable_txt_cache", config.enable_txt_cache);
            ReadOptionalValue(preprocess_json, "filter_leaf_size", config.filter.leaf_size);

            if (preprocess_json.contains("noise_filter") && preprocess_json["noise_filter"].is_object()) {
                const auto& noise_json = preprocess_json["noise_filter"];
                ReadOptionalValue(noise_json, "enabled", config.noise_filter.enabled);
                ReadOptionalValue(noise_json, "mean_k", config.noise_filter.mean_k);
                ReadOptionalValue(noise_json,
                                  "stddev_mul_thresh",
                                  config.noise_filter.stddev_mul_thresh);
            }

            if (preprocess_json.contains("mirror_axes")) {
                if (!ParseMirrorAxesConfig(preprocess_json["mirror_axes"], config.mirror_axes, error)) {
                    return false;
                }
            }
        }

        if (root.contains("pipeline") && root["pipeline"].is_object()) {
            const auto& pipeline_json = root["pipeline"];
            ReadOptionalValue(pipeline_json, "run_coarse", config.pipeline.run_coarse);
            ReadOptionalValue(pipeline_json, "run_refine", config.pipeline.run_refine);
        }

        if (root.contains("coarse_registration") && root["coarse_registration"].is_object()) {
            if (!ReadRegistrationParams(root["coarse_registration"],
                                        config.coarse_registration,
                                        error)) {
                return false;
            }
        }

        if (root.contains("refine_registration") && root["refine_registration"].is_object()) {
            if (!ReadRegistrationParams(root["refine_registration"],
                                        config.refine_registration,
                                        error)) {
                return false;
            }
        }

        if (root.contains("evaluation") && root["evaluation"].is_object()) {
            const auto& evaluation_json = root["evaluation"];
            ReadOptionalValue(evaluation_json, "enabled", config.evaluation.enabled);
            ReadOptionalValue(evaluation_json, "max_sample_points", config.evaluation.max_sample_points);
            if (evaluation_json.contains("stages")) {
                if (!ParseEvaluationStages(evaluation_json["stages"], config.evaluation, error)) {
                    return false;
                }
            }
        }

        if (root.contains("output") && root["output"].is_object()) {
            const auto& output_json = root["output"];

            std::string result_dir = config.output.result_dir;
            ReadOptionalValue(output_json, "result_dir", result_dir);
            config.output.result_dir = ResolvePathFromBase(result_dir, base_dir);

            ReadOptionalValue(output_json, "transformed_source_cloud", config.output.transformed_source_name);
            ReadOptionalValue(output_json, "result_json", config.output.result_json_name);
            ReadOptionalValue(output_json, "initial_transform_json", config.output.initial_transform_json_name);
        }

        if (root.contains("fusion") && root["fusion"].is_object()) {
            // fusion 模式需要额外解析候选目录、原始数据目录和优化超参数。
            const auto& fusion_json = root["fusion"];

            std::string candidate_result_dir = config.fusion.candidate_result_dir;
            std::string data_dir = config.fusion.data_dir;
            ReadOptionalValue(fusion_json, "candidate_result_dir", candidate_result_dir);
            ReadOptionalValue(fusion_json, "data_dir", data_dir);
            config.fusion.candidate_result_dir = ResolvePathFromBase(candidate_result_dir, base_dir);
            config.fusion.data_dir = ResolvePathFromBase(data_dir, base_dir);

            ReadOptionalValue(fusion_json, "output_json", config.fusion.output_json_name);
            ReadOptionalValue(fusion_json,
                              "max_sample_points_per_pair",
                              config.fusion.max_sample_points_per_pair);
            ReadOptionalValue(fusion_json,
                              "optimization_sample_points_per_pair",
                              config.fusion.optimization_sample_points_per_pair);
            ReadOptionalValue(fusion_json, "trim_ratio", config.fusion.trim_ratio);
            ReadOptionalValue(fusion_json, "max_refine_rmse", config.fusion.max_refine_rmse);
            ReadOptionalValue(fusion_json,
                              "optimization_max_iterations",
                              config.fusion.optimization_max_iterations);
            ReadOptionalValue(fusion_json,
                              "optimization_rotation_step_deg",
                              config.fusion.optimization_rotation_step_deg);
            ReadOptionalValue(fusion_json,
                              "optimization_translation_step",
                              config.fusion.optimization_translation_step);
            ReadOptionalValue(fusion_json,
                              "optimization_min_rotation_step_deg",
                              config.fusion.optimization_min_rotation_step_deg);
            ReadOptionalValue(fusion_json,
                              "optimization_min_translation_step",
                              config.fusion.optimization_min_translation_step);
        }
    } catch (const std::exception& ex) {
        error = "Invalid config field type in " + config_path.string() + ": " + ex.what();
        return false;
    }

    config.run_mode = ToLower(TrimWhitespace(config.run_mode));
    if (config.run_mode.empty()) {
        config.run_mode = "registration";
    }

    if (config.run_mode != "registration" && config.run_mode != "fusion") {
        error = "run_mode must be 'registration' or 'fusion'.";
        return false;
    }

    if (config.noise_filter.mean_k < 3) {
        error = "preprocess.noise_filter.mean_k must be >= 3.";
        return false;
    }

    if (config.noise_filter.stddev_mul_thresh <= 0.0) {
        error = "preprocess.noise_filter.stddev_mul_thresh must be > 0.";
        return false;
    }

    if (config.run_mode == "registration") {
        if (config.source_cloud_path.empty() || config.target_cloud_path.empty()) {
            error = "input.source_cloud and input.target_cloud must not be empty in registration mode.";
            return false;
        }

        if (!config.pipeline.run_coarse && !config.pipeline.run_refine) {
            error = "pipeline.run_coarse and pipeline.run_refine cannot both be false.";
            return false;
        }
    } else {
        if (config.fusion.max_sample_points_per_pair == 0) {
            error = "fusion.max_sample_points_per_pair must be > 0.";
            return false;
        }

        if (config.fusion.optimization_sample_points_per_pair == 0) {
            error = "fusion.optimization_sample_points_per_pair must be > 0.";
            return false;
        }

        if (config.fusion.trim_ratio < 0.0 || config.fusion.trim_ratio >= 0.5) {
            error = "fusion.trim_ratio must be in [0.0, 0.5).";
            return false;
        }

        if (config.fusion.output_json_name.empty()) {
            error = "fusion.output_json must not be empty.";
            return false;
        }
    }

    if (!config.fusion.output_json_name.empty()) {
        const std::filesystem::path output_name_path(config.fusion.output_json_name);
        if (output_name_path.has_parent_path()) {
            error = "fusion.output_json must be a file name only (without directory).";
            return false;
        }
    }

    if (config.fusion.max_refine_rmse > 0.0 &&
        !std::isfinite(config.fusion.max_refine_rmse)) {
        error = "fusion.max_refine_rmse must be finite when enabled.";
        return false;
    }

    if (config.fusion.optimization_max_iterations <= 0) {
        error = "fusion.optimization_max_iterations must be > 0.";
        return false;
    }

    if (config.fusion.optimization_rotation_step_deg <= 0.0 ||
        config.fusion.optimization_translation_step <= 0.0 ||
        config.fusion.optimization_min_rotation_step_deg <= 0.0 ||
        config.fusion.optimization_min_translation_step <= 0.0) {
        error = "fusion optimization step parameters must be > 0.";
        return false;
    }

    if (config.fusion.optimization_min_rotation_step_deg >
            config.fusion.optimization_rotation_step_deg ||
        config.fusion.optimization_min_translation_step >
            config.fusion.optimization_translation_step) {
        error = "fusion optimization min step must be <= initial step.";
        return false;
    }

    return true;
}

/**
 * @brief 将镜像轴配置转换为字符串
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
std::string MirrorAxesToString(const MirrorAxes& axes) {
    std::string text;
    if (axes.x) {
        text += "x";
    }
    if (axes.y) {
        text += "y";
    }
    if (axes.z) {
        text += "z";
    }
    if (text.empty()) {
        text = "none";
    }
    return text;
}

/**
 * @brief 加载 TXT 点云并按需使用缓存
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool LoadTxtPointCloud(const std::string& path,
                       PCLCloudPtr& cloud,
                       std::string& error,
                       bool& loaded_from_cache,
                       bool enable_txt_cache) {
    loaded_from_cache = false;
    const std::filesystem::path txt_path(path);
    const std::filesystem::path cache_path = BuildTxtCachePath(txt_path);

    std::error_code ec;
    if (enable_txt_cache && std::filesystem::exists(cache_path, ec) && !ec) {
        // 只有当缓存文件不比原始 TXT 更旧时，才直接复用缓存结果。
        ec.clear();
        const auto txt_time = std::filesystem::last_write_time(txt_path, ec);
        if (!ec) {
            ec.clear();
            const auto cache_time = std::filesystem::last_write_time(cache_path, ec);
            if (!ec && cache_time >= txt_time) {
                cloud.reset(new PCLCloud);
                if (pcl::io::loadPCDFile(cache_path.string(), *cloud) >= 0 && cloud && !cloud->empty()) {
                    loaded_from_cache = true;
                    return true;
                }
            }
        }
    }

    std::ifstream txt_file(path);
    if (!txt_file.is_open()) {
        error = "Failed to open txt cloud: " + path;
        return false;
    }

    std::vector<char> io_buffer(4 * 1024 * 1024);
    txt_file.rdbuf()->pubsetbuf(io_buffer.data(), static_cast<std::streamsize>(io_buffer.size()));

    cloud.reset(new PCLCloud);
    const std::uintmax_t file_size_bytes = std::filesystem::file_size(txt_path, ec);
    if (!ec && file_size_bytes > 0) {
        // 按经验值预估点数，减少大文件读取过程中的动态扩容次数。
        const std::size_t estimated_points = static_cast<std::size_t>(file_size_bytes / 28);
        if (estimated_points > 0) {
            cloud->points.reserve(estimated_points);
        }
    }

    std::string line;
    line.reserve(128);
    while (std::getline(txt_file, line)) {
        pcl::PointXYZ point;
        if (TryParseTxtPointLine(line, point)) {
            cloud->points.push_back(point);
        }
    }

    if (cloud->points.empty()) {
        error = "TXT cloud contains no valid xyz points: " + path;
        return false;
    }

    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    if (enable_txt_cache) {
        pcl::io::savePCDFileBinary(cache_path.string(), *cloud);
    }

    return true;
}

/**
 * @brief 加载点云文件
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool LoadPointCloud(const std::string& path,
                    PCLCloudPtr& cloud,
                    std::string& error,
                    bool enable_txt_cache) {
    const auto t0 = std::chrono::steady_clock::now();
    std::error_code ec;
    const std::filesystem::path p(path);
    if (!std::filesystem::exists(p, ec) || ec) {
        error = "File does not exist: " + path;
        return false;
    }

    if (!std::filesystem::is_regular_file(p, ec) || ec) {
        error = "Not a regular file: " + path;
        return false;
    }

    cloud.reset(new PCLCloud);
    const std::string ext = ToLower(p.extension().string());

    if (ext == ".txt") {
        bool loaded_from_cache = false;
        const bool ok = LoadTxtPointCloud(path, cloud, error, loaded_from_cache, enable_txt_cache);
        if (ok) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "Loaded cloud: " << p.filename().string()
                      << " points=" << cloud->size()
                      << " time(ms)=" << ms
                      << " source=" << (loaded_from_cache ? "txt-cache" : "txt-parse")
                      << std::endl;
        }
        return ok;
    }

    int ret = -1;
    if (ext == ".ply") {
        ret = pcl::io::loadPLYFile(path, *cloud);
    } else if (ext == ".pcd") {
        ret = pcl::io::loadPCDFile(path, *cloud);
    } else {
        error = "Unsupported extension: " + ext;
        return false;
    }

    if (ret < 0 || !cloud || cloud->empty()) {
        error = "Failed to read cloud: " + path;
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Loaded cloud: " << p.filename().string()
              << " points=" << cloud->size()
              << " time(ms)=" << ms
              << " source=" << ext.substr(1)
              << std::endl;

    return true;
}

/**
 * @brief 对点云执行体素降采样
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
PCLCloudPtr VoxelFilter(const PCLCloudPtr& input, float leaf_size) {
    if (!input || input->empty()) {
        return input;
    }

    PCLCloudPtr filtered(new PCLCloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(input);
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.filter(*filtered);

    if (!filtered || filtered->empty()) {
        return input;
    }
    return filtered;
}

/**
 * @brief 对点云执行统计离群点滤波
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
PCLCloudPtr StatisticalOutlierFilter(const PCLCloudPtr& input,
                                    const NoiseFilterConfig& noise_filter) {
    if (!input || input->empty() || !noise_filter.enabled) {
        return input;
    }

    PCLCloudPtr filtered(new PCLCloud);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(input);
    sor.setMeanK(noise_filter.mean_k);
    sor.setStddevMulThresh(noise_filter.stddev_mul_thresh);
    sor.filter(*filtered);

    if (!filtered || filtered->empty()) {
        return input;
    }
    return filtered;
}

/** @brief 融合候选结果及其元数据。 */
struct FusionCandidate {
    std::filesystem::path json_path;
    std::filesystem::path log_path;
    std::string index_token;
    double refine_rmse = std::numeric_limits<double>::quiet_NaN();
    double rotation[3][3] = {{0.0}};
    double translation[3] = {0.0, 0.0, 0.0};
};

/**
 * @brief 从日志中提取 refine 阶段 RMSE
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ParseRefineRmseFromLog(const std::filesystem::path& log_path,
                            double& refine_rmse) {
    refine_rmse = std::numeric_limits<double>::quiet_NaN();

    std::ifstream stream(log_path.string(), std::ios::in);
    if (!stream.is_open()) {
        return false;
    }

    const std::regex refine_regex(
        R"(\[Stage-2\s+Refine\s+Evaluation\].*?rmse\s*=\s*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?))",
        std::regex::icase);

    std::string line;
    std::smatch match;
    bool found = false;
    while (std::getline(stream, line)) {
        if (std::regex_search(line, match, refine_regex) && match.size() >= 2) {
            try {
                refine_rmse = std::stod(match[1].str());
                found = true;
            } catch (...) {
                continue;
            }
        }
    }

    return found && std::isfinite(refine_rmse);
}

/**
 * @brief 从结果 JSON 加载 source 到 target 的刚体变换
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool LoadTransformSourceToTargetJsonFile(const std::filesystem::path& json_path,
                                         FusionCandidate& candidate,
                                         std::string& error) {
    std::ifstream input(json_path.string(), std::ios::in);
    if (!input.is_open()) {
        error = "Failed to open json file: " + json_path.string();
        return false;
    }

    const nlohmann::json parsed = nlohmann::json::parse(input, nullptr, false);
    if (parsed.is_discarded()) {
        error = "Failed to parse json file: " + json_path.string();
        return false;
    }

    if (!parsed.contains("transform_source_to_target") ||
        !parsed["transform_source_to_target"].is_object()) {
        error = "json missing transform_source_to_target: " + json_path.string();
        return false;
    }

    const auto& tf = parsed["transform_source_to_target"];
    if (!tf.contains("R") || !tf["R"].is_array() || tf["R"].size() != 3) {
        error = "json missing valid R matrix: " + json_path.string();
        return false;
    }
    if (!tf.contains("t") || !tf["t"].is_array() || tf["t"].size() != 3) {
        error = "json missing valid t vector: " + json_path.string();
        return false;
    }

    for (int row = 0; row < 3; ++row) {
        const auto& rrow = tf["R"][row];
        if (!rrow.is_array() || rrow.size() != 3) {
            error = "json R row format invalid: " + json_path.string();
            return false;
        }
        for (int col = 0; col < 3; ++col) {
            if (!rrow[col].is_number()) {
                error = "json R element is not numeric: " + json_path.string();
                return false;
            }
            candidate.rotation[row][col] = rrow[col].get<double>();
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (!tf["t"][i].is_number()) {
            error = "json t element is not numeric: " + json_path.string();
            return false;
        }
        candidate.translation[i] = tf["t"][i].get<double>();
    }

    candidate.json_path = json_path;
    candidate.log_path = json_path.parent_path() / "log.txt";
    candidate.index_token = ExtractDigits(json_path.parent_path().filename().string());
    return true;
}

/**
 * @brief 从日志中提取 source 与 target 路径
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ParseSourceTargetFromLog(const std::filesystem::path& log_path,
                              std::string& source_cloud_path,
                              std::string& target_cloud_path) {
    source_cloud_path.clear();
    target_cloud_path.clear();

    std::ifstream log_stream(log_path.string(), std::ios::in);
    if (!log_stream.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(log_stream, line)) {
        if (StartsWithCaseInsensitive(line, "Source cloud:")) {
            source_cloud_path = TrimWhitespace(line.substr(std::string("Source cloud:").size()));
        } else if (StartsWithCaseInsensitive(line, "Target cloud:")) {
            target_cloud_path = TrimWhitespace(line.substr(std::string("Target cloud:").size()));
        }
    }
    return !source_cloud_path.empty() && !target_cloud_path.empty();
}

/**
 * @brief 按编号标识匹配点云文件
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool MatchCloudFileByToken(const std::filesystem::path& data_dir,
                           const std::vector<std::string>& prefixes,
                           const std::string& token,
                           std::filesystem::path& output) {
    if (token.empty()) {
        return false;
    }

    const std::string normalized_target_token = NormalizeNumericToken(token);

    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir, ec)) {
        if (ec) {
            return false;
        }

        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string ext = ToLower(entry.path().extension().string());
        if (!IsSupportedCloudExtension(ext)) {
            continue;
        }

        const std::string file_name = ToLower(entry.path().filename().string());
        for (const std::string& prefix : prefixes) {
            std::string file_token;
            if (!ExtractNumericTokenAfterPrefix(file_name, ToLower(prefix), file_token)) {
                continue;
            }

            if (NormalizeNumericToken(file_token) == normalized_target_token) {
                output = entry.path();
                return true;
            }
        }
    }

    return false;
}

/**
 * @brief 为融合候选解析对应点云对
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ResolveCandidateCloudPair(const FusionCandidate& candidate,
                               const FusionConfig& fusion,
                               std::filesystem::path& source_path,
                               std::filesystem::path& target_path,
                               std::string& error) {
    source_path.clear();
    target_path.clear();

    const std::filesystem::path data_dir(fusion.data_dir);
    if (!std::filesystem::exists(data_dir)) {
        error = "fusion.data_dir does not exist: " + data_dir.string();
        return false;
    }

    MatchCloudFileByToken(data_dir, {"left"}, candidate.index_token, source_path);
    MatchCloudFileByToken(data_dir, {"right", "rigth"}, candidate.index_token, target_path);
    if (!source_path.empty() && !target_path.empty()) {
        return true;
    }

    std::string source_from_log;
    std::string target_from_log;
    if (ParseSourceTargetFromLog(candidate.log_path, source_from_log, target_from_log)) {
        if (source_path.empty()) {
            source_path = source_from_log;
        }
        if (target_path.empty()) {
            target_path = target_from_log;
        }
    }

    if (source_path.empty() || target_path.empty()) {
        error = "Failed to map candidate to cloud pair: " + candidate.json_path.string();
        return false;
    }

    return true;
}

/**
 * @brief 计算单对点云在给定位姿下的 RMSE
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ComputeTransformPairRmse(const PCLCloudPtr& source_cloud,
                              const PCLCloudPtr& target_cloud,
                              const double rotation[3][3],
                              const double translation[3],
                              std::size_t max_sample_points,
                              double trim_ratio,
                              double& rmse,
                              std::string& error) {
    rmse = std::numeric_limits<double>::infinity();

    const PCLCloudPtr source_finite = MakeFiniteCloud(source_cloud);
    const PCLCloudPtr target_finite = MakeFiniteCloud(target_cloud);
    if (!source_finite || source_finite->empty()) {
        error = "Fusion source cloud is empty after removing invalid points.";
        return false;
    }
    if (!target_finite || target_finite->empty()) {
        error = "Fusion target cloud is empty after removing invalid points.";
        return false;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target_finite);

    std::size_t stride = 1;
    const std::size_t total = source_finite->size();
    if (max_sample_points > 0 && total > max_sample_points) {
        stride = static_cast<std::size_t>(
            std::ceil(static_cast<double>(total) / static_cast<double>(max_sample_points)));
    }

    std::vector<double> squared_distances;
    squared_distances.reserve((total + stride - 1) / stride);

    std::vector<int> indices(1);
    std::vector<float> nn_sq(1);
    for (std::size_t i = 0; i < total; i += stride) {
        const pcl::PointXYZ& p = source_finite->points[i];

        pcl::PointXYZ transformed;
        transformed.x = static_cast<float>(rotation[0][0] * p.x + rotation[0][1] * p.y + rotation[0][2] * p.z + translation[0]);
        transformed.y = static_cast<float>(rotation[1][0] * p.x + rotation[1][1] * p.y + rotation[1][2] * p.z + translation[1]);
        transformed.z = static_cast<float>(rotation[2][0] * p.x + rotation[2][1] * p.y + rotation[2][2] * p.z + translation[2]);

        if (kdtree.nearestKSearch(transformed, 1, indices, nn_sq) <= 0) {
            continue;
        }
        squared_distances.push_back(std::max(0.0, static_cast<double>(nn_sq[0])));
    }

    if (squared_distances.empty()) {
        error = "Fusion nearest-neighbor matching returned zero samples.";
        return false;
    }

    if (trim_ratio > 0.0) {
        std::sort(squared_distances.begin(), squared_distances.end());
        std::size_t keep_count = static_cast<std::size_t>(
            std::floor(static_cast<double>(squared_distances.size()) * (1.0 - trim_ratio)));
        keep_count = std::max<std::size_t>(1, keep_count);
        squared_distances.resize(keep_count);
    }

    double sum_sq = 0.0;
    for (double sq : squared_distances) {
        sum_sq += sq;
    }
    rmse = std::sqrt(sum_sq / static_cast<double>(squared_distances.size()));
    return true;
}

/** @brief 融合优化中使用的预处理点云对。 */
struct FusionPairData {
    std::filesystem::path source_path;
    std::filesystem::path target_path;
    PCLCloudPtr source_finite;
    PCLCloudPtr target_finite;
    std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> target_kdtree;
    MirrorAxes mirror_axes;
};

/**
 * @brief 计算 3x3 矩阵乘法
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void MultiplyMat3(const double a[3][3],
                  const double b[3][3],
                  double out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = a[i][0] * b[0][j] +
                        a[i][1] * b[1][j] +
                        a[i][2] * b[2][j];
        }
    }
}

/**
 * @brief 复制 3x3 矩阵
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void CopyMat3(const double src[3][3],
              double dst[3][3]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            dst[r][c] = src[r][c];
        }
    }
}

/**
 * @brief 计算 3x3 矩阵行列式
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
double DeterminantMat3(const double matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

/**
 * @brief 求解 6x6 矩阵逆
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool InvertMat6(const double input[6][6],
                double inverse[6][6]) {
    double augmented[6][12] = {{0.0}};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            augmented[r][c] = input[r][c];
        }
        augmented[r][r + 6] = 1.0;
    }

    for (int col = 0; col < 6; ++col) {
        int pivot_row = col;
        double pivot_abs = std::fabs(augmented[pivot_row][col]);
        for (int r = col + 1; r < 6; ++r) {
            const double candidate_abs = std::fabs(augmented[r][col]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = r;
            }
        }

        if (pivot_abs < 1e-15) {
            return false;
        }

        if (pivot_row != col) {
            for (int c = 0; c < 12; ++c) {
                std::swap(augmented[col][c], augmented[pivot_row][c]);
            }
        }

        const double pivot = augmented[col][col];
        for (int c = 0; c < 12; ++c) {
            augmented[col][c] /= pivot;
        }

        for (int r = 0; r < 6; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = augmented[r][col];
            if (std::fabs(factor) < 1e-18) {
                continue;
            }
            for (int c = 0; c < 12; ++c) {
                augmented[r][c] -= factor * augmented[col][c];
            }
        }
    }

    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            inverse[r][c] = augmented[r][c + 6];
        }
    }
    return true;
}

/**
 * @brief 将轴角向量转换为旋转矩阵
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void RodriguesToRotation(const double w[3],
                         double rotation[3][3]) {
    const double theta = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    rotation[0][0] = 1.0; rotation[0][1] = 0.0; rotation[0][2] = 0.0;
    rotation[1][0] = 0.0; rotation[1][1] = 1.0; rotation[1][2] = 0.0;
    rotation[2][0] = 0.0; rotation[2][1] = 0.0; rotation[2][2] = 1.0;
    if (theta < 1e-12) {
        return;
    }

    const double kx = w[0] / theta;
    const double ky = w[1] / theta;
    const double kz = w[2] / theta;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double v = 1.0 - c;

    rotation[0][0] = c + kx * kx * v;
    rotation[0][1] = kx * ky * v - kz * s;
    rotation[0][2] = kx * kz * v + ky * s;
    rotation[1][0] = ky * kx * v + kz * s;
    rotation[1][1] = c + ky * ky * v;
    rotation[1][2] = ky * kz * v - kx * s;
    rotation[2][0] = kz * kx * v - ky * s;
    rotation[2][1] = kz * ky * v + kx * s;
    rotation[2][2] = c + kz * kz * v;
}

/**
 * @brief 将旋转增量作用到基础旋转矩阵
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void ComposeRotationDelta(const double base_rotation[3][3],
                          const double delta_axis_angle[3],
                          double output_rotation[3][3]) {
    double delta_rotation[3][3] = {{0.0}};
    RodriguesToRotation(delta_axis_angle, delta_rotation);
    MultiplyMat3(delta_rotation, base_rotation, output_rotation);
}

/**
 * @brief 基于预处理点云对计算 RMSE
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ComputeTransformPairRmsePrepared(const FusionPairData& pair,
                                      const double rotation[3][3],
                                      const double translation[3],
                                      std::size_t max_sample_points,
                                      double trim_ratio,
                                      double& rmse,
                                      std::string& error) {
    rmse = std::numeric_limits<double>::infinity();
    if (!pair.source_finite || pair.source_finite->empty()) {
        error = "Prepared source cloud is empty.";
        return false;
    }
    if (!pair.target_finite || pair.target_finite->empty() || !pair.target_kdtree) {
        error = "Prepared target cloud/kdtree is invalid.";
        return false;
    }

    std::size_t stride = 1;
    const std::size_t total = pair.source_finite->size();
    if (max_sample_points > 0 && total > max_sample_points) {
        stride = static_cast<std::size_t>(
            std::ceil(static_cast<double>(total) / static_cast<double>(max_sample_points)));
    }

    std::vector<double> squared_distances;
    squared_distances.reserve((total + stride - 1) / stride);

    std::vector<int> indices(1);
    std::vector<float> nn_sq(1);
    for (std::size_t i = 0; i < total; i += stride) {
        const pcl::PointXYZ& p = pair.source_finite->points[i];

        pcl::PointXYZ transformed;
        transformed.x = static_cast<float>(rotation[0][0] * p.x + rotation[0][1] * p.y + rotation[0][2] * p.z + translation[0]);
        transformed.y = static_cast<float>(rotation[1][0] * p.x + rotation[1][1] * p.y + rotation[1][2] * p.z + translation[1]);
        transformed.z = static_cast<float>(rotation[2][0] * p.x + rotation[2][1] * p.y + rotation[2][2] * p.z + translation[2]);

        if (pair.target_kdtree->nearestKSearch(transformed, 1, indices, nn_sq) <= 0) {
            continue;
        }

        squared_distances.push_back(std::max(0.0, static_cast<double>(nn_sq[0])));
    }

    if (squared_distances.empty()) {
        error = "Prepared nearest-neighbor matching returned zero samples.";
        return false;
    }

    std::size_t keep_count = squared_distances.size();
    if (trim_ratio > 0.0) {
        keep_count = static_cast<std::size_t>(
            std::floor(static_cast<double>(squared_distances.size()) * (1.0 - trim_ratio)));
        keep_count = std::max<std::size_t>(1, keep_count);
        if (keep_count < squared_distances.size()) {
            std::nth_element(squared_distances.begin(),
                             squared_distances.begin() + static_cast<std::ptrdiff_t>(keep_count),
                             squared_distances.end());
        }
    }

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < keep_count; ++i) {
        sum_sq += squared_distances[i];
    }
    rmse = std::sqrt(sum_sq / static_cast<double>(keep_count));
    return true;
}

/**
 * @brief 计算融合位姿在全部点云对上的全局 RMSE
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ComputeGlobalRmseForTransform(const std::vector<FusionPairData>& pairs,
                                   const FusionConfig& fusion,
                                   const double rotation[3][3],
                                   const double translation[3],
                                   std::size_t objective_sample_points,
                                   double& global_rmse,
                                   std::string& error) {
    if (pairs.empty()) {
        error = "No fusion pairs available for global objective.";
        return false;
    }

    double sum_rmse = 0.0;
    std::size_t count = 0;
    for (const auto& pair : pairs) {
        double pair_rmse = 0.0;
        std::string pair_error;
        if (!ComputeTransformPairRmsePrepared(pair,
                                              rotation,
                                              translation,
                                              objective_sample_points,
                                              fusion.trim_ratio,
                                              pair_rmse,
                                              pair_error)) {
            error = pair_error;
            return false;
        }
        sum_rmse += pair_rmse;
        ++count;
    }

    if (count == 0) {
        error = "No valid fusion pair RMSE values.";
        return false;
    }

    global_rmse = sum_rmse / static_cast<double>(count);
    return true;
}

/**
 * @brief 计算候选变换集合的平均初始位姿
 * @param candidates 候选刚体变换集合，每项包含旋转矩阵与平移向量
 * @param avg_rotation 输出平均旋转矩阵，结果会被投影到合法 SO(3)
 * @param avg_translation 输出平均平移向量
 * @return 无返回值
 *
 * @description 工作流执行步骤：
 *   1. 遍历全部候选，累加每个候选的旋转矩阵与平移向量
 *   2. 对旋转矩阵累加结果执行 SVD 分解
 *   3. 使用 U * V^T 将均值旋转投影回 SO(3)
 *   4. 若行列式小于 0，则翻转最后一列以保证右手系旋转
 *   5. 对平移向量取算术平均，作为融合优化的初始平移
 */
void AverageCandidateTransformsSvd(const std::vector<FusionCandidate>& candidates,
                                   double avg_rotation[3][3],
                                   double avg_translation[3]) {
    Eigen::Matrix3d sum_rotation = Eigen::Matrix3d::Zero();
    Eigen::Vector3d sum_translation = Eigen::Vector3d::Zero();

    for (const auto& candidate : candidates) {
        Eigen::Matrix3d current_rotation;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                current_rotation(r, c) = candidate.rotation[r][c];
            }
            sum_translation(r) += candidate.translation[r];
        }
        sum_rotation += current_rotation;
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(sum_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d projected_rotation = svd.matrixU() * svd.matrixV().transpose();
    if (projected_rotation.determinant() < 0.0) {
        Eigen::Matrix3d corrected_v = svd.matrixV();
        corrected_v.col(2) *= -1.0;
        projected_rotation = svd.matrixU() * corrected_v.transpose();
    }

    const Eigen::Vector3d mean_translation =
        sum_translation / static_cast<double>(std::max<std::size_t>(1, candidates.size()));

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            avg_rotation[r][c] = projected_rotation(r, c);
        }
        avg_translation[r] = mean_translation(r);
    }
}

/**
 * @brief 计算当前融合目标在 6 自由度参数上的有限差分梯度
 * @param pairs 预处理后的融合点云对集合
 * @param fusion 融合配置参数
 * @param base_rotation 当前旋转矩阵
 * @param base_translation 当前平移向量
 * @param objective_sample_points 目标函数采样点数上限
 * @param base_rmse 当前位姿下的全局 RMSE
 * @param gradient 输出 6 维梯度，前三维为旋转扰动，后三维为平移扰动
 * @param error 失败时输出错误信息
 * @return true 表示梯度估计成功，false 表示估计失败
 *
 * @description 工作流执行步骤：
 *   1. 依据最小步长阈值为旋转与平移分别构造有限差分扰动量
 *   2. 逐个维度对当前位姿施加微小正向扰动
 *   3. 重新计算扰动后位姿的全局截尾 RMSE
 *   4. 使用前向差分公式 (f(x+eps)-f(x))/eps 估计该维度梯度
 *   5. 输出完整 6 维梯度，供后续 Gauss-Newton 更新使用
 */
bool ComputeFiniteDifferenceGradient(const std::vector<FusionPairData>& pairs,
                                     const FusionConfig& fusion,
                                     const double base_rotation[3][3],
                                     const double base_translation[3],
                                     std::size_t objective_sample_points,
                                     double base_rmse,
                                     double gradient[6],
                                     std::string& error) {
    const double rotation_eps = std::max(
        fusion.optimization_min_rotation_step_deg * (3.14159265358979323846 / 180.0), 1e-6);
    const double translation_eps = std::max(fusion.optimization_min_translation_step, 1e-6);

    for (int dim = 0; dim < 6; ++dim) {
        // 前三维使用 Rodrigues 小角度扰动旋转，后三维直接扰动平移。
        double trial_rotation[3][3] = {{0.0}};
        double trial_translation[3] = {
            base_translation[0],
            base_translation[1],
            base_translation[2]
        };

        if (dim < 3) {
            double delta_axis_angle[3] = {0.0, 0.0, 0.0};
            delta_axis_angle[dim] = rotation_eps;
            ComposeRotationDelta(base_rotation, delta_axis_angle, trial_rotation);
        } else {
            CopyMat3(base_rotation, trial_rotation);
            trial_translation[dim - 3] += translation_eps;
        }

        double trial_rmse = std::numeric_limits<double>::infinity();
        std::string trial_error;
        if (!ComputeGlobalRmseForTransform(pairs,
                                           fusion,
                                           trial_rotation,
                                           trial_translation,
                                           objective_sample_points,
                                           trial_rmse,
                                           trial_error)) {
            error = trial_error;
            return false;
        }

        const double eps = dim < 3 ? rotation_eps : translation_eps;
        gradient[dim] = (trial_rmse - base_rmse) / eps;
    }

    return true;
}

/**
 * @brief 使用有限差分 Gauss-Newton 优化融合位姿
 * @param pairs 预处理后的融合点云对集合
 * @param fusion 融合配置参数
 * @param init_rotation 初始旋转矩阵
 * @param init_translation 初始平移向量
 * @param objective_sample_points 优化阶段每对点云的采样点数上限
 * @param optimized_rotation 输出优化后的旋转矩阵
 * @param optimized_translation 输出优化后的平移向量
 * @param optimized_global_rmse 输出优化后的全局 RMSE
 * @param error 失败时输出错误信息
 * @return true 表示优化成功，false 表示优化失败
 *
 * @description 工作流执行步骤：
 *   1. 以 SVD 旋转均值和平移均值作为初始位姿，计算初始全局 RMSE
 *   2. 在每轮迭代中通过有限差分估计 6 自由度梯度
 *   3. 使用梯度外积加阻尼项构造近似 Hessian，并求解增量方向
 *   4. 采用回溯线搜索逐步缩小步长，直到找到可降低全局 RMSE 的更新
 *   5. 若本轮无可接受更新，则增大阻尼；当步长已足够小或达到最大迭代次数时停止
 */
bool OptimizeTransformGaussNewton(const std::vector<FusionPairData>& pairs,
                                  const FusionConfig& fusion,
                                  const double init_rotation[3][3],
                                  const double init_translation[3],
                                  std::size_t objective_sample_points,
                                  double optimized_rotation[3][3],
                                  double optimized_translation[3],
                                  double& optimized_global_rmse,
                                  std::string& error) {
    const double kDegToRad = 3.14159265358979323846 / 180.0;
    CopyMat3(init_rotation, optimized_rotation);
    for (int i = 0; i < 3; ++i) {
        optimized_translation[i] = init_translation[i];
    }

    if (!ComputeGlobalRmseForTransform(pairs,
                                       fusion,
                                       optimized_rotation,
                                       optimized_translation,
                                       objective_sample_points,
                                       optimized_global_rmse,
                                       error)) {
        return false;
    }

    double damping = 1e-3;
    const double min_rotation_step = fusion.optimization_min_rotation_step_deg * kDegToRad;
    const double min_translation_step = fusion.optimization_min_translation_step;

    for (int iter = 0; iter < fusion.optimization_max_iterations; ++iter) {
        // 使用有限差分梯度构造带阻尼项的近似 Hessian，提升数值稳定性。
        double gradient[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (!ComputeFiniteDifferenceGradient(pairs,
                                             fusion,
                                             optimized_rotation,
                                             optimized_translation,
                                             objective_sample_points,
                                             optimized_global_rmse,
                                             gradient,
                                             error)) {
            return false;
        }

        double hessian[6][6] = {{0.0}};
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                hessian[r][c] = gradient[r] * gradient[c];
            }
            hessian[r][r] += damping;
        }

        double inverse_hessian[6][6] = {{0.0}};
        if (!InvertMat6(hessian, inverse_hessian)) {
            damping *= 10.0;
            continue;
        }

        double delta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                delta[r] -= inverse_hessian[r][c] * gradient[c];
            }
        }

        double step_scale = 1.0;
        bool accepted = false;
        while (step_scale >= 1e-3) {
            double trial_rotation[3][3] = {{0.0}};
            double trial_translation[3] = {
                optimized_translation[0],
                optimized_translation[1],
                optimized_translation[2]
            };

            double delta_axis_angle[3] = {
                delta[0] * step_scale,
                delta[1] * step_scale,
                delta[2] * step_scale
            };
            ComposeRotationDelta(optimized_rotation, delta_axis_angle, trial_rotation);
            for (int i = 0; i < 3; ++i) {
                trial_translation[i] += delta[i + 3] * step_scale;
            }

            double trial_rmse = std::numeric_limits<double>::infinity();
            std::string trial_error;
            if (ComputeGlobalRmseForTransform(pairs,
                                              fusion,
                                              trial_rotation,
                                              trial_translation,
                                              objective_sample_points,
                                              trial_rmse,
                                              trial_error) &&
                trial_rmse + 1e-12 < optimized_global_rmse) {
                CopyMat3(trial_rotation, optimized_rotation);
                for (int i = 0; i < 3; ++i) {
                    optimized_translation[i] = trial_translation[i];
                }
                optimized_global_rmse = trial_rmse;
                damping = std::max(1e-6, damping * 0.5);
                accepted = true;
                break;
            }

            step_scale *= 0.5;
        }

        const double rotation_step_norm =
            std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
        const double translation_step_norm =
            std::sqrt(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);

        if (!accepted) {
            damping *= 10.0;
            if (rotation_step_norm < min_rotation_step &&
                translation_step_norm < min_translation_step) {
                break;
            }
        }

        if ((iter + 1) % 5 == 0 || iter == fusion.optimization_max_iterations - 1) {
            std::cout << "[Fusion] gauss-newton iter=" << (iter + 1)
                      << " best_rmse=" << std::fixed << std::setprecision(6)
                      << optimized_global_rmse
                      << " damping=" << damping
                      << " step_rot=" << rotation_step_norm
                      << " step_trans=" << translation_step_norm
                      << std::endl;
        }
    }

    if (std::fabs(DeterminantMat3(optimized_rotation) - 1.0) > 1e-3) {
        error = "Optimized rotation determinant deviates from SO(3).";
        return false;
    }

    return true;
}

/**
 * @brief 执行 fusion 模式的多位姿融合求解
 * @param config 应用配置，包含候选目录、数据目录、优化参数和输出参数
 * @param error 失败时输出错误信息
 * @return true 表示融合成功，false 表示融合失败
 *
 * @description 工作流执行步骤：
 *   1. 递归扫描候选结果目录，加载多个 registration_result.json
 *   2. 按需读取各候选 log.txt 中的 refine RMSE，并过滤低质量候选
 *   3. 根据候选编号或日志中的路径信息，映射并加载对应 source/target 点云
 *   4. 计算每个候选在全局目标上的初始评分，并基于全部候选构造 SVD 平均初值
 *   5. 调用有限差分 Gauss-Newton 优化统一的 source->target 刚体变换，并将结果写入融合输出 JSON
 */
bool RunFusionMode(const AppConfig& config,
                   std::string& error) {
    const std::filesystem::path candidate_root(config.fusion.candidate_result_dir);
    if (!std::filesystem::exists(candidate_root)) {
        error = "fusion.candidate_result_dir does not exist: " + candidate_root.string();
        return false;
    }

    std::vector<FusionCandidate> candidates;
    std::size_t discovered_candidate_files = 0;
    std::size_t skipped_root_result = 0;
    std::size_t skipped_invalid_json = 0;
    std::size_t skipped_missing_refine_rmse = 0;
    std::size_t skipped_refine_rmse_threshold = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(candidate_root, ec)) {
        // 第一阶段只做候选扫描与过滤，先把明显无效项剔除。
        if (ec) {
            error = "Failed to iterate candidate result directory: " + candidate_root.string();
            return false;
        }

        if (!entry.is_regular_file()) {
            continue;
        }

        if (entry.path().filename().string() != config.output.result_json_name) {
            continue;
        }

        ++discovered_candidate_files;

        // Skip candidate_root/registration_result.json to avoid mixing current run output as an input candidate.
        if (entry.path().parent_path() == candidate_root) {
            ++skipped_root_result;
            std::cerr << "[Fusion] Skip root-level candidate json: "
                      << entry.path().string() << std::endl;
            continue;
        }

        FusionCandidate candidate;
        std::string load_error;
        if (!LoadTransformSourceToTargetJsonFile(entry.path(), candidate, load_error)) {
            ++skipped_invalid_json;
            std::cerr << "[Fusion] Skip invalid candidate: " << load_error << std::endl;
            continue;
        }

        if (config.fusion.max_refine_rmse > 0.0) {
            double refine_rmse = std::numeric_limits<double>::quiet_NaN();
            const bool has_refine_rmse = ParseRefineRmseFromLog(candidate.log_path, refine_rmse);
            if (!has_refine_rmse) {
                ++skipped_missing_refine_rmse;
                std::cerr << "[Fusion] Skip candidate (missing refine rmse): "
                          << candidate.json_path.string() << std::endl;
                continue;
            }
            candidate.refine_rmse = refine_rmse;
            if (refine_rmse > config.fusion.max_refine_rmse) {
                ++skipped_refine_rmse_threshold;
                std::cerr << "[Fusion] Skip candidate (refine rmse=" << refine_rmse
                          << " > threshold=" << config.fusion.max_refine_rmse << "): "
                          << candidate.json_path.string() << std::endl;
                continue;
            }
        }

        candidates.push_back(candidate);
    }

    if (candidates.empty()) {
        error = "No valid registration_result.json candidates after filtering.";
        return false;
    }

    std::cout << "[Fusion] candidate scan summary: discovered=" << discovered_candidate_files
              << " accepted=" << candidates.size()
              << " skipped_root=" << skipped_root_result
              << " skipped_invalid_json=" << skipped_invalid_json
              << " skipped_missing_refine_rmse=" << skipped_missing_refine_rmse
              << " skipped_refine_threshold=" << skipped_refine_rmse_threshold
              << std::endl;

    std::unordered_map<std::string, PCLCloudPtr> cloud_cache;
    auto load_cloud_cached = [&](const std::filesystem::path& cloud_path,
                                 PCLCloudPtr& cloud,
                                 std::string& load_error) -> bool {
        // 多个候选可能引用同一份点云文件，这里按绝对路径缓存复用。
        const std::string key = std::filesystem::absolute(cloud_path).lexically_normal().string();
        auto it = cloud_cache.find(key);
        if (it != cloud_cache.end()) {
            cloud = it->second;
            return true;
        }

        if (!LoadPointCloud(key, cloud, load_error, true)) {
            return false;
        }

        cloud_cache[key] = cloud;
        return true;
    };

    std::vector<FusionPairData> fusion_pairs;
    fusion_pairs.reserve(candidates.size());
    std::size_t skipped_unmapped_pairs = 0;
    std::size_t skipped_load_fail_pairs = 0;
    std::size_t prepared_pair_index = 0;
    const std::size_t fusion_pair_cloud_limit =
        std::max<std::size_t>(200000, config.fusion.max_sample_points_per_pair * 8);
    // 第二阶段把候选变换映射回真实点云对，并提前完成有限点过滤与 KD-tree 构建。
    for (const auto& pair_candidate : candidates) {
        ++prepared_pair_index;
        std::filesystem::path source_path;
        std::filesystem::path target_path;
        std::string map_error;
        if (!ResolveCandidateCloudPair(pair_candidate,
                                       config.fusion,
                                       source_path,
                                       target_path,
                                       map_error)) {
            ++skipped_unmapped_pairs;
            std::cerr << "[Fusion] Skip unmapped candidate: " << map_error << std::endl;
            continue;
        }

        PCLCloudPtr source_cloud;
        PCLCloudPtr target_cloud;
        std::string load_error;
        if (!load_cloud_cached(source_path, source_cloud, load_error) ||
            !load_cloud_cached(target_path, target_cloud, load_error)) {
            ++skipped_load_fail_pairs;
            std::cerr << "[Fusion] Skip candidate due to load failure: "
                      << load_error << std::endl;
            continue;
        }

        FusionPairData pair;
        pair.source_path = source_path;
        pair.target_path = target_path;
        pair.mirror_axes = config.mirror_axes;
        const PCLCloudPtr source_finite_full = MakeFiniteCloud(source_cloud);
        const PCLCloudPtr target_finite_full = MakeFiniteCloud(target_cloud);
        pair.source_finite = LimitCloudPoints(source_finite_full, fusion_pair_cloud_limit);
        pair.target_finite = LimitCloudPoints(target_finite_full, fusion_pair_cloud_limit);
        if (!pair.source_finite || pair.source_finite->empty() ||
            !pair.target_finite || pair.target_finite->empty()) {
            std::cerr << "[Fusion] Skip invalid pair due to empty finite cloud: "
                      << source_path.string() << " | " << target_path.string() << std::endl;
            continue;
        }
        // Apply mirror to the working source copy so the stored points are already mirrored.
        if (pair.mirror_axes.AnyEnabled()) {
            PCLCloudPtr mirrored_source(new PCLCloud(*pair.source_finite));
            MirrorCloudByNegatingX(mirrored_source, pair.mirror_axes, false);
            pair.source_finite = mirrored_source;
        }
        pair.target_kdtree = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
        pair.target_kdtree->setInputCloud(pair.target_finite);

        std::cout << "[Fusion] prepared pair " << prepared_pair_index << "/" << candidates.size()
                  << " source=" << source_path.filename().string()
                  << " (" << pair.source_finite->size() << " pts"
                  << ", full=" << (source_finite_full ? source_finite_full->size() : 0) << ")"
                  << " target=" << target_path.filename().string()
                  << " (" << pair.target_finite->size() << " pts"
                  << ", full=" << (target_finite_full ? target_finite_full->size() : 0) << ")"
                  << std::endl;
        fusion_pairs.push_back(pair);
    }

    if (fusion_pairs.empty()) {
        error = "No valid fusion pairs were built from candidates.";
        return false;
    }

    std::cout << "[Fusion] pair build summary: prepared=" << fusion_pairs.size()
              << " skipped_unmapped=" << skipped_unmapped_pairs
              << " skipped_load_fail=" << skipped_load_fail_pairs
              << std::endl;

    // ---- Step 1: evaluate each candidate's initial RMSE for scoring ----
    nlohmann::json candidate_scores = nlohmann::json::array();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const FusionCandidate& candidate = candidates[i];

        std::cout << "[Fusion] scoring candidate " << (i + 1) << "/" << candidates.size()
                  << ": " << candidate.json_path.string() << std::endl;

        double initial_rmse = std::numeric_limits<double>::infinity();
        std::string initial_error;
        if (!ComputeGlobalRmseForTransform(fusion_pairs,
                                           config.fusion,
                                           candidate.rotation,
                                           candidate.translation,
                                           config.fusion.optimization_sample_points_per_pair,
                                           initial_rmse,
                                           initial_error)) {
            std::cerr << "[Fusion] Scoring failed for candidate "
                      << candidate.json_path.string() << ": " << initial_error << std::endl;
        }

        candidate_scores.push_back(
            {
                {"candidate", candidate.json_path.string()},
                {"index", candidate.index_token},
                {"initial_global_rmse", initial_rmse},
                {"refine_rmse", std::isfinite(candidate.refine_rmse) ? candidate.refine_rmse : -1.0}
            }
        );

        std::cout << "[Fusion] candidate=" << candidate.json_path.string()
                  << " initial_rmse=" << std::fixed << std::setprecision(6)
                  << initial_rmse << std::endl;
    }

    // ---- Step 2: optimize from an SVD-projected mean rotation + mean translation ----
    double avg_rotation[3][3] = {{0.0}};
    double avg_translation[3] = {0.0, 0.0, 0.0};
    AverageCandidateTransformsSvd(candidates, avg_rotation, avg_translation);
    std::cout << "[Fusion] Starting Gauss-Newton optimization from SVD mean transform." << std::endl;

    double best_rotation[3][3] = {{0.0}};
    double best_translation[3] = {0.0, 0.0, 0.0};
    double best_global_rmse = std::numeric_limits<double>::infinity();
    std::string optimize_error;
    if (!OptimizeTransformGaussNewton(fusion_pairs,
                                      config.fusion,
                                      avg_rotation,
                                      avg_translation,
                                      config.fusion.optimization_sample_points_per_pair,
                                      best_rotation,
                                      best_translation,
                                      best_global_rmse,
                                      optimize_error)) {
        error = "Fusion optimization failed: " + optimize_error;
        return false;
    }

    if (!std::isfinite(best_global_rmse)) {
        error = "Fusion optimization produced non-finite RMSE.";
        return false;
    }

    nlohmann::json output_json;
    output_json["transform_source_to_target"] =
        RigidTransformToJson(best_rotation, best_translation);
    output_json["fusion_meta"] = {
        {"method", "svd_rotation_mean_finite_difference_gauss_newton"},
        {"candidate_count", candidates.size()},
        {"global_rmse", best_global_rmse},
        {"trim_ratio", config.fusion.trim_ratio},
        {"max_sample_points_per_pair", config.fusion.max_sample_points_per_pair},
        {"optimization_sample_points_per_pair", config.fusion.optimization_sample_points_per_pair},
        {"max_refine_rmse", config.fusion.max_refine_rmse},
        {"optimization_max_iterations", config.fusion.optimization_max_iterations},
        {"optimization_rotation_step_deg", config.fusion.optimization_rotation_step_deg},
        {"optimization_translation_step", config.fusion.optimization_translation_step},
        {"candidate_scores", candidate_scores}
    };

    const std::filesystem::path output_path =
        candidate_root / config.fusion.output_json_name;
    std::ofstream output_stream(output_path.string(), std::ios::out | std::ios::trunc);
    if (!output_stream.is_open()) {
        error = "Failed to open fused output file: " + output_path.string();
        return false;
    }
    output_stream << output_json.dump(2);
    output_stream.close();

    std::cout << "[Fusion] fused result saved: " << output_path.string() << std::endl;
    return true;
}

/**
 * @brief 将 PCL 点云保存为 PLY 文件
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool SavePCLCloudAsPly(const PCLCloudPtr& cloud,
                       const std::filesystem::path& output_path,
                       std::string& error) {
    if (!cloud || cloud->empty()) {
        error = "Cloud is empty: " + output_path.string();
        return false;
    }

    if (pcl::io::savePLYFileBinary(output_path.string(), *cloud) < 0) {
        error = "Failed to save PLY: " + output_path.string();
        return false;
    }

    return true;
}

/**
 * @brief 移除无效点并生成有限点云
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
PCLCloudPtr MakeFiniteCloud(const PCLCloudPtr& input) {
    if (!input) {
        return nullptr;
    }

    PCLCloudPtr finite_cloud(new PCLCloud);
    finite_cloud->points.reserve(input->points.size());
    for (const auto& point : input->points) {
        if (IsFinitePoint(point)) {
            finite_cloud->points.push_back(point);
        }
    }

    finite_cloud->width = static_cast<std::uint32_t>(finite_cloud->points.size());
    finite_cloud->height = 1;
    finite_cloud->is_dense = false;
    return finite_cloud;
}

/**
 * @brief 限制点云数量以控制采样规模
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
PCLCloudPtr LimitCloudPoints(const PCLCloudPtr& input,
                             std::size_t max_points) {
    if (!input || input->empty() || max_points == 0 || input->size() <= max_points) {
        return input;
    }

    const std::size_t stride = static_cast<std::size_t>(
        std::ceil(static_cast<double>(input->size()) / static_cast<double>(max_points)));

    PCLCloudPtr limited(new PCLCloud);
    limited->points.reserve((input->size() + stride - 1) / stride);
    for (std::size_t i = 0; i < input->size(); i += stride) {
        limited->points.push_back(input->points[i]);
    }

    limited->width = static_cast<std::uint32_t>(limited->points.size());
    limited->height = 1;
    limited->is_dense = input->is_dense;
    return limited;
}

/**
 * @brief 计算最近邻评估指标
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ComputeNearestNeighborMetrics(const PCLCloudPtr& query_cloud,
                                   const PCLCloudPtr& reference_cloud,
                                   std::size_t max_sample_points,
                                   DistanceMetrics& metrics,
                                   std::string& error) {
    metrics = DistanceMetrics{};

    const PCLCloudPtr query_finite = MakeFiniteCloud(query_cloud);
    const PCLCloudPtr reference_finite = MakeFiniteCloud(reference_cloud);
    if (!query_finite || query_finite->empty()) {
        error = "Query cloud is empty after removing invalid points.";
        return false;
    }
    if (!reference_finite || reference_finite->empty()) {
        error = "Reference cloud is empty after removing invalid points.";
        return false;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(reference_finite);

    const std::size_t total_points = query_finite->size();
    if (max_sample_points > 0 && total_points > max_sample_points) {
        metrics.sample_stride = static_cast<std::size_t>(
            std::ceil(static_cast<double>(total_points) / static_cast<double>(max_sample_points)));
    }

    double distance_sum = 0.0;
    double squared_distance_sum = 0.0;
    std::vector<int> nearest_indices(1);
    std::vector<float> nearest_squared_distances(1);

    for (std::size_t index = 0; index < total_points; index += metrics.sample_stride) {
        if (kdtree.nearestKSearch(query_finite->points[index], 1, nearest_indices, nearest_squared_distances) <= 0) {
            continue;
        }

        const double squared_distance = std::max(0.0, static_cast<double>(nearest_squared_distances[0]));
        const double distance = std::sqrt(squared_distance);

        distance_sum += distance;
        squared_distance_sum += squared_distance;
        metrics.max_distance = std::max(metrics.max_distance, distance);
        ++metrics.sampled_points;
    }

    if (metrics.sampled_points == 0) {
        error = "No valid nearest-neighbor matches were found during evaluation.";
        return false;
    }

    metrics.mean_distance = distance_sum / static_cast<double>(metrics.sampled_points);
    metrics.rmse = std::sqrt(squared_distance_sum / static_cast<double>(metrics.sampled_points));
    return true;
}

/**
 * @brief 按指定坐标轴镜像点云
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void MirrorCloudByNegatingX(PCLCloudPtr& cloud,
                            const MirrorAxes& axes,
                            bool save_mirrored_cloud) {
    if (!cloud || !axes.AnyEnabled()) {
        return;
    }

    for (auto& point : cloud->points) {
        if (!IsFinitePoint(point)) {
            continue;
        }
        if (axes.x) {
            point.x = -point.x;
        }
        if (axes.y) {
            point.y = -point.y;
        }
        if (axes.z) {
            point.z = -point.z;
        }
    }

    if (!save_mirrored_cloud) {
        return;
    }

    std::error_code ec;
    const std::filesystem::path mirror_result_dir = std::filesystem::path("result");
    std::filesystem::create_directories(mirror_result_dir, ec);
    if (ec) {
        std::cerr << "Failed to create mirror result directory: "
                  << mirror_result_dir.string() << std::endl;
        return;
    }

    const std::string axis_text = MirrorAxesToString(axes);
    const std::filesystem::path mirror_cloud_path =
        mirror_result_dir / ("mirrored_source_" + axis_text + ".ply");

    std::string save_error;
    if (!SavePCLCloudAsPly(cloud, mirror_cloud_path, save_error)) {
        std::cerr << "Save mirrored source cloud failed: " << save_error << std::endl;
        return;
    }

    std::cout << "Saved mirrored source cloud: " << mirror_cloud_path.string() << std::endl;
}

/**
 * @brief 求刚体变换的逆变换
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void InvertRigidTransform(const double rotation[3][3],
                          const double translation[3],
                          double inverse_rotation[3][3],
                          double inverse_translation[3]) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            inverse_rotation[row][col] = rotation[col][row];
        }
    }

    for (int row = 0; row < 3; ++row) {
        inverse_translation[row] = -(inverse_rotation[row][0] * translation[0] +
                                     inverse_rotation[row][1] * translation[1] +
                                     inverse_rotation[row][2] * translation[2]);
    }
}

/**
 * @brief 从结果 JSON 提取刚体变换
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ExtractRigidTransformFromResultJson(const std::string& result_json,
                                         double rotation[3][3],
                                         double translation[3],
                                         std::string& error) {
    nlohmann::json parsed = nlohmann::json::parse(result_json, nullptr, false);
    if (parsed.is_discarded()) {
        error = "Failed to parse result json.";
        return false;
    }

    if (!parsed.contains("transform_target_to_source") ||
        !parsed["transform_target_to_source"].is_object()) {
        error = "result json missing transform_target_to_source.";
        return false;
    }

    const auto& tf = parsed["transform_target_to_source"];
    if (!tf.contains("R") || !tf["R"].is_array() || tf["R"].size() != 3) {
        error = "result json missing valid rotation matrix R.";
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        const auto& row = tf["R"][i];
        if (!row.is_array() || row.size() != 3) {
            error = "result json rotation row format invalid.";
            return false;
        }
        for (int j = 0; j < 3; ++j) {
            if (!row[j].is_number()) {
                error = "result json rotation element is not numeric.";
                return false;
            }
            rotation[i][j] = row[j].get<double>();
        }
    }

    if (!tf.contains("t") || !tf["t"].is_array() || tf["t"].size() != 3) {
        error = "result json missing valid translation vector t.";
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        if (!tf["t"][i].is_number()) {
            error = "result json translation element is not numeric.";
            return false;
        }
        translation[i] = tf["t"][i].get<double>();
    }

    return true;
}

/**
 * @brief 从配准结果构造初始位姿 JSON
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool BuildInitialGuessJsonFromResult(const std::string& result_json,
                                     std::string& initial_transform_json,
                                     std::string& error) {
    const nlohmann::json parsed = nlohmann::json::parse(result_json, nullptr, false);
    if (parsed.is_discarded()) {
        error = "Failed to parse stage-1 result json.";
        return false;
    }

    if (!parsed.contains("transform_target_to_source") ||
        !parsed["transform_target_to_source"].is_object()) {
        error = "Stage-1 result json missing transform_target_to_source.";
        return false;
    }

    const auto& tf = parsed["transform_target_to_source"];
    if (!tf.contains("matrix") || !tf["matrix"].is_array() || tf["matrix"].size() != 4) {
        error = "Stage-1 result json missing valid transform matrix.";
        return false;
    }

    for (int row = 0; row < 4; ++row) {
        if (!tf["matrix"][row].is_array() || tf["matrix"][row].size() != 4) {
            error = "Stage-1 transform matrix row format invalid.";
            return false;
        }
        for (int col = 0; col < 4; ++col) {
            if (!tf["matrix"][row][col].is_number()) {
                error = "Stage-1 transform matrix element is not numeric.";
                return false;
            }
        }
    }

    nlohmann::json init_json;
    init_json["matrix"] = tf["matrix"];
    initial_transform_json = init_json.dump();
    return true;
}

/**
 * @brief 执行单次点云对配准
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool RunCloudPairRegistration(const std::shared_ptr<HVPointCloud>& source_cloud,
                              const std::shared_ptr<HVPointCloud>& target_cloud,
                              const RegistrationParams& registration_params,
                              std::shared_ptr<HVPointCloud>& aligned_target,
                              std::shared_ptr<HVPointCloud>& merged_cloud,
                              std::string& result_json,
                              int& run_time_ms,
                              std::string& error) {
    aligned_target.reset();
    merged_cloud.reset();
    result_json.clear();
    run_time_ms = 0;

    if (!source_cloud || source_cloud->points.empty()) {
        error = "Source cloud is empty before registration.";
        return false;
    }
    if (!target_cloud || target_cloud->points.empty()) {
        error = "Target cloud is empty before registration.";
        return false;
    }

    auto source_param = source_cloud;
    auto target_param = target_cloud;
    int method = registration_params.method;
    double voxel_size = registration_params.voxel_size;
    double max_corr_dist = registration_params.max_correspondence_distance;
    int max_iter = registration_params.max_iterations;
    bool use_initial_guess = registration_params.use_initial_guess;
    std::string initial_transform_json = registration_params.initial_transform_json;

    HVCloudPairICP icp;
    if (icp.init() != SUCCESS) {
        error = "init() failed: " + icp.get_algorithm_error_message();
        return false;
    }

    std::vector<void*> params;
    params.push_back(&source_param);
    params.push_back(&target_param);
    params.push_back(&method);
    params.push_back(&voxel_size);
    params.push_back(&max_corr_dist);
    params.push_back(&max_iter);
    params.push_back(&use_initial_guess);
    params.push_back(&initial_transform_json);

    if (icp.set_algorithm_params(params) != SUCCESS) {
        error = "set_algorithm_params() failed.";
        return false;
    }

    if (icp.run() != SUCCESS) {
        error = "run() failed: " + icp.get_algorithm_error_message();
        return false;
    }

    auto outputs = icp.get_algorithm_result();
    if (outputs.size() < 3 || outputs[0] == nullptr || outputs[1] == nullptr || outputs[2] == nullptr) {
        error = "Invalid run output.";
        return false;
    }

    aligned_target = *static_cast<std::shared_ptr<HVPointCloud>*>(outputs[0]);
    merged_cloud = *static_cast<std::shared_ptr<HVPointCloud>*>(outputs[1]);
    result_json = *static_cast<std::string*>(outputs[2]);
    run_time_ms = icp.get_algorithm_use_time();
    return true;
}

/**
 * @brief 按刚体变换生成变换后点云
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
PCLCloudPtr TransformCloudByRt(
    const PCLCloudPtr& source_cloud,
    const double rotation[3][3],
    const double translation[3]) {
    if (!source_cloud || source_cloud->empty()) {
        return nullptr;
    }

    PCLCloudPtr transformed(new PCLCloud);
    transformed->points.reserve(source_cloud->points.size());

    for (const auto& p : source_cloud->points) {
        if (!IsFinitePoint(p)) {
            continue;
        }

        pcl::PointXYZ tp;
        tp.x = rotation[0][0] * p.x + rotation[0][1] * p.y + rotation[0][2] * p.z + translation[0];
        tp.y = rotation[1][0] * p.x + rotation[1][1] * p.y + rotation[1][2] * p.z + translation[1];
        tp.z = rotation[2][0] * p.x + rotation[2][1] * p.y + rotation[2][2] * p.z + translation[2];
        transformed->points.push_back(tp);
    }

    transformed->width = static_cast<std::uint32_t>(transformed->points.size());
    transformed->height = 1;
    transformed->is_dense = false;

    return transformed;
}

/**
 * @brief 在原始点云上评估配准结果
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool EvaluateResultOnRawClouds(const std::string& result_json,
                               const PCLCloudPtr& source_cloud_raw,
                               const PCLCloudPtr& target_cloud_raw,
                               std::size_t max_sample_points,
                               DistanceMetrics& after_metrics,
                               std::string& error) {
    double target_to_source_rotation[3][3] = {{0.0}};
    double target_to_source_translation[3] = {0.0, 0.0, 0.0};
    if (!ExtractRigidTransformFromResultJson(result_json,
                                             target_to_source_rotation,
                                             target_to_source_translation,
                                             error)) {
        return false;
    }

    double source_to_target_rotation[3][3] = {{0.0}};
    double source_to_target_translation[3] = {0.0, 0.0, 0.0};
    InvertRigidTransform(target_to_source_rotation,
                         target_to_source_translation,
                         source_to_target_rotation,
                         source_to_target_translation);

    // 对原始 source 点云施加最终 source->target 变换，并导出可视化结果。
    const PCLCloudPtr transformed_source = TransformCloudByRt(source_cloud_raw,
                                                              source_to_target_rotation,
                                                              source_to_target_translation);
    if (!transformed_source || transformed_source->empty()) {
        error = "Transformed source cloud is empty during stage evaluation.";
        return false;
    }

    return ComputeNearestNeighborMetrics(transformed_source,
                                         target_cloud_raw,
                                         max_sample_points,
                                         after_metrics,
                                         error);
}

/**
 * @brief 打印阶段评估指标
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
void PrintStageEvaluationMetrics(const std::string& stage_name,
                                 const DistanceMetrics& metrics) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[" << stage_name << " Evaluation] mean=" << metrics.mean_distance
              << ", rmse=" << metrics.rmse
              << ", max=" << metrics.max_distance
              << ", sampled=" << metrics.sampled_points
              << ", stride=" << metrics.sample_stride
              << std::endl;
}

/**
 * @brief 将刚体变换转换为 JSON 对象
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
nlohmann::json RigidTransformToJson(const double rotation[3][3],
                                    const double translation[3]) {
    return {
        {
            "R",
            {
                { rotation[0][0], rotation[0][1], rotation[0][2] },
                { rotation[1][0], rotation[1][1], rotation[1][2] },
                { rotation[2][0], rotation[2][1], rotation[2][2] }
            }
        },
        { "t", { translation[0], translation[1], translation[2] } }
    };
}

/**
 * @brief 保存最终变换结果与点云文件
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool SaveFinalTransformAndCloud(const PCLCloudPtr& source_cloud_raw,
                                const std::string& result_json,
                                const OutputConfig& output_config,
                                std::string& saved_result_json,
                                std::string& error) {
    std::error_code ec;
    const std::filesystem::path result_dir(output_config.result_dir);
    std::filesystem::create_directories(result_dir, ec);
    if (ec) {
        error = "Failed to create result directory: " + result_dir.string();
        return false;
    }

    double target_to_source_rotation[3][3] = {{0.0}};
    double target_to_source_translation[3] = {0.0, 0.0, 0.0};
    if (!ExtractRigidTransformFromResultJson(result_json,
                                             target_to_source_rotation,
                                             target_to_source_translation,
                                             error)) {
        return false;
    }

    double source_to_target_rotation[3][3] = {{0.0}};
    double source_to_target_translation[3] = {0.0, 0.0, 0.0};
    InvertRigidTransform(target_to_source_rotation,
                         target_to_source_translation,
                         source_to_target_rotation,
                         source_to_target_translation);

    const PCLCloudPtr transformed_source = TransformCloudByRt(source_cloud_raw,
                                                              source_to_target_rotation,
                                                              source_to_target_translation);
    if (!transformed_source || transformed_source->empty()) {
        error = "Transformed source cloud is empty, cannot save final output.";
        return false;
    }

    const std::filesystem::path transformed_source_path = result_dir / output_config.transformed_source_name;
    if (!SavePCLCloudAsPly(transformed_source, transformed_source_path, error)) {
        return false;
    }

    nlohmann::json final_json;
    final_json["transform_source_to_target"] =
        RigidTransformToJson(source_to_target_rotation, source_to_target_translation);
    saved_result_json = final_json.dump(2);

    std::string initial_transform_json_text;
    if (!BuildInitialGuessJsonFromResult(result_json, initial_transform_json_text, error)) {
        return false;
    }

    nlohmann::json initial_transform_json = nlohmann::json::parse(initial_transform_json_text, nullptr, false);
    if (initial_transform_json.is_discarded()) {
        error = "Failed to parse generated initial transform json text.";
        return false;
    }

    const std::filesystem::path json_path = result_dir / output_config.result_json_name;
    std::ofstream json_file(json_path.string(), std::ios::out | std::ios::trunc);
    if (!json_file.is_open()) {
        error = "Failed to open result json file: " + json_path.string();
        return false;
    }
    json_file << saved_result_json;
    json_file.close();

    const std::filesystem::path initial_transform_json_path =
        result_dir / output_config.initial_transform_json_name;
    std::ofstream initial_transform_json_file(initial_transform_json_path.string(),
                                              std::ios::out | std::ios::trunc);
    if (!initial_transform_json_file.is_open()) {
        error = "Failed to open initial transform json file: " +
                initial_transform_json_path.string();
        return false;
    }
    initial_transform_json_file << initial_transform_json.dump(2);
    initial_transform_json_file.close();

    std::cout << "Saved transformed source cloud: " << transformed_source_path.string() << std::endl;
    std::cout << "Saved source->target transform json: " << json_path.string() << std::endl;
    std::cout << "Saved reusable initial_transform_json: "
              << initial_transform_json_path.string() << std::endl;
    return true;
}

/**
 * @brief 判断当前阶段是否需要执行评估
 * @return 见函数返回类型定义
 *
 * @description 工作流执行步骤：
 *   1. 根据函数输入执行对应的数据处理或状态判断
 *   2. 在必要时完成参数校验、格式转换或中间结果构建
 *   3. 输出处理结果，或在失败时通过返回值/错误信息上报状态
 */
bool ShouldEvaluateStage(const EvaluationConfig& evaluation,
                         bool is_coarse_stage) {
    if (!evaluation.enabled) {
        return false;
    }

    return is_coarse_stage ? evaluation.evaluate_coarse : evaluation.evaluate_refine;
}

}  // namespace

/**
 * @brief 程序主函数入口
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组，argv[1] 可传入配置文件路径
 * @return 程序退出码：0 表示成功，-1 表示失败
 *
 * @description 工作流执行步骤：
 *   1. 初始化工作目录与日志系统，并解析命令行中的配置文件路径
 *   2. 加载 JSON 配置，判断当前运行模式为 registration 或 fusion
 *   3. 若为 fusion 模式，则执行多位姿融合流程并输出融合结果
 *   4. 若为 registration 模式，则执行点云加载、预处理、粗配准、精配准与评估
 *   5. 保存最终结果文件，并将执行状态打印到控制台与日志文件
 */
int main(int argc, char** argv) {
    EnsureWorkingDirectoryAtProjectRoot(argc, argv);

    // Create logger manager and auto-handle log file lifecycle.
    LogManager log_manager;

    // 统一先加载配置；后续 registration / fusion 两条主流程都依赖同一份解析结果。
    const std::filesystem::path config_path = ResolveConfigPath(argc, argv);
    AppConfig config;
    std::string config_error;
    if (!LoadConfigFromJsonFile(config_path, config, config_error)) {
        std::cerr << config_error << std::endl;
        std::cerr << "Config path hint: pass custom path as first argument, or input it interactively; "
                  << "press Enter to use default path: "
                  << ResolveDefaultConfigPath(argc, argv).string() << std::endl;
        return -1;
    }

    if (config.run_mode == "fusion") {
        // fusion 模式直接走多位姿联合优化，不进入单对点云配准主流程。
        std::cout << "Loaded config: " << config_path.string() << std::endl;
        std::cout << "Run mode: fusion" << std::endl;

        std::string fusion_error;
        if (!RunFusionMode(config, fusion_error)) {
            std::cerr << "Fusion mode failed: " << fusion_error << std::endl;
            return -1;
        }

        std::cout << "Fusion mode finished successfully." << std::endl;
        return 0;
    }

    std::cout << "Loaded config: " << config_path.string() << std::endl;
    std::cout << "Run mode: registration" << std::endl;
    std::cout << "Source cloud: " << config.source_cloud_path << std::endl;
    std::cout << "Target cloud: " << config.target_cloud_path << std::endl;
    std::cout << "Mirror axes: " << MirrorAxesToString(config.mirror_axes) << std::endl;
    std::cout << "Pipeline run_coarse=" << (config.pipeline.run_coarse ? "true" : "false")
              << ", run_refine=" << (config.pipeline.run_refine ? "true" : "false")
              << std::endl;

    // registration 模式先加载原始点云，再视配置执行镜像、滤波和多阶段配准。
    PCLCloudPtr left_raw;
    PCLCloudPtr right_raw;
    std::string load_error;

    if (!LoadPointCloud(config.source_cloud_path, left_raw, load_error, config.enable_txt_cache)) {
        std::cerr << load_error << std::endl;
        return -1;
    }
    if (!LoadPointCloud(config.target_cloud_path, right_raw, load_error, config.enable_txt_cache)) {
        std::cerr << load_error << std::endl;
        return -1;
    }

    if (config.noise_filter.enabled) {
        const std::size_t left_before = left_raw ? left_raw->size() : 0;
        const std::size_t right_before = right_raw ? right_raw->size() : 0;

        left_raw = StatisticalOutlierFilter(left_raw, config.noise_filter);
        right_raw = StatisticalOutlierFilter(right_raw, config.noise_filter);

        std::cout << "Applied statistical noise filter (mean_k=" << config.noise_filter.mean_k
                  << ", stddev_mul_thresh=" << config.noise_filter.stddev_mul_thresh
                  << ") source: " << left_before << " -> " << (left_raw ? left_raw->size() : 0)
                  << ", target: " << right_before << " -> " << (right_raw ? right_raw->size() : 0)
                  << std::endl;
    }

    MirrorCloudByNegatingX(left_raw, config.mirror_axes);
    if (config.mirror_axes.AnyEnabled()) {
        std::cout << "Applied source mirror: " << MirrorAxesToString(config.mirror_axes)
                  << " axis negation." << std::endl;
    } else {
        std::cout << "Source mirror disabled." << std::endl;
    }

    std::string coarse_result_json;
    bool coarse_completed = false;

    if (config.pipeline.run_coarse) {
        const PCLCloudPtr left_filtered = VoxelFilter(left_raw, config.filter.leaf_size);
        const PCLCloudPtr right_filtered = VoxelFilter(right_raw, config.filter.leaf_size);

        std::cout << "Left raw points: " << left_raw->size()
                  << ", coarse filtered: " << left_filtered->size() << std::endl;
        std::cout << "Right raw points: " << right_raw->size()
                  << ", coarse filtered: " << right_filtered->size() << std::endl;

        auto source_cloud_coarse = std::make_shared<HVPointCloud>(PointCloudConverter::FromPCL(*left_filtered));
        auto target_cloud_coarse = std::make_shared<HVPointCloud>(PointCloudConverter::FromPCL(*right_filtered));

        std::shared_ptr<HVPointCloud> coarse_aligned_target;
        std::shared_ptr<HVPointCloud> coarse_merged_cloud;
        int coarse_time_ms = 0;
        std::string registration_error;
        if (!RunCloudPairRegistration(source_cloud_coarse,
                                      target_cloud_coarse,
                                      config.coarse_registration,
                                      coarse_aligned_target,
                                      coarse_merged_cloud,
                                      coarse_result_json,
                                      coarse_time_ms,
                                      registration_error)) {
            std::cerr << "Stage-1 coarse registration failed: " << registration_error << std::endl;
            return -1;
        }

        coarse_completed = true;
        std::cout << "[Stage-1 Coarse] success, time(ms): " << coarse_time_ms << std::endl;

        if (ShouldEvaluateStage(config.evaluation, true)) {
            DistanceMetrics coarse_metrics;
            std::string eval_error;
            if (!EvaluateResultOnRawClouds(coarse_result_json,
                                           left_raw,
                                           right_raw,
                                           config.evaluation.max_sample_points,
                                           coarse_metrics,
                                           eval_error)) {
                std::cerr << "Evaluate stage-1 metrics failed: " << eval_error << std::endl;
                return -1;
            }

            PrintStageEvaluationMetrics("Stage-1 Coarse", coarse_metrics);
        }
    }

    std::string final_result_json;
    std::string final_stage_name;

    if (config.pipeline.run_refine) {
        RegistrationParams refine_params = config.refine_registration;
        std::string registration_error;

        if (coarse_completed) {
            std::string stage1_init_guess_json;
            if (!BuildInitialGuessJsonFromResult(coarse_result_json, stage1_init_guess_json, registration_error)) {
                std::cerr << "Build stage-1 initial transform failed: " << registration_error << std::endl;
                return -1;
            }

            // When coarse and refine are both enabled, always chain stage-1 result into stage-2.
            refine_params.use_initial_guess = true;
            refine_params.initial_transform_json = stage1_init_guess_json;
            std::cout << "[Stage-2 Refine] using Stage-1 transform as initial_transform_json." << std::endl;
        }

        if (!coarse_completed && refine_params.use_initial_guess && refine_params.initial_transform_json.empty()) {
            std::cerr << "Stage-2 refine requires initial guess, but coarse stage is disabled and "
                      << "refine_registration.initial_transform_json is empty." << std::endl;
            return -1;
        }

        auto source_cloud_refine = std::make_shared<HVPointCloud>(PointCloudConverter::FromPCL(*left_raw));
        auto target_cloud_refine = std::make_shared<HVPointCloud>(PointCloudConverter::FromPCL(*right_raw));

        std::shared_ptr<HVPointCloud> refine_aligned_target;
        std::shared_ptr<HVPointCloud> refine_merged_cloud;
        std::string refine_result_json;
        int refine_time_ms = 0;
        if (!RunCloudPairRegistration(source_cloud_refine,
                                      target_cloud_refine,
                                      refine_params,
                                      refine_aligned_target,
                                      refine_merged_cloud,
                                      refine_result_json,
                                      refine_time_ms,
                                      registration_error)) {
            std::cerr << "Stage-2 refine registration failed: " << registration_error << std::endl;
            return -1;
        }

        final_result_json = refine_result_json;
        final_stage_name = "Stage-2 Refine";
        std::cout << "[Stage-2 Refine] success, time(ms): " << refine_time_ms << std::endl;

        if (ShouldEvaluateStage(config.evaluation, false)) {
            DistanceMetrics refine_metrics;
            std::string eval_error;
            if (!EvaluateResultOnRawClouds(refine_result_json,
                                           left_raw,
                                           right_raw,
                                           config.evaluation.max_sample_points,
                                           refine_metrics,
                                           eval_error)) {
                std::cerr << "Evaluate stage-2 metrics failed: " << eval_error << std::endl;
                return -1;
            }

            PrintStageEvaluationMetrics("Stage-2 Refine", refine_metrics);
        }
    } else {
        final_result_json = coarse_result_json;
        final_stage_name = "Stage-1 Coarse";
    }

    std::string saved_result_json;
    std::string save_error;
    if (!SaveFinalTransformAndCloud(left_raw,
                                    final_result_json,
                                    config.output,
                                    saved_result_json,
                                    save_error)) {
        std::cerr << "Save final result failed: " << save_error << std::endl;
        return -1;
    }

    std::cout << "Final output stage: " << final_stage_name << std::endl;
    return 0;
}


