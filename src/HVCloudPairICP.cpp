// Author: weitermachen
// Time: 2026-03-24

#include "HVCloudPairICP.h"

#include "HVUtils.h"
#include "HVI18n.h"
#include "ICPCore.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

const hvi18n::Dictionary kTexts = {
    { "algorithm.display", { "Pair cloud ICP registration", "Pair cloud ICP registration" } },
    { "output.aligned_target", { "aligned target cloud", "aligned target cloud" } },
    { "output.merged_cloud", { "merged cloud", "merged cloud" } },
    { "output.result_json", { "registration result json", "registration result json" } },
    { "option.icp", { "ICP", "ICP" } },
    { "option.gicp", { "GICP", "GICP" } },
    { "option.lp_icp", { "LP-ICP", "LP-ICP" } },
    { "msg.source_null", { "Source cloud is null", "Source cloud is null" } },
    { "msg.target_null", { "Target cloud is null", "Target cloud is null" } },
    { "msg.cloud_too_small", { "Point cloud has too few points for registration", "Point cloud has too few points for registration" } },
    { "msg.invalid_distance", { "Max correspondence distance must be greater than 0", "Max correspondence distance must be greater than 0" } },
    { "msg.invalid_iterations", { "Max iterations must be greater than 0", "Max iterations must be greater than 0" } },
    { "msg.invalid_init_json", { "Failed to parse initial transform json", "Failed to parse initial transform json" } },
    { "msg.registration_failed", { "Point cloud registration failed", "Point cloud registration failed" } },
    { "msg.success", { "Point cloud registration succeeded", "Point cloud registration succeeded" } },
    { "name.0", { "source cloud", "source cloud" } },
    { "name.1", { "target cloud", "target cloud" } },
    { "name.2", { "registration method", "registration method" } },
    { "name.3", { "voxel size", "voxel size" } },
    { "name.4", { "max correspondence distance", "max correspondence distance" } },
    { "name.5", { "max iterations", "max iterations" } },
    { "name.6", { "use initial guess", "use initial guess" } },
    { "name.7", { "initial transform json", "initial transform json" } },
    { "desc.0", { "Source point cloud used as registration reference", "Source point cloud used as registration reference" } },
    { "desc.1", { "Target point cloud to be aligned to source cloud", "Target point cloud to be aligned to source cloud" } },
    { "desc.2", { "Point cloud registration algorithm type", "Point cloud registration algorithm type" } },
    { "desc.3", { "Voxel size for preprocessing. <= 0 means disabled", "Voxel size for preprocessing. <= 0 means disabled" } },
    { "desc.4", { "Maximum distance threshold for correspondence search", "Maximum distance threshold for correspondence search" } },
    { "desc.5", { "Maximum number of optimization iterations", "Maximum number of optimization iterations" } },
    { "desc.6", { "Whether to use an externally provided initial transform", "Whether to use an externally provided initial transform" } },
    { "desc.7", { "4x4 initial transform matrix json. Supports {\"matrix\":[...]} or a direct 2D array", "4x4 initial transform matrix json. Supports {\"matrix\":[...]} or a direct 2D array" } }
};

std::string Tr(int language, const std::string& key) {
    return hvi18n::Translate(kTexts, key, language);
}

std::string BoolToJson(bool v) {
    return v ? "true" : "false";
}

std::string MatrixToJsonString(const Eigen::Matrix4f& mat) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    oss << "[";
    for (int i = 0; i < 4; ++i) {
        oss << "[";
        for (int j = 0; j < 4; ++j) {
            oss << mat(i, j);
            if (j < 3) oss << ",";
        }
        oss << "]";
        if (i < 3) oss << ",";
    }
    oss << "]";
    return oss.str();
}

std::string RotationToJsonString(const Eigen::Matrix4f& mat) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    oss << "[";
    for (int i = 0; i < 3; ++i) {
        oss << "[";
        for (int j = 0; j < 3; ++j) {
            oss << mat(i, j);
            if (j < 2) oss << ",";
        }
        oss << "]";
        if (i < 2) oss << ",";
    }
    oss << "]";
    return oss.str();
}

std::string BuildResultJson(const ICPResult& result,
                            int method,
                            std::size_t source_points,
                            std::size_t target_points,
                            std::size_t aligned_points,
                            double voxel_size,
                            double max_correspondence_distance,
                            int max_iterations,
                            bool use_initial_guess) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    oss << "{";
    //oss << "\"method\":\"" << (method == 1 ? "gicp" : "icp") << "\",";
    std::string method_name = "icp";
    if (method == 1) {
        method_name = "gicp";
    }
    else if (method == 2) {
        method_name = "lp_icp";
    }
    oss << "\"method\":\"" << method_name << "\",";
    oss << "\"converged\":" << BoolToJson(result.converged) << ",";
    oss << "\"success\":" << BoolToJson(result.success) << ",";
    oss << "\"fitness_score\":" << result.fitness_score << ",";
    oss << "\"rmse\":" << result.rmse << ",";
    oss << "\"source_points\":" << source_points << ",";
    oss << "\"target_points\":" << target_points << ",";
    oss << "\"aligned_points\":" << aligned_points << ",";
    oss << "\"params\":{";
    oss << "\"voxel_size\":" << voxel_size << ",";
    oss << "\"max_correspondence_distance\":" << max_correspondence_distance << ",";
    oss << "\"max_iterations\":" << max_iterations << ",";
    oss << "\"use_initial_guess\":" << BoolToJson(use_initial_guess);
    oss << "},";
    oss << "\"transform_target_to_source\":{";
    oss << "\"matrix\":" << MatrixToJsonString(result.transform) << ",";
    oss << "\"R\":" << RotationToJsonString(result.transform) << ",";
    oss << "\"t\":[" << result.transform(0,3) << "," << result.transform(1,3) << "," << result.transform(2,3) << "]";
    oss << "}";
    if (!result.error_message.empty()) {
        oss << ",\"message\":\"" << result.error_message << "\"";
    }
    oss << "}";
    return oss.str();
}

} // namespace

HVCloudPairICP::HVCloudPairICP() = default;
HVCloudPairICP::~HVCloudPairICP() = default;

int HVCloudPairICP::init() {
    execute_status = NODE_STATUS_NOT_RUN;
    run_time = 0;
    error_msg.clear();
    resultJson.clear();
    alignedTargetCloud.reset();
    mergedCloud.reset();
    return SUCCESS;
}

int HVCloudPairICP::run() {
    const auto start = std::chrono::steady_clock::now();
    execute_status = NODE_STATUS_RUNNING;
    error_msg.clear();
    resultJson.clear();
    alignedTargetCloud.reset();
    mergedCloud.reset();

    if (!sourceCloud) {
        error_msg = Tr(language_, "msg.source_null");
        execute_status = ALGORITHM_RUN_ERROR;
        return ALGORITHM_RUN_ERROR;
    }
    if (!targetCloud) {
        error_msg = Tr(language_, "msg.target_null");
        execute_status = ALGORITHM_RUN_ERROR;
        return ALGORITHM_RUN_ERROR;
    }
    if (sourceCloud->points.size() < 3 || targetCloud->points.size() < 3) {
        error_msg = Tr(language_, "msg.cloud_too_small");
        execute_status = ALGORITHM_RUN_ERROR;
        return ALGORITHM_RUN_ERROR;
    }
    if (max_correspondence_distance <= 0.0) {
        error_msg = Tr(language_, "msg.invalid_distance");
        execute_status = ALGORITHM_RUN_ERROR;
        return ALGORITHM_RUN_ERROR;
    }
    if (max_iterations <= 0) {
        error_msg = Tr(language_, "msg.invalid_iterations");
        execute_status = ALGORITHM_RUN_ERROR;
        return ALGORITHM_RUN_ERROR;
    }

    ICPParams params;
    params.method = method;
    params.voxel_size = voxel_size;
    params.max_correspondence_distance = max_correspondence_distance;
    params.max_iterations = max_iterations;
    params.use_initial_guess = use_initial_guess;

    if (use_initial_guess) {
        std::string parse_error;
        if (!ParseTransformJson(initial_transform_json, params.initial_guess, parse_error)) {
            error_msg = Tr(language_, "msg.invalid_init_json") + ": " + parse_error;
            execute_status = ALGORITHM_RUN_ERROR;
            return ALGORITHM_RUN_ERROR;
        }
    }

    auto sourcePcl = PointCloudConverter::ToPCL(*sourceCloud);
    auto targetPcl = PointCloudConverter::ToPCL(*targetCloud);

    const ICPResult result = RunPairRegistration(sourcePcl, targetPcl, params);
    if (!result.success) {
        error_msg = result.error_message.empty() ? Tr(language_, "msg.registration_failed") : result.error_message;
        execute_status = ALGORITHM_RUN_ERROR;
        return ALGORITHM_RUN_ERROR;
    }

    alignedTargetCloud = std::make_shared<HVPointCloud>(PointCloudConverter::FromPCL(*result.aligned_target));
    mergedCloud = std::make_shared<HVPointCloud>(PointCloudConverter::FromPCL(*result.merged_cloud));

    resultJson = BuildResultJson(result,
                                 method,
                                 sourceCloud->points.size(),
                                 targetCloud->points.size(),
                                 result.aligned_target ? result.aligned_target->size() : 0,
                                 voxel_size,
                                 max_correspondence_distance,
                                 max_iterations,
                                 use_initial_guess);

    error_msg = Tr(language_, "msg.success");
    const auto end = std::chrono::steady_clock::now();
    run_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    execute_status = SUCCESS;
    return SUCCESS;
}

int HVCloudPairICP::set_algorithm_params(const std::vector<void*>& params,
                                         const std::vector<int>& paramID) {
    if (paramID.empty()) {
        sourceCloud = cast_param_sharedPtr<HVPointCloud>(params, 0);
        targetCloud = cast_param_sharedPtr<HVPointCloud>(params, 1);
        method = cast_param<int>(params, 2);
        voxel_size = cast_param<double>(params, 3);
        max_correspondence_distance = cast_param<double>(params, 4);
        max_iterations = cast_param<int>(params, 5);
        use_initial_guess = cast_param<bool>(params, 6);
        initial_transform_json = cast_param<std::string>(params, 7);
    } else {
        for (size_t i = 0; i < paramID.size(); ++i) {
            switch (paramID[i]) {
            case 0: sourceCloud = cast_param_sharedPtr<HVPointCloud>(params, static_cast<int>(i)); break;
            case 1: targetCloud = cast_param_sharedPtr<HVPointCloud>(params, static_cast<int>(i)); break;
            case 2: method = cast_param<int>(params, static_cast<int>(i)); break;
            case 3: voxel_size = cast_param<double>(params, static_cast<int>(i)); break;
            case 4: max_correspondence_distance = cast_param<double>(params, static_cast<int>(i)); break;
            case 5: max_iterations = cast_param<int>(params, static_cast<int>(i)); break;
            case 6: use_initial_guess = cast_param<bool>(params, static_cast<int>(i)); break;
            case 7: initial_transform_json = cast_param<std::string>(params, static_cast<int>(i)); break;
            default: break;
            }
        }
    }
    return SUCCESS;
}

std::vector<void*> HVCloudPairICP::get_current_params() {
    return {
        &sourceCloud,
        &targetCloud,
        &method,
        &voxel_size,
        &max_correspondence_distance,
        &max_iterations,
        &use_initial_guess,
        &initial_transform_json
    };
}

std::vector<void*> HVCloudPairICP::get_algorithm_result() {
    if (execute_status == SUCCESS) {
        return { &alignedTargetCloud, &mergedCloud, &resultJson };
    }
    return { nullptr, nullptr, nullptr };
}

std::vector<int> HVCloudPairICP::get_algorithm_input_params_type() {
    return {
        HV_POINTCLOUD,
        HV_POINTCLOUD,
        HV_INT,
        HV_DOUBLE,
        HV_DOUBLE,
        HV_INT,
        HV_BOOLEAN,
        HV_STRING
    };
}

std::vector<int> HVCloudPairICP::get_algorithm_output_params_type() {
    return { HV_POINTCLOUD, HV_POINTCLOUD, HV_STRING };
}

std::vector<std::string> HVCloudPairICP::get_algorithm_input_params_name() {
    return {
        Tr(language_, "name.0"),
        Tr(language_, "name.1"),
        Tr(language_, "name.2"),
        Tr(language_, "name.3"),
        Tr(language_, "name.4"),
        Tr(language_, "name.5"),
        Tr(language_, "name.6"),
        Tr(language_, "name.7")
    };
}

std::vector<std::string> HVCloudPairICP::get_algorithm_output_params_name() {
    return {
        Tr(language_, "output.aligned_target"),
        Tr(language_, "output.merged_cloud"),
        Tr(language_, "output.result_json")
    };
}

std::vector<bool> HVCloudPairICP::get_algorithm_input_params_bindable() {
    return { true, true, false, false, false, false, false, false };
}

std::vector<ParamMetadata> HVCloudPairICP::get_algorithm_input_params_metadata() {
    std::vector<ParamMetadata> metadata_list;

    ParamMetadata meta0;
    meta0.param_name = "source cloud";
    meta0.param_description = Tr(language_, "desc.0");
    meta0.param_type = HV_POINTCLOUD;
    meta0.constraint_type = CONSTRAINT_NONE;
    metadata_list.push_back(meta0);

    ParamMetadata meta1;
    meta1.param_name = "target cloud";
    meta1.param_description = Tr(language_, "desc.1");
    meta1.param_type = HV_POINTCLOUD;
    meta1.constraint_type = CONSTRAINT_NONE;
    metadata_list.push_back(meta1);

    ParamMetadata meta2;
    meta2.param_name = "registration method";
    meta2.param_description = Tr(language_, "desc.2");
    meta2.param_type = HV_INT;
    meta2.constraint_type = CONSTRAINT_OPTIONS;
    meta2.options_constraint.AddOption(0, Tr(language_, "option.icp"));
    meta2.options_constraint.AddOption(1, Tr(language_, "option.gicp"));
    meta2.options_constraint.AddOption(2, Tr(language_, "option.lp_icp"));
    meta2.options_constraint.default_index = 0;
    metadata_list.push_back(meta2);

    ParamMetadata meta3;
    meta3.param_name = "voxel size";
    meta3.param_description = Tr(language_, "desc.3");
    meta3.param_type = HV_DOUBLE;
    meta3.constraint_type = CONSTRAINT_RANGE;
    meta3.range_constraint = RangeConstraint(0.0, 1000.0, 0.0);
    metadata_list.push_back(meta3);

    ParamMetadata meta4;
    meta4.param_name = "max correspondence distance";
    meta4.param_description = Tr(language_, "desc.4");
    meta4.param_type = HV_DOUBLE;
    meta4.constraint_type = CONSTRAINT_RANGE;
    meta4.range_constraint = RangeConstraint(1e-6, 10000.0, 1.0);
    metadata_list.push_back(meta4);

    ParamMetadata meta5;
    meta5.param_name = "max iterations";
    meta5.param_description = Tr(language_, "desc.5");
    meta5.param_type = HV_INT;
    meta5.constraint_type = CONSTRAINT_RANGE;
    meta5.range_constraint = RangeConstraint(1, 100000, 50);
    metadata_list.push_back(meta5);

    ParamMetadata meta6;
    meta6.param_name = "use initial guess";
    meta6.param_description = Tr(language_, "desc.6");
    meta6.param_type = HV_BOOLEAN;
    meta6.constraint_type = CONSTRAINT_NONE;
    metadata_list.push_back(meta6);

    ParamMetadata meta7;
    meta7.param_name = "initial transform json";
    meta7.param_description = Tr(language_, "desc.7");
    meta7.param_type = HV_STRING;
    meta7.constraint_type = CONSTRAINT_NONE;
    meta7.dependencies.push_back(ParamDependency(6, DEPENDS_ON_IN_LIST, { "1", "true", "True" }));
    metadata_list.push_back(meta7);

    return metadata_list;
}

int HVCloudPairICP::get_algorithm_execute_status() {
    return execute_status;
}

std::string HVCloudPairICP::get_algorithm_error_message() {
    return error_msg;
}

long HVCloudPairICP::get_algorithm_use_time() {
    return run_time;
}

bool HVCloudPairICP::algorithm_params_setting_status() {
    return true;
}

bool HVCloudPairICP::algorithm_init_status() {
    return true;
}

bool HVCloudPairICP::save_params_to_json(const std::string& filePath) {
    try {
        nlohmann::json params_json = nlohmann::json::array();
        add_param(params_json, "method", HV_INT, method);
        add_param(params_json, "voxel_size", HV_DOUBLE, voxel_size);
        add_param(params_json, "max_correspondence_distance", HV_DOUBLE, max_correspondence_distance);
        add_param(params_json, "max_iterations", HV_INT, max_iterations);
        add_param(params_json, "use_initial_guess", HV_BOOLEAN, use_initial_guess);
        add_param(params_json, "initial_transform_json", HV_STRING, initial_transform_json);

        std::ofstream file(filePath);
        if (!file.is_open()) {
            return false;
        }
        file << params_json.dump(4);
        file.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool HVCloudPairICP::load_params_from_json(const std::string& filePath) {
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            return false;
        }

        nlohmann::json params_json;
        file >> params_json;
        file.close();

        if (!params_json.is_array()) {
            return false;
        }

        for (const auto& param_json : params_json) {
            if (!param_json.contains("name") || !param_json.contains("type")) {
                continue;
            }
            const std::string param_name = param_json["name"];
            if (param_name == "method") {
                method = param_json["value"];
            } else if (param_name == "voxel_size") {
                voxel_size = param_json["value"];
            } else if (param_name == "max_correspondence_distance") {
                max_correspondence_distance = param_json["value"];
            } else if (param_name == "max_iterations") {
                max_iterations = param_json["value"];
            } else if (param_name == "use_initial_guess") {
                use_initial_guess = param_json["value"];
            } else if (param_name == "initial_transform_json") {
                initial_transform_json = param_json["value"];
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

AlgorithmType HVCloudPairICP::get_algorithm_type() {
    return AlgorithmType::PointCloudProcess;
}

void HVCloudPairICP::set_language(int language) {
    if (hvi18n::IsSupportedLanguage(language)) {
        language_ = language;
    }
}

int HVCloudPairICP::get_language() const {
    return language_;
}

std::string HVCloudPairICP::get_algorithm_display_name() {
    return Tr(language_, "algorithm.display");
}

extern "C" __declspec(dllexport) NodeEngine* CreateInstance() {
    return new HVCloudPairICP();
}

extern "C" __declspec(dllexport) std::string GetInstanceName() {
    return "Point cloud pair ICP";
}

extern "C" __declspec(dllexport) int GetNodeEngineAbiVersion() {
    return NODE_ENGINE_ABI_VERSION;
}

