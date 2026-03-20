#pragma once

#include "node_engine.h"
#include "3d_pliot_error.h"
#include "3d_pilot_public_def.h"
#include "param_meta_data.h"

#include <memory>
#include <string>
#include <vector>

class HVCloudPairICP : public NodeEngine {
public:
    HVCloudPairICP();
    ~HVCloudPairICP() override;

    int init() override;
    int run() override;

    // 0: source cloud (reference / base)
    // 1: target cloud (to be aligned to source)
    // 2: method (0: ICP, 1: GICP, 2: LP-ICP / point-to-plane ICP)
    // 3: voxel size
    // 4: max correspondence distance
    // 5: max iterations
    // 6: use initial guess
    // 7: initial transform json
    int set_algorithm_params(const std::vector<void*>& params,
                             const std::vector<int>& paramID = std::vector<int>()) override;

    std::vector<void*> get_current_params() override;
    std::vector<void*> get_algorithm_result() override;

    std::vector<int> get_algorithm_input_params_type() override;
    std::vector<int> get_algorithm_output_params_type() override;

    std::vector<std::string> get_algorithm_input_params_name() override;
    std::vector<std::string> get_algorithm_output_params_name() override;
    std::vector<bool> get_algorithm_input_params_bindable() override;
    std::vector<ParamMetadata> get_algorithm_input_params_metadata() override;

    int get_algorithm_execute_status() override;
    std::string get_algorithm_error_message() override;
    long get_algorithm_use_time() override;

    bool algorithm_params_setting_status() override;
    bool algorithm_init_status() override;

    bool save_params_to_json(const std::string& filePath) override;
    bool load_params_from_json(const std::string& filePath) override;

    AlgorithmType get_algorithm_type() override;

    void set_language(int language) override;
    int get_language() const override;
    std::string get_algorithm_display_name() override;

private:
    std::shared_ptr<HVPointCloud> sourceCloud;
    std::shared_ptr<HVPointCloud> targetCloud;

    std::shared_ptr<HVPointCloud> alignedTargetCloud;
    std::shared_ptr<HVPointCloud> mergedCloud;
    std::string resultJson;

    int method = 0; // 0: ICP, 1: GICP, 2: LP-ICP / point-to-plane ICP
    double voxel_size = 0.0;
    double max_correspondence_distance = 1.0;
    int max_iterations = 50;
    bool use_initial_guess = false;
    std::string initial_transform_json;

    int execute_status = NODE_STATUS_NOT_RUN;
    long run_time = 0;
    std::string error_msg;
    int language_ = static_cast<int>(UIPilotLanguage::ZH_CN);
};

extern "C" __declspec(dllexport) NodeEngine* CreateInstance();
extern "C" __declspec(dllexport) std::string GetInstanceName();
extern "C" __declspec(dllexport) int GetNodeEngineAbiVersion();
