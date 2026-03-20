#pragma once

#include <windows.h>

#undef min
#undef max

#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>

#include "3d_pilot_data_define.h"
#include "3d_pliot_error.h"
#include "geometry_def.h"
#include "param_meta_data.h"

#ifdef HV_3DPILOT_EXPORTS
    #define HV3DPILOT_API __declspec(dllexport)
#else
    #define HV3DPILOT_API __declspec(dllimport)
#endif

struct Node;
class Canvas;
class NodeFactory;

using NodeIDInt = int;
using ParamIDInt = int;

// 算法类别定义
enum class AlgorithmType
{
    Capture = 0,
    ImageProcess = 1,
    PointCloudProcess = 2,
    Recognition = 3,
    Measurement = 4,
    Localization = 5,
    Calibration = 6
};

enum class UIPilotLanguage
{
    ZH_CN = 0,
    EN_US = 1
};

enum class ParamInputMode
{
    VALUE = 0,
    BIND = 1
};

enum class ParamBindingSourceType
{
    NONE = 0,
    NODE = 1,
    GLOBAL_VARIABLE = 2
};

// 节点绑定信息结构体定义
struct NodeParamBindInfo
{
    // 绑定节点 id
    NodeIDInt bind_node_id_;
    // 绑定节点对应的结果 id
    ParamIDInt bind_node_result_id_;
    // 当前节点的对应参数id
    ParamIDInt bind_cur_node_param_id_;

    NodeParamBindInfo()
        : bind_node_id_(-1)
        , bind_node_result_id_(-1)
        , bind_cur_node_param_id_(-1)
    {
    };

    NodeParamBindInfo(NodeIDInt bind_node_id, ParamIDInt bind_node_result_id, ParamIDInt bind_cur_node_param_id)
        : bind_node_id_(bind_node_id),
        bind_node_result_id_(bind_node_result_id),
        bind_cur_node_param_id_(bind_cur_node_param_id)
    {
    }
};

// 节点信息结构体定义
struct NodeInfo
{
    AlgorithmType category_; // 算法类别
    std::string algorithm_name_;// 算法名称
    std::string algorithm_display_name_; // 算法显示名称（可本地化）
    std::string icon_full_path_; // 算法图标绝对路径
    int input_params_num;// 输入参数个数
    std::vector<std::string> input_params_name_;// 输入参数名称
	std::vector<int> input_params_type_;// 输入参数类型
    std::vector<bool> input_params_bindable_; // 输入参数是否可绑定
    std::vector<ParamInputMode> input_params_default_input_mode_; // 输入参数默认输入模式
    std::vector<ParamMetadata> input_params_metadata_; // 输入参数元数据
    int output_params_num;// 输出参数个数
    std::vector<std::string> output_params_name_;// 输出参数名称
	std::vector<int> output_params_type_;// 输出参数类型

	NodeIDInt node_id_; // 节点 id
	std::string node_name_; // 节点名称
    HVPoint3D node_pos; // 节点位置
	std::vector<NodeParamBindInfo> register_node_; // 绑定节点信息
	float running_time_; // 节点运行时间
	int status_; // 节点状态

    NodeInfo()
    {
        category_ = AlgorithmType::Capture;
        algorithm_name_ = "";
        algorithm_display_name_ = "";
        icon_full_path_ = "";
        input_params_num = 0;
        output_params_num = 0;
        node_id_ = -1;
        node_name_ = "";
        running_time_ = 0;
        status_ = NODE_STATUS_NOT_RUN;
    }
};

struct ConnectionId
{
    NodeIDInt outNodeId;
    NodeIDInt inNodeId;

    ConnectionId() : outNodeId(-1), inNodeId(-1) {}
    ConnectionId(NodeIDInt out, NodeIDInt in) : outNodeId(out), inNodeId(in) {}

    bool operator==(const ConnectionId& other) const {
        return outNodeId == other.outNodeId && inNodeId == other.inNodeId;
    }
};

namespace std {
    template <>
    struct hash<ConnectionId> {
        size_t operator()(const ConnectionId& c) const {
            return hash<int>()(c.outNodeId) ^ (hash<int>()(c.inNodeId) << 16);
        }
    };
}

struct NodeGeometryData
{
    HVPoint3D pos;
    
    NodeGeometryData()
    {
        pos.x = 0;
        pos.y = 0;
	};

    NodeGeometryData(double x, double y)
    {
        pos.x = x;
        pos.y = y;
    };
};

struct CanvasInfo
{
    int canvas_id_; // 画布 id
	std::vector<std::vector<NodeInfo>> nodes; // 画布包含的节点信息
	std::vector<std::vector<NodeIDInt>> node_connect; // 节点连接关系

    std::unordered_map<NodeIDInt, std::unique_ptr<NodeInfo>> models;
    std::unordered_set<ConnectionId> connectivity;
    std::unordered_map<NodeIDInt, NodeGeometryData> nodeGeometryData;
};

struct ParamInfo
{
    std::vector<int> input_params_type_;// 输入参数类型
    std::vector<std::string> input_params_name_;// 输入参数名称
    std::vector<bool> input_params_is_bind_; // 输入参数是否绑定
    std::vector<bool> input_params_is_bindable_; // 输入参数是否可绑定
    std::vector<ParamInputMode> input_params_default_input_mode_; // 输入参数默认输入模式
    std::vector<ParamInputMode> input_params_current_input_mode_; // 输入参数当前输入模式
    std::vector<ParamBindingSourceType> input_params_binding_source_type_; // 输入参数当前绑定来源
    std::vector<int> input_params_bound_global_var_id_; // 输入参数绑定的全局变量 id，未绑定为 -1
    std::vector<ParamGroupType> input_params_group_; // 输入参数分组（基础/高级）
	std::vector<NodeParamBindInfo> input_params_bind_info_; // 输入参数绑定信息
	std::vector<void*> input_params_value_;// 输入参数值

    std::vector<int> output_params_type_;// 输出参数类型
    std::vector<std::string> output_params_name_;// 输出参数名称
    std::vector<void*> output_params_value_;// 输出参数值
    std::vector<ParamMetadata> input_params_metadata_; // 输入参数元数据
};

struct NodeResultGlobalInfo
{
    int var_id_;                                    // 全局变量唯一id

    // Row-style global variable fields.
    std::string var_name_;                          // 名称
    std::string var_comment_;                       // 注释
    int var_type_;                                  // 类型
    void* current_value_;                           // 当前值

    NodeResultGlobalInfo()
        : var_id_(-1)
        , var_type_(-1)
        , current_value_(nullptr)
    {
    }
};
