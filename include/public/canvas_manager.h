#pragma once

#include "public/3d_pilot_public_def.h"
#include "public/param_meta_data.h"

struct CanvasInfoList
{
	std::vector<NodeInfo> node_info_list_;
	std::vector<NodeParamBindInfo> node_param_bind_info_list_;
};

class GlobalVariableManager;

class HV3DPILOT_API CanvasManager {
public:
    CanvasManager();
    ~CanvasManager();

    // 设置界面语言（UIPilotLanguage）
    int SetLanguage(int language);
    // 获取当前界面语言（UIPilotLanguage）
    int GetLanguage() const;

    // 设置算法插件路径
    bool SetPluginPath(const std::string& path);

    // 创建画布，成功返回画布 id（>= 0），失败返回错误码（< 0）
    int CreateCanvas();

    // 获取可用节点列表
	const std::vector<NodeInfo>& GetAvailableNodeList() const;

    // 获取可用节点列表（按类别）
    const std::map<AlgorithmType, std::unordered_map<std::string, NodeInfo>>& GetAvailableNodeListByCategory() const;

    // 更新指定画布
    int UpdateCanvas(const int canvas_id);
    
    // 删除指定画布
    int CanvasDelete(const int canvas_id);
    
    // 保存指定画布
    int CanvasSave(const int canvas_id, const std::string& filePath);
    
    // 加载保存的画布，返回节点信息列表
    CanvasInfo CanvasLoad(const std::string& filePath);

    // 运行指定画布
    int CanvasRun(const int canvas_id);

    // 运行指定画布中的单个节点（不自动运行上游节点）
    int CanvasRunNode(const int canvas_id, const NodeIDInt node_id);

    // 输出执行画布指定节点的结果
    std::vector<void*> GetNodeResult(const int canvas_id, const NodeIDInt node_id);

    // 获取全部全局变量
    std::vector<NodeResultGlobalInfo> GetGlobalVariables();
    // 获取单个全局变量
    NodeResultGlobalInfo GetGlobalVariable(const int var_id);

    // 创建全局变量（返回变量id）
    int CreateGlobalVariable(const std::string& var_name, int var_type, void* value = nullptr, const std::string& var_comment = "");
    // 更新全局变量（名称/类型/值/注释）
    int UpdateGlobalVariable(const int var_id, const std::string& var_name, const int var_type, void* value, const std::string& var_comment = "");
    // 删除全局变量
    int DeleteGlobalVariable(const int var_id);
    // 绑定节点输入参数到全局变量
    int BindNodeParamToGlobalVariable(const int var_id, const int canvas_id, const NodeIDInt node_id, const ParamIDInt input_param_id);
    // 获取节点某个输入参数可绑定的全局变量列表
    std::vector<NodeResultGlobalInfo> GetNodeParamBindableGlobalVariables(
        const int canvas_id,
        const NodeIDInt node_id,
        const ParamIDInt input_param_id);

		// 获取指定画布、指定节点的运行消息
	std::string GetNodeRunMessage(const int canvas_id, const NodeIDInt node_id);

    // 为指定画布新增节点
	// 返回值为节点 id，失败返回 -1
    NodeIDInt CanvasAddNode(const int canvas_id, const int node_info_id, const std::string node_name, const HVPoint3D& node_pos = HVPoint3D());

    // 为指定画布新增节点
    // 返回值为节点 id，失败返回 -1
    NodeIDInt CanvasAddNode(const int canvas_id, const AlgorithmType type, const std::string& algorithm_name, const std::string node_name, const HVPoint3D& node_pos = HVPoint3D());

	// 获取当前画布所有节点 ID 列表(默认进行拓扑排序)
	std::vector<NodeIDInt> GetCanvasAllNodeIDs(const int canvas_id, bool topologySort = true);

    // 设置指定画布的指定节点输入参数(不绑定的参数)
    int CanvasSetNodeParam(const int canvas_id, const NodeIDInt node_id, const std::vector<void*>& params, const std::vector<ParamIDInt>& paramID);

    // 设置指定输入参数的当前输入模式（值模式/绑定模式）
    int SetNodeParamInputMode(const int canvas_id, const NodeIDInt node_id, const ParamIDInt input_param_id, const ParamInputMode input_mode);

    // 清除指定输入参数的节点绑定与全局变量绑定，切回值模式
    int ClearNodeParamBinding(const int canvas_id, const NodeIDInt node_id, const ParamIDInt input_param_id);

    // 设置指定画布、指定节点的位置
    int CanvasSetNodePos(const int canvas_id, const NodeIDInt node_id, const HVPoint3D& node_pos);

    // 重命名指定画布中的节点（节点别名）
    int CanvasRenameNode(const int canvas_id, const NodeIDInt node_id, const std::string& node_alias);

	// 获取指定画布的指定节点输入参数
	std::vector<void*> CanvasGetNodeParam(const int canvas_id, const NodeIDInt node_id);

	// 连接节点 src_node_id -> dst_node_id
	int CanvasConnectNode(const int canvas_id, const NodeIDInt src_node_id, const NodeIDInt dst_node_id);

	// 断开节点连接 src_node_id -> dst_node_id
	int CanvasDisconnectNode(const int canvas_id, const NodeIDInt src_node_id, const NodeIDInt dst_node_id);

    // 删除指定画布的指定节点
    int CanvasDeleteNode(const int canvas_id, const NodeIDInt node_id);

    // 获取当前节点对应参数id(-1时返回全部参数)可绑定的参数信息(需先进行节点连接)
	std::vector<NodeParamBindInfo> GetCanvasNodeBindableParams(const int canvas_id, const NodeIDInt node_id, const ParamIDInt node_param_id = -1);

    // 为指定画布指定节点绑定参数信息
    int CanvasNodeBind(const int canvas_id, const NodeIDInt node_id, const std::vector<NodeParamBindInfo>& node_param_config);

	// 返回指定画布、指定节点的输入参数类型
    std::vector<int> GetCanvasNodeInputParamType(const int canvas_id, const NodeIDInt node_id);

	// 返回指定画布、指定节点的输出结果类型
	std::vector<int> GetCanvasNodeResultType(const int canvas_id, const NodeIDInt node_id);

    // 返回当前画布信息
	CanvasInfo GetCanvasInfo(const int canvas_id);

    // 返回指定画布、指定节点参数信息（含 input_params_metadata_）
    ParamInfo GetCanvasNodeParamInfo(const int canvas_id, const NodeIDInt node_id);

    // 根据节点id获取当前节点信息
    NodeInfo GetCanvasNodeInfo(const int canvas_id, const NodeIDInt node_id);

    // 获取指定画布、指定节点的输入参数元数据
    std::vector<ParamMetadata> GetNodeInputParamsMetadata(const int canvas_id, const NodeIDInt node_id);

private:
    // 可分配的 id 最大值
    const int kAllocIdCapacity = 1000;

    // 为新增画布分配 id
    // 返回分配后的画布 id
    int AllocCanvasId();

    // 节点工厂（插件加载 & 节点类型注册 & 实例创建）
    std::unique_ptr<NodeFactory> node_factory_;

    std::unique_ptr<GlobalVariableManager> global_variable_manager_;

    int language_;

    // 画布集合
    std::map<int, std::shared_ptr<Canvas>> canvas_table_;
};
