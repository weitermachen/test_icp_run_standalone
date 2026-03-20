#pragma once

#include <vector>
#include <string>

#include "3d_pilot_public_def.h"
#include "param_meta_data.h"

constexpr int NODE_ENGINE_ABI_VERSION = 2;

/**
 * ************************************************************
 * @brief 节点算法定义
 * 规定，封装进节点的算法都应当满足以下接口功能
 *
 * @author zengxin
 * @data 2025/8/22
 * ************************************************************
 */

class NodeEngine {
public:
	NodeEngine() {};
	virtual ~NodeEngine() {};

public:
    /**
	 * @brief 算法模型初始化
	 *
	 * @return 状态码，初始化成功、失败
	 */
    virtual int init() = 0;

    /**
	 * @brief 算法执行
	 *
	 * @return 状态码，算法是否执行成功、正在执行、失败...
	 */
    virtual int run() = 0;

    /**
	 * @brief 设置算法执行参数
	 *
     * @param_in: 将入参按照算法要求的顺序以 vector 的形式传入
	 * @param_in paramID: 可选参数，指定参数 ID 列表，若不传入则默认按顺序设置所有参数
	 * @return 状态码，参数设置成功、失败
	 */
    virtual int set_algorithm_params(const std::vector<void*>& params, const std::vector<int>& paramID = std::vector<int>()) = 0;

	// 获取算法参数
	virtual std::vector<void*> get_current_params() = 0;

    /**
	 * @brief 返回算法执行结果
	 *
	 * @return 将算法输出结果按顺序以 vector 形式返回
	 */
    virtual std::vector<void*> get_algorithm_result() = 0;

    /**
	 * @brief 算法入参类型返回
	 *
	 * @return 返回算法入参类型列表，e.g. 算法 test(int a, cv::Mat b)，
     *         则该函数入参类型为 {INT, MAT}
	 */
    virtual std::vector<int> get_algorithm_input_params_type() = 0;

    /**
	 * @brief 算法出参类型返回
	 *
	 * @return 返回算法出参类型列表，e.g. 算法 test(int a, cv::Mat b) 返回值为 int, int, cv::Mat，
     *         则该函数返回值为 {INT, INT, MAT}
	 */
    virtual std::vector<int> get_algorithm_output_params_type() = 0;

	/**
	 * @brief 算法入参名称返回
	 *
	 * @return 返回算法入参名称列表，e.g. 算法 test(cv::Mat a, double threshold)，
	 *         则该函数入参名称为 {输入图像, 二值化阈值}
	 */
	virtual std::vector<std::string> get_algorithm_input_params_name() = 0;

	/**
	 * @brief 算法出参名称返回
	 *
	 */
	virtual std::vector<std::string> get_algorithm_output_params_name() = 0;

	/**
	 * @brief 算法入参可绑定状态返回
	 *
	 * @return 返回算法入参默认输入模式提示列表，主库会将 true 解释为默认绑定模式，
	 *         false 解释为默认值编辑模式；它不再作为硬性绑定候选过滤条件。
	 */
	virtual std::vector<bool> get_algorithm_input_params_bindable() = 0;

	/**
	 * @brief 获取算法输入参数的元数据
	 *
	 * @return 返回算法输入参数的元数据列表，包含参数约束信息
	 */
	virtual std::vector<ParamMetadata> get_algorithm_input_params_metadata() = 0;

    /**
	 * @brief 算法执行状态
	 *
	 * @return 算法当前的执行状态：Running、Start、Error、End
	 */
    virtual int get_algorithm_execute_status() = 0;

	/**
	 * @brief 算法错误消息
	 *
	 * @return 算法执行消息
	 */
	virtual std::string get_algorithm_error_message() = 0;

    /**
	 * @brief 算法运行耗时
	 *
	 * @return 返回算法执行时间
	 */
    virtual long get_algorithm_use_time() = 0;

	/**
	 * 算法参数设置状态
	 * 
	 * false - 未设置；true - 已设置
	 */
	virtual bool algorithm_params_setting_status() = 0;

	/**
	 * 算法初始化状态
	 * 
	 * false - 未初始化；true - 已初始化
	 */
	virtual bool algorithm_init_status() = 0;

	// 保存算法参数至json文件
	virtual bool save_params_to_json(const std::string& filePath) = 0;

	// 从json文件加载参数
	virtual bool load_params_from_json(const std::string& filePath) = 0;

	// 获取算法类别
	virtual AlgorithmType get_algorithm_type() = 0;

    // 设置算法实例的语言（UIPilotLanguage）
    virtual void set_language(int language) = 0;

    // 获取算法实例当前语言（UIPilotLanguage）
    virtual int get_language() const = 0;

    // 获取算法显示名称（可本地化）
    virtual std::string get_algorithm_display_name() = 0;

private:
	// 是否设置算法参数
	bool is_set_algorithm_params = false;
	// 是否完成算法初始化
	bool is_algorithm_init = false;
};

/*
在算法实现中需要包含以下导出函数
extern "C" __declspec(dllexport) NodeEngine* CreateInstance();
extern "C" __declspec(dllexport) int GetAlgorithmType();

extern "C" __declspec(dllexport) NodeEngine* CreateInstance() {
	// 每一个 DLL 内部返回自己具体的实现类
	return new LogisticsAlgorithm();
}

extern "C" __declspec(dllexport) std::string GetInstanceName() {
	return "Image filter"; // 告知主程序此 DLL 算法名
}
*/
