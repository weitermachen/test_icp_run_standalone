#pragma once

#include "3d_pilot_data_define.h"

// ============================================================
// 参数约束类型定义
// ============================================================

// 参数约束类型
enum ParamConstraintType {
    CONSTRAINT_NONE = 0,        // 无约束
    CONSTRAINT_RANGE,           // 范围约束（数值类型）
    CONSTRAINT_OPTIONS,         // 可选项约束（枚举类型）
    CONSTRAINT_REGEX,           // 正则表达式约束（字符串）
    CONSTRAINT_FILE_PATH,       // 文件路径约束
    CONSTRAINT_DEPENDENCY       // 依赖约束（依赖其他参数）
};

// 参数分组类型
enum ParamGroupType {
    PARAM_GROUP_BASIC = 0,      // 基础参数
    PARAM_GROUP_ADVANCED        // 高级参数
};

// 参数依赖条件
enum ParamDependencyCondition {
    DEPENDS_ON_EQUALS,          // 等于某值时生效
    DEPENDS_ON_NOT_EQUALS,      // 不等于某值时生效
    DEPENDS_ON_GREATER,         // 大于某值时生效
    DEPENDS_ON_LESS,            // 小于某值时生效
    DEPENDS_ON_IN_LIST,         // 在列表中时生效
    DEPENDS_ON_NOT_IN_LIST      // 不在列表中时生效
};

// ============================================================
// 参数元数据结构体
// ============================================================

// 数值范围约束
struct RangeConstraint {
    double min_value;           // 最小值
    double max_value;           // 最大值
    double default_value;       // 默认值

    RangeConstraint()
        : min_value(0.0), max_value(100.0), default_value(0.0) {
    }

    RangeConstraint(double min, double max, double def = 0.0)
        : min_value(min), max_value(max), default_value(def) {
    }
};

// 可选项约束
struct OptionsConstraint {
    std::vector<int> option_values;           // 可选值
    std::vector<std::string> option_labels;   // 显示标签
    int default_index;                        // 默认选项索引

    OptionsConstraint() : default_index(0) {}

    void AddOption(int value, const std::string& label) {
        option_values.push_back(value);
        option_labels.push_back(label);
    }
};

// 参数依赖关系
struct ParamDependency {
    int depend_on_param_index;              // 依赖的参数索引
    ParamDependencyCondition condition;      // 依赖条件
    std::vector<std::string> condition_values; // 条件值列表

    ParamDependency()
        : depend_on_param_index(-1), condition(DEPENDS_ON_EQUALS) {
    }

    ParamDependency(int param_idx, ParamDependencyCondition cond,
        const std::vector<std::string>& values = {})
        : depend_on_param_index(param_idx), condition(cond), condition_values(values) {
    }
};

// 完整的参数元数据
struct ParamMetadata {
    // 基本信息
    std::string param_name;                 // 参数名称
    std::string param_description;          // 参数描述
    int param_type;                         // 参数类型（HV_INT, HV_STRING等）
    ParamGroupType param_group;             // 参数分组（基础/高级）

    // 约束信息
    ParamConstraintType constraint_type;    // 约束类型
    RangeConstraint range_constraint;       // 范围约束
    OptionsConstraint options_constraint;   // 可选项约束
    std::string regex_pattern;              // 正则表达式
    std::string file_filter;                // 文件过滤器（如 "*.jpg;*.png"）

    // 依赖关系
    std::vector<ParamDependency> dependencies; // 依赖列表

    ParamMetadata()
        : param_type(HV_INT)
        , param_group(PARAM_GROUP_BASIC)
        , constraint_type(CONSTRAINT_NONE) {
    }
};
