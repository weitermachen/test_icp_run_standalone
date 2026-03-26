# test_icp_run_standalone

一个可独立编译运行的点云配准示例工程，核心目标是提供“可配置、可复用、可评估”的双点云 ICP 流水线。

本项目从原工程中抽离，保留了完整的两阶段配准能力（Coarse + Refine）、多尺度迭代、评估指标计算和结果落盘流程，便于在外部环境快速复现算法效果。

## 1. 项目功能总览

### 1.1 支持的输入格式

- TXT 点云（每行至少 3 个数值，支持空格、Tab、逗号、分号分隔）
- PCD 点云
- PLY 点云

### 1.2 预处理能力

- TXT 点云自动缓存为 `.cache.pcd`，加速重复运行
- 体素降采样（可用于 coarse 阶段）
- 统计离群点滤波（source/target 去噪）
- 源点云镜像（x/y/z 轴取反）

### 1.3 配准能力

- 支持算法：ICP、GICP、LP-ICP（point-to-plane）
- 支持两阶段流程：
  - Stage-1 Coarse（通常先做降采样）
  - Stage-2 Refine（默认用原始点云）
- 支持 `initial_transform_json` 初始位姿
- 粗配准结果可自动串联为精配准初值

### 1.4 评估能力

- 基于最近邻距离评估配准结果
- 输出指标：mean、rmse、max、sampled、stride
- 可按阶段开启：coarse / refine / both

### 1.5 输出能力

- 变换后的 source 点云（PLY）
- source->target 的最终刚体变换 JSON
- 下一次可复用的 `initial_transform_json`

### 1.6 多位姿联合优化能力（新增）

- 读取 `result` 目录下多个 `registration_result.json`
- 读取 `data` 中对应 `left_*` 与 `right_*` / `rigth_*` 点云
- 以多个 `registration_result.json` 作为初值进行全局联合优化（类似 bundle adjustment）
- 可通过 `log.txt` 中 `Stage-2 Refine Evaluation` 的 RMSE 阈值过滤候选

## 2. 算法流程说明

主流程位于 `src/main.cpp`，整体顺序如下：

0. 根据 `run_mode` 选择功能（`registration` 或 `fusion`）

1. 读取配置 JSON（默认 `./config/icp_run_config.json`）
2. 加载 source/target 点云
3. 对 source 按配置执行镜像
4. 执行 Stage-1 Coarse（可选）
5. 执行 Stage-2 Refine（可选）
6. 可选计算评估指标
7. 保存最终点云与变换结果

说明：

- `run_mode=registration`：执行原有配准功能
- `run_mode=fusion`：执行融合功能，不进入原始配准流程
- 两种功能互斥，满足隔离要求

### 2.1 两阶段流水线

- `pipeline.run_coarse=true` 时运行 Stage-1
- `pipeline.run_refine=true` 时运行 Stage-2
- 两者不能同时为 false
- 当两阶段都开启时，Stage-2 会强制使用 Stage-1 输出的变换作为初值

### 2.2 多尺度求解策略

核心实现位于 `src/ICPCore.cpp`，每次配准会走 3 个尺度：

- coarse level
- middle level
- fine level

说明：

- 第一个尺度会强制使用 ICP 作为稳健初始化
- 后续尺度按配置算法运行（GICP 或 LP-ICP）
- 当点数过少（例如 <30）时，GICP/LP-ICP 会跳过当前尺度

## 3. 目录结构

```text
test_icp_run_standalone/
├─ CMakeLists.txt
├─ CMakePresets.json
├─ README.md
├─ config/
│  └─ icp_run_config.json
├─ data/
│  └─ README.txt
├─ include/
│  ├─ HVCloudPairICP.h
│  ├─ ICPCore.h
│  ├─ HVUtils.h
│  ├─ HVI18n.h
│  └─ public/...
└─ src/
   ├─ main.cpp
   ├─ HVCloudPairICP.cpp
   ├─ ICPCore.cpp
   ├─ HVUtils.cpp
   └─ HVI18n.cpp
```

## 4. 核心模块说明

### 4.1 src/main.cpp

命令行入口，负责：

- 配置解析
- 点云读取与预处理
- 两阶段调度
- 评估计算
- 结果落盘

### 4.2 src/HVCloudPairICP.cpp

面向业务节点的封装层，提供：

- 统一参数接口（void* 参数表）
- 运行状态与错误信息
- 结果 JSON 组装
- 双语文案（中文/英文）

### 4.3 src/ICPCore.cpp

真正的配准核心：

- 初始变换解析
- 多尺度迭代
- ICP / GICP / LP-ICP 执行
- 输出变换、fitness、rmse、融合点云

### 4.4 src/HVUtils.cpp

通用工具能力：

- HV / Open3D / PCL 点云互转
- 2D 图像与 ROI 操作
- 3D 几何裁剪（Box/RotatedBox）

### 4.5 src/HVI18n.cpp

国际化文案映射及格式化，服务于算法节点对外输出。

## 5. 环境与依赖

- Windows 10/11
- Visual Studio 2022（MSVC）
- CMake 3.16+
- OpenCV 4.8.x
- Open3D 0.17.x
- PCL 1.15.x
- Eigen3 3.4+

## 6. 构建指南

### 6.1 推荐：使用 CMake Presets

项目已提供：

- `release` 预设（输出到 `build/release`）
- `debug` 预设（输出到 `build/debug`）

```powershell
# Release
cmake --preset release
cmake --build --preset release --target test_icp_run_standalone

# Debug
cmake --preset debug
cmake --build --preset debug --target test_icp_run_standalone
```

### 6.2 手动配置（可选）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR="D:/path/to/opencv/build/x64/vc16/lib" `
  -DOpen3D_DIR="D:/path/to/open3d/CMake" `
  -DPCL_DIR="D:/path/to/pcl/cmake" `
  -DEigen3_DIR="D:/path/to/eigen/share/eigen3/cmake"

cmake --build build --config Release --target test_icp_run_standalone
```

### 6.3 运行时 DLL 处理

构建后会自动把 OpenCV / Open3D 的 DLL 拷贝到可执行文件目录：

- OpenCV：支持从 `OpenCV_DIR` 自动推导 bin 目录，找不到时可手动设置 `OpenCV_DLL_DIR`
- Open3D：从 `Open3D_DIR` 的上级目录推导 `bin`

如果仍提示缺 DLL，可检查：

1. 输出目录是否存在 `opencv_world*.dll`
2. 输出目录是否存在 `Open3D.dll`
3. 第三方版本与编译架构是否一致（x64 / Debug / Release）

## 7. 运行指南

### 7.1 命令行

```powershell
# Release
./build/release/Release/test_icp_run_standalone.exe ./config/icp_run_config.json

# Debug
./build/debug/Debug/test_icp_run_standalone.exe ./config/icp_run_config.json
```

融合模式使用同一可执行文件，只需把配置中的 `run_mode` 改为 `fusion`。

程序会优先读取命令行参数中的配置路径；未传参时使用默认路径 `./config/icp_run_config.json`。

### 7.2 退出码

- `0`：运行成功
- `-1`：失败（配置错误、读点云失败、配准失败、保存失败等）

## 8. 配置文件详解

示例文件：`config/icp_run_config.json`

### 8.1 完整示例

```json
{
  "run_mode": "registration",
  "input": {
    "source_cloud": "./data/source.txt",
    "target_cloud": "./data/target.txt"
  },
  "preprocess": {
    "enable_txt_cache": true,
    "filter_leaf_size": 0.6,
    "noise_filter": {
      "enabled": true,
      "mean_k": 20,
      "stddev_mul_thresh": 1.0
    },
    "mirror_axes": "x"
  },
  "pipeline": {
    "run_coarse": true,
    "run_refine": true
  },
  "coarse_registration": {
    "method": 1,
    "voxel_size": 0.0,
    "max_correspondence_distance": 8.0,
    "max_iterations": 120,
    "use_initial_guess": false,
    "initial_transform_json": ""
  },
  "refine_registration": {
    "method": 2,
    "voxel_size": 0.3,
    "max_correspondence_distance": 2.5,
    "max_iterations": 140,
    "use_initial_guess": true,
    "initial_transform_json": ""
  },
  "evaluation": {
    "enabled": true,
    "stages": ["coarse", "refine"],
    "max_sample_points": 50000
  },
  "output": {
    "result_dir": "./result",
    "transformed_source_cloud": "transformed_source_by_registration.ply",
    "result_json": "registration_result.json",
    "initial_transform_json": "initial_transform_for_icp.json"
  },
  "fusion": {
    "candidate_result_dir": "./result",
    "data_dir": "./data",
    "output_json": "fused_registration_result.json",
    "max_sample_points_per_pair": 50000,
    "trim_ratio": 0.1,
    "max_refine_rmse": 0.5,
    "optimization_max_iterations": 40,
    "optimization_rotation_step_deg": 0.5,
    "optimization_translation_step": 0.5,
    "optimization_min_rotation_step_deg": 0.005,
    "optimization_min_translation_step": 0.01
  }
}
```

### 8.2 run_mode

| 字段 | 类型 | 说明 |
|---|---|---|
| run_mode | string | `registration` 或 `fusion` |

说明：

- `registration`：原始标定流程
- `fusion`：多位姿标定联合优化流程
- 两种模式互斥，不能同时启用

### 8.3 input（仅 registration 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| source_cloud | string | 源点云路径（支持相对路径） |
| target_cloud | string | 目标点云路径（支持相对路径） |

说明：

- 相对路径按配置文件所在目录解析
- source/target 不能为空

### 8.4 preprocess（仅 registration 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| enable_txt_cache | bool | 是否启用 TXT->PCD 缓存 |
| filter_leaf_size | number | coarse 阶段体素降采样尺寸 |
| noise_filter | object | 统计离群点滤波配置 |
| mirror_axes | string/array/object | 对 source 做轴向镜像 |

`mirror_axes` 支持三种写法：

- 字符串：`"x"`、`"xy"`、`"x,y,z"`
- 数组：`["x", "z"]`
- 对象：`{"x": true, "y": false, "z": true}`

`noise_filter` 字段：

- `enabled`：是否开启去噪
- `mean_k`：邻域点数
- `stddev_mul_thresh`：标准差倍数阈值

### 8.5 pipeline（仅 registration 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| run_coarse | bool | 是否执行 Stage-1 |
| run_refine | bool | 是否执行 Stage-2 |

约束：两者不能同时为 false。

### 8.6 coarse_registration / refine_registration（仅 registration 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| method | int | 0=ICP, 1=GICP, 2=LP-ICP |
| voxel_size | number | 配准内部降采样尺寸，<=0 表示关闭 |
| max_correspondence_distance | number | 最大对应点距离 |
| max_iterations | int | 最大迭代次数 |
| use_initial_guess | bool | 是否使用初值 |
| initial_transform_json | string/object | 初始位姿，支持 JSON 字符串或对象 |

`initial_transform_json` 格式示例：

```json
{
  "matrix": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

### 8.7 evaluation（仅 registration 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| enabled | bool | 是否启用评估 |
| stages | string/array | 评估阶段：coarse/refine/both/all |
| max_sample_points | int | 最大采样点数（用于降采样评估） |

`stages` 兼容：

- `coarse` / `stage1` / `stage-1`
- `refine` / `stage2` / `stage-2`
- `both` / `all`

### 8.8 output（仅 registration 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| result_dir | string | 输出目录 |
| transformed_source_cloud | string | 变换后 source 点云文件名（PLY） |
| result_json | string | 最终变换 JSON 文件名 |
| initial_transform_json | string | 复用初值 JSON 文件名 |

### 8.9 fusion（仅 fusion 模式）

| 字段 | 类型 | 说明 |
|---|---|---|
| candidate_result_dir | string | 候选标定目录（递归查找 `registration_result.json`） |
| data_dir | string | 原始点云目录（匹配 `left_*` 与 `right_*`/`rigth_*`） |
| output_json | string | 融合输出 JSON 文件名 |
| max_sample_points_per_pair | int | 每个位姿评估最大采样点数 |
| trim_ratio | number | 截尾比例，范围 `[0,0.5)` |
| max_refine_rmse | number | Refine Evaluation RMSE 阈值，`<=0` 表示关闭 |
| optimization_max_iterations | int | 联合优化最大迭代次数 |
| optimization_rotation_step_deg | number | 联合优化旋转初始步长（度） |
| optimization_translation_step | number | 联合优化平移初始步长 |
| optimization_min_rotation_step_deg | number | 联合优化最小旋转步长（度） |
| optimization_min_translation_step | number | 联合优化最小平移步长 |

融合策略：

- 先从每个候选目录的 `log.txt` 读取 `Stage-2 Refine Evaluation` 的 `rmse`
- 若 `rmse > max_refine_rmse`，则该候选被跳过
- 对剩余候选，以其刚体变换作为初值，在全部位姿数据上优化统一全局目标函数
- 从多初值优化结果中选择全局 RMSE 最小的最终结果

## 9. 输出结果说明

### 9.1 transformed_source_cloud

- 文件格式：PLY（二进制）
- 内容：source 经过最终 source->target 刚体变换后的点云

### 9.2 result_json

示例结构：

```json
{
  "transform_source_to_target": {
    "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
    "t": [tx, ty, tz]
  }
}
```

### 9.3 initial_transform_json

用于下次配准直接作为 `initial_transform_json` 输入，减少收敛时间、提升稳定性。

### 9.4 fused_registration_result.json（fusion 模式）

示例结构：

```json
{
  "transform_source_to_target": {
    "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
    "t": [tx, ty, tz]
  },
  "fusion_meta": {
    "method": "global_bundle_adjustment_like_multi_start",
    "candidate_count": 6,
    "best_initial_candidate": "result/3/registration_result.json",
    "global_rmse": 0.123456,
    "max_refine_rmse": 0.5,
    "candidate_scores": []
  }
}
```

## 10. 常见问题

### 10.1 运行时报缺少 OpenCV/Open3D DLL

检查输出目录是否存在：

- `opencv_world*.dll`
- `Open3D.dll`

如果没有：

1. 确认 `OpenCV_DIR` / `Open3D_DIR` 配置正确
2. 手动设置 `OpenCV_DLL_DIR`
3. 重新执行 configure + build

### 10.2 Debug 与 Release 混用导致异常

请确保：

- 使用对应 preset 构建
- 运行对应配置目录下的 exe
- 第三方库版本与编译配置匹配（尤其是 OpenCV world 的 d 后缀）

### 10.3 refine 阶段提示缺少初值

当 `run_coarse=false` 且 `refine_registration.use_initial_guess=true` 时，必须在
`refine_registration.initial_transform_json` 中提供有效 4x4 初始变换。

## 11. 二次开发建议

- 作为可执行工具：继续扩展 `main.cpp` 的输入输出协议
- 作为算法节点：使用 `HVCloudPairICP` 的参数接口嵌入现有流程
- 作为库能力：复用 `ICPCore` 的多尺度配准与初值解析逻辑

## 12. 许可证与说明

本目录为内部工程拆分版，主要用于算法复现、演示和交付集成。
若需对外发布，请按你的组织规范补充许可证、版本记录和第三方依赖声明。
