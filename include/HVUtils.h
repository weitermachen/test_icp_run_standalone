// Author: weitermachen
// Time: 2026-03-24

#pragma once

#include "3d_pilot_public_def.h"
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <json.hpp>

#include <fstream>
#include <chrono>

// 辅助函数：向JSON参数列表中添加参数
template <typename T>
void add_param(nlohmann::json& params_json, const std::string& name, int type, const T& value) {
    nlohmann::json param_json;
    param_json["name"] = name;
    param_json["type"] = type;
    param_json["value"] = value;
    params_json.push_back(param_json);
};

// 将void*转为对应参数类型指针
template <typename T>
T cast_param(const std::vector<void*>& params, int param_id)
{
    if (param_id < 0 || param_id > params.size() || params[param_id] == nullptr)
    {
        return T();
    }
    return *static_cast<T*>(params[param_id]);
}

// 将void*转为对应参数类型智能指针
template <typename T>
std::shared_ptr<T> cast_param_sharedPtr(const std::vector<void*>& params, int param_id)
{
    if (param_id < 0 || param_id > params.size() || params[param_id] == nullptr)
    {
        return nullptr;
    }
    return *static_cast<std::shared_ptr<T>*>(params[param_id]);
}

class ImageConverter {
public:
    /**
     * @brief 将 ImageDataInfo2D 转换为 cv::Mat
     * @param input 自定义2D图像结构体
     * @return OpenCV Mat
     */
    static cv::Mat ToMat(const ImageDataInfo2D& input);

    /**
     * @brief 将 cv::Mat 转换为 ImageDataInfo2D
     * @param input OpenCV Mat
     * @return 自定义2D图像结构体
     */
    static ImageDataInfo2D FromMat(const cv::Mat& input);

    /**
     * @brief 将 ImageDataInfo2D 转换为 cv::Mat (零拷贝，共享数据)
     * @param input 自定义2D图像结构体
     * @return OpenCV Mat (与输入共享数据)
     * @warning 返回的Mat与输入共享数据，修改会影响原始数据
     */
    static cv::Mat ToMatShallow(ImageDataInfo2D& input);

    /**
     * @brief 将 cv::Mat 转换为 ImageDataInfo2D (零拷贝，共享数据)
     * @param input OpenCV Mat
     * @return 自定义2D图像结构体 (与输入共享数据)
     * @warning 返回的ImageDataInfo2D与输入共享数据，需要手动管理生命周期
     */
    static ImageDataInfo2D FromMatShallow(cv::Mat& input);

private:
    /**
     * @brief 将OpenCV类型转换为通道数
     * @param cv_type OpenCV类型 (CV_8UC1, CV_8UC3等)
     * @return 通道数
     */
    static int GetChannelsFromCvType(int cv_type);

    /**
     * @brief 将通道数转换为OpenCV类型
     * @param channels 通道数
     * @return OpenCV类型 (CV_8UC1, CV_8UC3等)
     */
    static int GetCvTypeFromChannels(int channels);
};

// HVPointCloud转Open3D/PCL点云
class PointCloudConverter {
public:
    /**
     * @brief 将 HVPointCloud 转换为 Open3D PointCloud
     * @param input 自定义点云
     * @return Open3D 点云智能指针
     */
    static std::shared_ptr<open3d::geometry::PointCloud> ToOpen3D(const HVPointCloud& input);

    /**
     * @brief 将 Open3D PointCloud 转换为 HVPointCloud
     * @param input Open3D 点云
     * @return 自定义点云结构
     */
    static HVPointCloud FromOpen3D(const open3d::geometry::PointCloud& input);

    /**
     * @brief 将 HVPointCloud 转换为 PCL PointCloud
     * @param input 自定义点云
     * @return PCL 点云智能指针
     */
    static pcl::PointCloud<pcl::PointXYZ>::Ptr ToPCL(const HVPointCloud& input);

    /**
     * @brief 将 PCL PointCloud 转换为 HVPointCloud
     * @param input PCL 点云
     * @return 自定义点云结构
     */
    static HVPointCloud FromPCL(const pcl::PointCloud<pcl::PointXYZ>& input);

    /**
     * @brief 将 Open3D PointCloud 转换为 PCL PointCloud
     * @param input Open3D 点云
     * @return PCL 点云智能指针
     */
    static pcl::PointCloud<pcl::PointXYZ>::Ptr Open3DToPCL(const open3d::geometry::PointCloud& input);

    /**
     * @brief 将 PCL PointCloud 转换为 Open3D PointCloud
     * @param input PCL 点云
     * @return Open3D 点云智能指针
     */
    static std::shared_ptr<open3d::geometry::PointCloud> PCLToOpen3D(const pcl::PointCloud<pcl::PointXYZ>& input);
};

// ROI 坐标中，2D 几何使用图像像素坐标，原点在左上角。
// 旋转矩形角度在图像坐标系下按顺时针为正。
bool IsValidRoiInfo(const HVGeometryInfo& roi);
bool BuildRoiMask(const HVGeometryInfo& roi, int image_width, int image_height, cv::Mat& mask);
bool BuildMaskedImageFromRoi(const HVGeometryInfo& roi, const ImageDataInfo2D& src_image, ImageDataInfo2D& out_image);

// 3D ROI 当前只支持 Box / RotatedBox 两种几何。
// 传入 2D 图形时直接返回 false。
template <typename PointCloudT>
bool CropPointCloudByGeometry(
    const HVGeometryInfo& geometry,
    const PointCloudT& input,
    PointCloudT& output);

template <>
bool CropPointCloudByGeometry<pcl::PointCloud<pcl::PointXYZ>>(
    const HVGeometryInfo& geometry,
    const pcl::PointCloud<pcl::PointXYZ>& input,
    pcl::PointCloud<pcl::PointXYZ>& output);

template <>
bool CropPointCloudByGeometry<open3d::geometry::PointCloud>(
    const HVGeometryInfo& geometry,
    const open3d::geometry::PointCloud& input,
    open3d::geometry::PointCloud& output);

