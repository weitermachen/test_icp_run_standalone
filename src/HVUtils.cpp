// Author: weitermachen
// Time: 2026-03-24

#include "HVUtils.h"

#include <cmath>

#pragma region Image Conversion
// 将 ImageDataInfo2D 转换为 cv::Mat (深拷贝)
cv::Mat ImageConverter::ToMat(const ImageDataInfo2D& input) {
    if (input.empty()) {
        return cv::Mat();
    }

    int cv_type = GetCvTypeFromChannels(input.channels);
    cv::Mat output(input.height, input.width, cv_type);

    // 深拷贝数据
    std::memcpy(output.data, input.image_data, input.getDataSize());

    return output;
}

// 将 cv::Mat 转换为 ImageDataInfo2D (深拷贝)
ImageDataInfo2D ImageConverter::FromMat(const cv::Mat& input) {
    if (input.empty()) {
        return ImageDataInfo2D();
    }

    // 确保是连续的8位无符号数据
    if (input.depth() != CV_8U) {
        throw std::invalid_argument("Only CV_8U depth is supported");
    }

    if (!input.isContinuous()) {
        // 如果不连续，先克隆为连续数据
        cv::Mat continuous = input.clone();
        return FromMat(continuous);
    }

    int channels = input.channels();
    ImageDataInfo2D output(input.cols, input.rows, channels);

    // 深拷贝数据
    std::memcpy(output.image_data, input.data, output.getDataSize());

    return output;
}

// 将 ImageDataInfo2D 转换为 cv::Mat (零拷贝)
cv::Mat ImageConverter::ToMatShallow(ImageDataInfo2D& input) {
    if (input.empty()) {
        return cv::Mat();
    }

    int cv_type = GetCvTypeFromChannels(input.channels);

    // 创建Mat，直接使用ImageDataInfo2D的数据指针
    // 注意：不会释放数据，生命周期由ImageDataInfo2D管理
    cv::Mat output(input.height, input.width, cv_type, input.image_data);

    return output;
}

// 将 cv::Mat 转换为 ImageDataInfo2D (零拷贝)
ImageDataInfo2D ImageConverter::FromMatShallow(cv::Mat& input) {
    if (input.empty()) {
        return ImageDataInfo2D();
    }

    // 确保是连续的8位无符号数据
    if (input.depth() != CV_8U) {
        throw std::invalid_argument("Only CV_8U depth is supported");
    }

    if (!input.isContinuous()) {
        throw std::invalid_argument("Mat must be continuous for shallow copy");
    }

    ImageDataInfo2D output;
    output.width = input.cols;
    output.height = input.rows;
    output.channels = input.channels();
    output.image_data = input.data;  // 共享数据指针

    // 警告：此处不拥有数据所有权，不应调用release()
    // 数据生命周期由输入的Mat管理

    return output;
}

// 辅助函数：获取OpenCV类型的通道数
int ImageConverter::GetChannelsFromCvType(int cv_type) {
    return CV_MAT_CN(cv_type);
}

// 辅助函数：根据通道数生成OpenCV类型
int ImageConverter::GetCvTypeFromChannels(int channels) {
    switch (channels) {
    case 1: return CV_8UC1;
    case 2: return CV_8UC2;
    case 3: return CV_8UC3;
    case 4: return CV_8UC4;
    default:
        throw std::invalid_argument("Unsupported number of channels");
    }
}
#pragma endregion

#pragma region Point Cloud Conversion
/**
    * @brief 将 HVPointCloud 转换为 Open3D PointCloud
    * @param input 自定义点云
    * @return Open3D 点云智能指针
    */
std::shared_ptr<open3d::geometry::PointCloud> PointCloudConverter::ToOpen3D(const HVPointCloud& input) {
    auto output = std::make_shared<open3d::geometry::PointCloud>();
    // 1. 预分配内存
    size_t num_points = input.points.size();
    output->points_.resize(num_points);
    // 2. 逐个赋值
    for (size_t i = 0; i < num_points; ++i) {
        output->points_[i] = Eigen::Vector3d(
            input.points[i].x,
            input.points[i].y,
            input.points[i].z
        );
    }
    return output;
}

/**
    * @brief 将 Open3D PointCloud 转换为 HVPointCloud
    * @param input Open3D 点云
    * @return 自定义点云结构
    */
HVPointCloud PointCloudConverter::FromOpen3D(const open3d::geometry::PointCloud& input) {
    HVPointCloud output;
    // 1. 预分配内存
    size_t num_points = input.points_.size();
    output.points.resize(num_points);
    // 2. 逐个赋值
    for (size_t i = 0; i < num_points; ++i) {
        const auto& pt = input.points_[i]; // Eigen::Vector3d
        output.points[i].x = pt.x();
        output.points[i].y = pt.y();
        output.points[i].z = pt.z();
    }
    return output;
}

/**
    * @brief 将 HVPointCloud 转换为 PCL PointCloud
    * @param input 自定义点云
    * @return PCL 点云智能指针
    */
pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudConverter::ToPCL(const HVPointCloud& input) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    // 1. 预分配内存
    size_t num_points = input.points.size();
    output->points.resize(num_points);
    output->width = num_points;
    output->height = 1;
    output->is_dense = false;
    // 2. 逐个赋值
    for (size_t i = 0; i < num_points; ++i) {
        output->points[i].x = input.points[i].x;
        output->points[i].y = input.points[i].y;
        output->points[i].z = input.points[i].z;
    }
    return output;
}

/**
    * @brief 将 PCL PointCloud 转换为 HVPointCloud
    * @param input PCL 点云
    * @return 自定义点云结构
    */
HVPointCloud PointCloudConverter::FromPCL(const pcl::PointCloud<pcl::PointXYZ>& input) {
    HVPointCloud output;
    // 1. 预分配内存
    size_t num_points = input.points.size();
    output.points.resize(num_points);
    // 2. 逐个赋值
    for (size_t i = 0; i < num_points; ++i) {
        output.points[i].x = input.points[i].x;
        output.points[i].y = input.points[i].y;
        output.points[i].z = input.points[i].z;
    }
    return output;
}

/**
    * @brief 将 Open3D PointCloud 转换为 PCL PointCloud
    * @param input Open3D 点云
    * @return PCL 点云智能指针
    */
pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudConverter::Open3DToPCL(const open3d::geometry::PointCloud& input) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    size_t num_points = input.points_.size();
    output->points.resize(num_points);
    output->width = num_points;
    output->height = 1;
    output->is_dense = false;

    for (size_t i = 0; i < num_points; ++i) {
        const auto& pt = input.points_[i];
        output->points[i].x = pt.x();
        output->points[i].y = pt.y();
        output->points[i].z = pt.z();
    }
    return output;
}

/**
    * @brief 将 PCL PointCloud 转换为 Open3D PointCloud
    * @param input PCL 点云
    * @return Open3D 点云智能指针
    */
std::shared_ptr<open3d::geometry::PointCloud> PointCloudConverter::PCLToOpen3D(const pcl::PointCloud<pcl::PointXYZ>& input) {
    auto output = std::make_shared<open3d::geometry::PointCloud>();
    size_t num_points = input.points.size();
    output->points_.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        output->points_[i] = Eigen::Vector3d(
            input.points[i].x,
            input.points[i].y,
            input.points[i].z
        );
    }
    return output;
}
#pragma endregion

namespace {

bool IsFiniteDouble(double value) {
    return std::isfinite(value) != 0;
}

bool IsFinitePoint(const HVPoint& point) {
    return IsFiniteDouble(point.x) && IsFiniteDouble(point.y);
}

bool IsFinitePoint3D(const HVPoint3D& point) {
    return IsFiniteDouble(point.x) &&
        IsFiniteDouble(point.y) &&
        IsFiniteDouble(point.z);
}

bool IsFiniteOrientation3D(const HVOrientation3D& orientation) {
    return IsFiniteDouble(orientation.roll_deg_) &&
        IsFiniteDouble(orientation.pitch_deg_) &&
        IsFiniteDouble(orientation.yaw_deg_);
}

bool IsPointInsideBox(const HVBox& box, double x, double y, double z) {
    const HVPoint3D min_corner = box.MinCorner();
    const HVPoint3D max_corner = box.MaxCorner();
    return x >= min_corner.x && x <= max_corner.x &&
        y >= min_corner.y && y <= max_corner.y &&
        z >= min_corner.z && z <= max_corner.z;
}

bool IsPointInsideRotatedBox(const HVRotatedBox& box, double x, double y, double z) {
    const double roll_rad = box.orientation_.roll_deg_ * std::acos(-1.0) / 180.0;
    const double pitch_rad = box.orientation_.pitch_deg_ * std::acos(-1.0) / 180.0;
    const double yaw_rad = box.orientation_.yaw_deg_ * std::acos(-1.0) / 180.0;
    const double cos_roll = std::cos(roll_rad);
    const double sin_roll = std::sin(roll_rad);
    const double cos_pitch = std::cos(pitch_rad);
    const double sin_pitch = std::sin(pitch_rad);
    const double cos_yaw = std::cos(yaw_rad);
    const double sin_yaw = std::sin(yaw_rad);

    const double dx = x - box.center_.x;
    const double dy = y - box.center_.y;
    const double dz = z - box.center_.z;

    // 逆变换顺序与 HVRotatedBox::Vertices() 的正向 roll->pitch->yaw 保持一致。
    const double undo_yaw_x = dx * cos_yaw + dy * sin_yaw;
    const double undo_yaw_y = -dx * sin_yaw + dy * cos_yaw;
    const double undo_pitch_x = undo_yaw_x * cos_pitch - dz * sin_pitch;
    const double undo_pitch_z = undo_yaw_x * sin_pitch + dz * cos_pitch;
    const double local_y = undo_yaw_y * cos_roll + undo_pitch_z * sin_roll;
    const double local_z = -undo_yaw_y * sin_roll + undo_pitch_z * cos_roll;

    const double half_length = box.length_ * 0.5;
    const double half_width = box.width_ * 0.5;
    const double half_height = box.height_ * 0.5;

    return undo_pitch_x >= -half_length && undo_pitch_x <= half_length &&
        local_y >= -half_width && local_y <= half_width &&
        local_z >= -half_height && local_z <= half_height;
}

bool IsSupportedPointCloudGeometry(const HVGeometryInfo& geometry) {
    return geometry.shape_type_ == HVGeometryShapeType::Box ||
        geometry.shape_type_ == HVGeometryShapeType::RotatedBox;
}

bool IsPointInsideGeometry(const HVGeometryInfo& geometry, double x, double y, double z) {
    switch (geometry.shape_type_) {
    case HVGeometryShapeType::Box:
        return IsPointInsideBox(geometry.AsBox(), x, y, z);
    case HVGeometryShapeType::RotatedBox:
        return IsPointInsideRotatedBox(geometry.AsRotatedBox(), x, y, z);
    default:
        return false;
    }
}

void MarkMaskPixel(int x, int y, cv::Mat& mask) {
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) {
        return;
    }
    mask.at<unsigned char>(y, x) = 255;
}

void RasterizePoint(const HVPoint& point, cv::Mat& mask) {
    MarkMaskPixel(
        static_cast<int>(std::lround(point.x)),
        static_cast<int>(std::lround(point.y)),
        mask);
}

void RasterizeLine(const HVLineSegment& line_segment, cv::Mat& mask) {
    const cv::Point start(
        static_cast<int>(std::lround(line_segment.start_point_.x)),
        static_cast<int>(std::lround(line_segment.start_point_.y)));
    const cv::Point end(
        static_cast<int>(std::lround(line_segment.end_point_.x)),
        static_cast<int>(std::lround(line_segment.end_point_.y)));
    cv::line(mask, start, end, cv::Scalar(255), 1, cv::LINE_8);
}

void RasterizeAxisAlignedRect(const HVRect& rect, cv::Mat& mask) {
    for (int row = 0; row < mask.rows; ++row) {
        const double pixel_center_y = static_cast<double>(row) + 0.5;
        if (pixel_center_y < rect.y_ || pixel_center_y >= rect.y_ + rect.height_) {
            continue;
        }

        for (int col = 0; col < mask.cols; ++col) {
            const double pixel_center_x = static_cast<double>(col) + 0.5;
            if (pixel_center_x >= rect.x_ && pixel_center_x < rect.x_ + rect.width_) {
                mask.at<unsigned char>(row, col) = 255;
            }
        }
    }
}

void RasterizeRotatedRect(const HVRotatedRect& rect, cv::Mat& mask) {
    const double angle_rad = rect.angle_deg_ * std::acos(-1.0) / 180.0;
    const double cos_angle = std::cos(angle_rad);
    const double sin_angle = std::sin(angle_rad);
    const double half_width = rect.width_ * 0.5;
    const double half_height = rect.height_ * 0.5;
    const double sample_bias = 1e-9;

    for (int row = 0; row < mask.rows; ++row) {
        const double pixel_center_y = static_cast<double>(row) + 0.5 + sample_bias;
        for (int col = 0; col < mask.cols; ++col) {
            const double pixel_center_x = static_cast<double>(col) + 0.5 + sample_bias;
            const double dx = pixel_center_x - rect.center_.x;
            const double dy = pixel_center_y - rect.center_.y;

            // Project the pixel center back into the rectangle-local frame
            // before testing axis-aligned bounds in that local frame.
            const double local_x = dx * cos_angle + dy * sin_angle;
            const double local_y = -dx * sin_angle + dy * cos_angle;
            if (local_x >= -half_width &&
                local_x < half_width &&
                local_y >= -half_height &&
                local_y < half_height) {
                mask.at<unsigned char>(row, col) = 255;
            }
        }
    }
}

} // namespace

bool IsValidRoiInfo(const HVGeometryInfo& roi) {
    switch (roi.shape_type_) {
    case HVGeometryShapeType::Point:
        return IsFinitePoint(roi.AsPoint());
    case HVGeometryShapeType::LineSegment:
        return IsFinitePoint(roi.AsLineSegment().start_point_) &&
            IsFinitePoint(roi.AsLineSegment().end_point_);
    case HVGeometryShapeType::Rectangle:
        return IsFiniteDouble(roi.AsRect().x_) &&
            IsFiniteDouble(roi.AsRect().y_) &&
            IsFiniteDouble(roi.AsRect().width_) &&
            IsFiniteDouble(roi.AsRect().height_) &&
            roi.AsRect().width_ > 0.0 &&
            roi.AsRect().height_ > 0.0;
    case HVGeometryShapeType::RotatedRectangle:
        return IsFinitePoint(roi.AsRotatedRect().center_) &&
            IsFiniteDouble(roi.AsRotatedRect().width_) &&
            IsFiniteDouble(roi.AsRotatedRect().height_) &&
            IsFiniteDouble(roi.AsRotatedRect().angle_deg_) &&
            roi.AsRotatedRect().width_ > 0.0 &&
            roi.AsRotatedRect().height_ > 0.0;
    case HVGeometryShapeType::Box:
        return IsFinitePoint3D(roi.AsBox().center_) &&
            IsFiniteDouble(roi.AsBox().length_) &&
            IsFiniteDouble(roi.AsBox().width_) &&
            IsFiniteDouble(roi.AsBox().height_) &&
            roi.AsBox().IsValid();
    case HVGeometryShapeType::RotatedBox:
        return IsFinitePoint3D(roi.AsRotatedBox().center_) &&
            IsFiniteDouble(roi.AsRotatedBox().length_) &&
            IsFiniteDouble(roi.AsRotatedBox().width_) &&
            IsFiniteDouble(roi.AsRotatedBox().height_) &&
            IsFiniteOrientation3D(roi.AsRotatedBox().orientation_) &&
            roi.AsRotatedBox().IsValid();
    default:
        return false;
    }
}

bool BuildRoiMask(const HVGeometryInfo& roi, int image_width, int image_height, cv::Mat& mask) {
    if (!IsValidRoiInfo(roi) || image_width <= 0 || image_height <= 0) {
        mask.release();
        return false;
    }

    mask = cv::Mat::zeros(image_height, image_width, CV_8UC1);
    switch (roi.shape_type_) {
    case HVGeometryShapeType::Point:
        RasterizePoint(roi.AsPoint(), mask);
        return true;
    case HVGeometryShapeType::LineSegment:
        RasterizeLine(roi.AsLineSegment(), mask);
        return true;
    case HVGeometryShapeType::Rectangle:
        RasterizeAxisAlignedRect(roi.AsRect(), mask);
        return true;
    case HVGeometryShapeType::RotatedRectangle:
        RasterizeRotatedRect(roi.AsRotatedRect(), mask);
        return true;
    default:
        mask.release();
        return false;
    }
}

bool BuildMaskedImageFromRoi(const HVGeometryInfo& roi, const ImageDataInfo2D& src_image, ImageDataInfo2D& out_image) {
    if (src_image.empty()) {
        out_image = ImageDataInfo2D();
        return false;
    }

    cv::Mat mask;
    if (!BuildRoiMask(roi, src_image.width, src_image.height, mask)) {
        out_image = ImageDataInfo2D();
        return false;
    }

    const cv::Mat src = ImageConverter::ToMat(src_image);
    cv::Mat masked = cv::Mat::zeros(src.size(), src.type());
    src.copyTo(masked, mask);
    out_image = ImageConverter::FromMat(masked);
    return true;
}

template <>
bool CropPointCloudByGeometry<pcl::PointCloud<pcl::PointXYZ>>(
    const HVGeometryInfo& geometry,
    const pcl::PointCloud<pcl::PointXYZ>& input,
    pcl::PointCloud<pcl::PointXYZ>& output) {
    output.clear();
    output.width = 0;
    output.height = 1;
    output.is_dense = input.is_dense;

    if (!IsSupportedPointCloudGeometry(geometry) || !IsValidRoiInfo(geometry)) {
        return false;
    }

    output.points.reserve(input.points.size());
    for (const pcl::PointXYZ& point : input.points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        if (IsPointInsideGeometry(geometry, point.x, point.y, point.z)) {
            output.points.push_back(point);
        }
    }

    output.width = static_cast<std::uint32_t>(output.points.size());
    return true;
}

template <>
bool CropPointCloudByGeometry<open3d::geometry::PointCloud>(
    const HVGeometryInfo& geometry,
    const open3d::geometry::PointCloud& input,
    open3d::geometry::PointCloud& output) {
    output.Clear();

    if (!IsSupportedPointCloudGeometry(geometry) || !IsValidRoiInfo(geometry)) {
        return false;
    }

    const bool has_colors = input.HasColors();
    const bool has_normals = input.HasNormals();
    output.points_.reserve(input.points_.size());
    if (has_colors) {
        output.colors_.reserve(input.colors_.size());
    }
    if (has_normals) {
        output.normals_.reserve(input.normals_.size());
    }

    for (size_t i = 0; i < input.points_.size(); ++i) {
        const Eigen::Vector3d& point = input.points_[i];
        if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
            continue;
        }
        if (!IsPointInsideGeometry(geometry, point.x(), point.y(), point.z())) {
            continue;
        }

        output.points_.push_back(point);
        if (has_colors) {
            output.colors_.push_back(input.colors_[i]);
        }
        if (has_normals) {
            output.normals_.push_back(input.normals_[i]);
        }
    }

    return true;
}

