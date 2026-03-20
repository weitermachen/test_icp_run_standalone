#pragma once

#include "3d_pilot_data_define.h"

#include <array>
#include <cmath>
#include <new>
#include <string>
#include <utility>

// 新的前端几何定义统一放在这里。
// 3d_pilot_data_define.h 里的旧 ImageRoi 仍保留为历史轴对齐矩形占位类型。

enum class HVGeometryShapeType
{
    Point = 0,
    LineSegment = 1,
    Rectangle = 2,
    RotatedRectangle = 3,
    Box = 4,
    RotatedBox = 5
};

struct HVPoint
{
    double x;
    double y;

    HVPoint()
        : x(0.0)
        , y(0.0)
    {
    }

    HVPoint(double x_value, double y_value)
        : x(x_value)
        , y(y_value)
    {
    }

    double DistanceTo(const HVPoint& other) const
    {
        const double dx = other.x - x;
        const double dy = other.y - y;
        return std::sqrt(dx * dx + dy * dy);
    }

    HVPoint MidpointTo(const HVPoint& other) const
    {
        return HVPoint((x + other.x) * 0.5, (y + other.y) * 0.5);
    }
};

struct HVLineSegment
{
    HVPoint start_point_;
    HVPoint end_point_;

    HVLineSegment() = default;

    HVLineSegment(const HVPoint& start_point, const HVPoint& end_point)
        : start_point_(start_point)
        , end_point_(end_point)
    {
    }

    double LengthSquared() const
    {
        const double dx = end_point_.x - start_point_.x;
        const double dy = end_point_.y - start_point_.y;
        return dx * dx + dy * dy;
    }

    double Length() const
    {
        return std::sqrt(LengthSquared());
    }

    HVPoint Midpoint() const
    {
        return start_point_.MidpointTo(end_point_);
    }

    bool IsDegenerate() const
    {
        return start_point_.x == end_point_.x && start_point_.y == end_point_.y;
    }
};

struct HVRect
{
    double x_;
    double y_;
    double width_;
    double height_;

    HVRect()
        : x_(0.0)
        , y_(0.0)
        , width_(0.0)
        , height_(0.0)
    {
    }

    HVRect(double x_value, double y_value, double width_value, double height_value)
        : x_(x_value)
        , y_(y_value)
        , width_(width_value)
        , height_(height_value)
    {
    }

    bool IsValid() const
    {
        return width_ > 0.0 && height_ > 0.0;
    }

    double Area() const
    {
        return width_ * height_;
    }

    HVPoint Center() const
    {
        return HVPoint(x_ + width_ * 0.5, y_ + height_ * 0.5);
    }

    HVPoint TopLeft() const
    {
        return HVPoint(x_, y_);
    }

    HVPoint TopRight() const
    {
        return HVPoint(x_ + width_, y_);
    }

    HVPoint BottomLeft() const
    {
        return HVPoint(x_, y_ + height_);
    }

    HVPoint BottomRight() const
    {
        return HVPoint(x_ + width_, y_ + height_);
    }
};

struct HVRotatedRect
{
    HVPoint center_;
    double width_;
    double height_;
    double angle_deg_;

    HVRotatedRect()
        : center_()
        , width_(0.0)
        , height_(0.0)
        , angle_deg_(0.0)
    {
    }

    HVRotatedRect(
        const HVPoint& center,
        double width_value,
        double height_value,
        double angle_deg_value)
        : center_(center)
        , width_(width_value)
        , height_(height_value)
        , angle_deg_(angle_deg_value)
    {
    }

    bool IsValid() const
    {
        return width_ > 0.0 && height_ > 0.0;
    }

    double Area() const
    {
        return width_ * height_;
    }

    HVPoint Center() const
    {
        return center_;
    }

    std::array<HVPoint, 4> VerticesClockwise() const
    {
        const double half_width = width_ * 0.5;
        const double half_height = height_ * 0.5;
        const double angle_rad = angle_deg_ * std::acos(-1.0) / 180.0;
        const double cos_angle = std::cos(angle_rad);
        const double sin_angle = std::sin(angle_rad);
        const std::array<HVPoint, 4> local_vertices = {
            HVPoint(-half_width, -half_height),
            HVPoint(half_width, -half_height),
            HVPoint(half_width, half_height),
            HVPoint(-half_width, half_height)
        };

        std::array<HVPoint, 4> vertices = {};
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            const HVPoint& local = local_vertices[i];
            // 图像坐标系里，正角度按顺时针旋转。
            vertices[i] = HVPoint(
                center_.x + local.x * cos_angle - local.y * sin_angle,
                center_.y + local.x * sin_angle + local.y * cos_angle);
        }
        return vertices;
    }
};

struct HVOrientation3D
{
    double roll_deg_;
    double pitch_deg_;
    double yaw_deg_;

    HVOrientation3D()
        : roll_deg_(0.0)
        , pitch_deg_(0.0)
        , yaw_deg_(0.0)
    {
    }

    HVOrientation3D(double roll_deg, double pitch_deg, double yaw_deg)
        : roll_deg_(roll_deg)
        , pitch_deg_(pitch_deg)
        , yaw_deg_(yaw_deg)
    {
    }
};

struct HVBox
{
    HVPoint3D center_;
    double length_;
    double width_;
    double height_;

    HVBox()
        : center_()
        , length_(0.0)
        , width_(0.0)
        , height_(0.0)
    {
    }

    HVBox(const HVPoint3D& center, double length, double width, double height)
        : center_(center)
        , length_(length)
        , width_(width)
        , height_(height)
    {
    }

    bool IsValid() const
    {
        return length_ > 0.0 && width_ > 0.0 && height_ > 0.0;
    }

    double Volume() const
    {
        return length_ * width_ * height_;
    }

    HVPoint3D Center() const
    {
        return center_;
    }

    HVPoint3D MinCorner() const
    {
        return HVPoint3D(center_.x - length_ * 0.5, center_.y - width_ * 0.5, center_.z - height_ * 0.5);
    }

    HVPoint3D MaxCorner() const
    {
        return HVPoint3D(center_.x + length_ * 0.5, center_.y + width_ * 0.5, center_.z + height_ * 0.5);
    }

    std::array<HVPoint3D, 8> Vertices() const
    {
        const HVPoint3D min_corner = MinCorner();
        const HVPoint3D max_corner = MaxCorner();
        return {
            HVPoint3D(min_corner.x, min_corner.y, min_corner.z),
            HVPoint3D(max_corner.x, min_corner.y, min_corner.z),
            HVPoint3D(max_corner.x, max_corner.y, min_corner.z),
            HVPoint3D(min_corner.x, max_corner.y, min_corner.z),
            HVPoint3D(min_corner.x, min_corner.y, max_corner.z),
            HVPoint3D(max_corner.x, min_corner.y, max_corner.z),
            HVPoint3D(max_corner.x, max_corner.y, max_corner.z),
            HVPoint3D(min_corner.x, max_corner.y, max_corner.z)
        };
    }
};

struct HVRotatedBox
{
    HVPoint3D center_;
    double length_;
    double width_;
    double height_;
    HVOrientation3D orientation_;

    HVRotatedBox()
        : center_()
        , length_(0.0)
        , width_(0.0)
        , height_(0.0)
        , orientation_()
    {
    }

    HVRotatedBox(
        const HVPoint3D& center,
        double length,
        double width,
        double height,
        const HVOrientation3D& orientation)
        : center_(center)
        , length_(length)
        , width_(width)
        , height_(height)
        , orientation_(orientation)
    {
    }

    bool IsValid() const
    {
        return length_ > 0.0 && width_ > 0.0 && height_ > 0.0;
    }

    double Volume() const
    {
        return length_ * width_ * height_;
    }

    HVPoint3D Center() const
    {
        return center_;
    }

    std::array<HVPoint3D, 8> Vertices() const
    {
        const double half_length = length_ * 0.5;
        const double half_width = width_ * 0.5;
        const double half_height = height_ * 0.5;
        const double roll_rad = orientation_.roll_deg_ * std::acos(-1.0) / 180.0;
        const double pitch_rad = orientation_.pitch_deg_ * std::acos(-1.0) / 180.0;
        const double yaw_rad = orientation_.yaw_deg_ * std::acos(-1.0) / 180.0;
        const double cos_roll = std::cos(roll_rad);
        const double sin_roll = std::sin(roll_rad);
        const double cos_pitch = std::cos(pitch_rad);
        const double sin_pitch = std::sin(pitch_rad);
        const double cos_yaw = std::cos(yaw_rad);
        const double sin_yaw = std::sin(yaw_rad);

        const std::array<HVPoint3D, 8> local_vertices = {
            HVPoint3D(-half_length, -half_width, -half_height),
            HVPoint3D(half_length, -half_width, -half_height),
            HVPoint3D(half_length, half_width, -half_height),
            HVPoint3D(-half_length, half_width, -half_height),
            HVPoint3D(-half_length, -half_width, half_height),
            HVPoint3D(half_length, -half_width, half_height),
            HVPoint3D(half_length, half_width, half_height),
            HVPoint3D(-half_length, half_width, half_height)
        };

        std::array<HVPoint3D, 8> vertices = {};
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            const HVPoint3D& local = local_vertices[i];
            // 按 roll(x) -> pitch(y) -> yaw(z) 顺序应用右手系旋转。
            const double roll_y = local.y * cos_roll - local.z * sin_roll;
            const double roll_z = local.y * sin_roll + local.z * cos_roll;

            const double pitch_x = local.x * cos_pitch + roll_z * sin_pitch;
            const double pitch_z = -local.x * sin_pitch + roll_z * cos_pitch;

            const double yaw_x = pitch_x * cos_yaw - roll_y * sin_yaw;
            const double yaw_y = pitch_x * sin_yaw + roll_y * cos_yaw;

            vertices[i] = HVPoint3D(
                center_.x + yaw_x,
                center_.y + yaw_y,
                center_.z + pitch_z);
        }
        return vertices;
    }
};

union HVGeometry
{
    HVPoint point_;
    HVLineSegment line_segment_;
    HVRect rect_;
    HVRotatedRect rotated_rect_;
    HVBox box_;
    HVRotatedBox rotated_box_;

    HVGeometry() {}
    ~HVGeometry() {}
};

struct HVGeometryInfo
{
    int geometry_id_;
    std::string geometry_name_;
    HVGeometryShapeType shape_type_;
    // 当前 HVGeometryInfo 里的 2D 几何按图像像素坐标解释。
    // 原点在左上角，旋转矩形 angle_deg_ 的正方向为顺时针。
    HVGeometry geometry_;

    HVGeometryInfo()
        : geometry_id_(-1)
        , geometry_name_()
        , shape_type_(HVGeometryShapeType::Point)
        , geometry_()
    {
        new (&geometry_.point_) HVPoint();
    }

    HVGeometryInfo(const HVGeometryInfo& other)
        : geometry_id_(other.geometry_id_)
        , geometry_name_(other.geometry_name_)
        , shape_type_(HVGeometryShapeType::Point)
        , geometry_()
    {
        CopyActiveValueFrom(other);
    }

    HVGeometryInfo(HVGeometryInfo&& other) noexcept
        : geometry_id_(other.geometry_id_)
        , geometry_name_(std::move(other.geometry_name_))
        , shape_type_(HVGeometryShapeType::Point)
        , geometry_()
    {
        CopyActiveValueFrom(other);
    }

    HVGeometryInfo& operator=(const HVGeometryInfo& other)
    {
        if (this == &other) {
            return *this;
        }

        geometry_id_ = other.geometry_id_;
        geometry_name_ = other.geometry_name_;
        DestroyActiveValue();
        CopyActiveValueFrom(other);
        return *this;
    }

    HVGeometryInfo& operator=(HVGeometryInfo&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }

        geometry_id_ = other.geometry_id_;
        geometry_name_ = std::move(other.geometry_name_);
        DestroyActiveValue();
        CopyActiveValueFrom(other);
        return *this;
    }

    ~HVGeometryInfo()
    {
        DestroyActiveValue();
    }

    void SetPoint(const HVPoint& point)
    {
        DestroyActiveValue();
        shape_type_ = HVGeometryShapeType::Point;
        new (&geometry_.point_) HVPoint(point);
    }

    void SetLineSegment(const HVLineSegment& line_segment)
    {
        DestroyActiveValue();
        shape_type_ = HVGeometryShapeType::LineSegment;
        new (&geometry_.line_segment_) HVLineSegment(line_segment);
    }

    void SetRect(const HVRect& rect)
    {
        DestroyActiveValue();
        shape_type_ = HVGeometryShapeType::Rectangle;
        new (&geometry_.rect_) HVRect(rect);
    }

    void SetRotatedRect(const HVRotatedRect& rotated_rect)
    {
        DestroyActiveValue();
        shape_type_ = HVGeometryShapeType::RotatedRectangle;
        new (&geometry_.rotated_rect_) HVRotatedRect(rotated_rect);
    }

    void SetBox(const HVBox& box)
    {
        DestroyActiveValue();
        shape_type_ = HVGeometryShapeType::Box;
        new (&geometry_.box_) HVBox(box);
    }

    void SetRotatedBox(const HVRotatedBox& rotated_box)
    {
        DestroyActiveValue();
        shape_type_ = HVGeometryShapeType::RotatedBox;
        new (&geometry_.rotated_box_) HVRotatedBox(rotated_box);
    }

    HVPoint& AsPoint()
    {
        return geometry_.point_;
    }

    const HVPoint& AsPoint() const
    {
        return geometry_.point_;
    }

    HVLineSegment& AsLineSegment()
    {
        return geometry_.line_segment_;
    }

    const HVLineSegment& AsLineSegment() const
    {
        return geometry_.line_segment_;
    }

    HVRect& AsRect()
    {
        return geometry_.rect_;
    }

    const HVRect& AsRect() const
    {
        return geometry_.rect_;
    }

    HVRotatedRect& AsRotatedRect()
    {
        return geometry_.rotated_rect_;
    }

    const HVRotatedRect& AsRotatedRect() const
    {
        return geometry_.rotated_rect_;
    }

    HVBox& AsBox()
    {
        return geometry_.box_;
    }

    const HVBox& AsBox() const
    {
        return geometry_.box_;
    }

    HVRotatedBox& AsRotatedBox()
    {
        return geometry_.rotated_box_;
    }

    const HVRotatedBox& AsRotatedBox() const
    {
        return geometry_.rotated_box_;
    }

private:
    void DestroyActiveValue()
    {
        switch (shape_type_) {
        case HVGeometryShapeType::Point:
            geometry_.point_.~HVPoint();
            break;
        case HVGeometryShapeType::LineSegment:
            geometry_.line_segment_.~HVLineSegment();
            break;
        case HVGeometryShapeType::Rectangle:
            geometry_.rect_.~HVRect();
            break;
        case HVGeometryShapeType::RotatedRectangle:
            geometry_.rotated_rect_.~HVRotatedRect();
            break;
        case HVGeometryShapeType::Box:
            geometry_.box_.~HVBox();
            break;
        case HVGeometryShapeType::RotatedBox:
            geometry_.rotated_box_.~HVRotatedBox();
            break;
        default:
            break;
        }
    }

    void CopyActiveValueFrom(const HVGeometryInfo& other)
    {
        shape_type_ = other.shape_type_;
        switch (other.shape_type_) {
        case HVGeometryShapeType::Point:
            new (&geometry_.point_) HVPoint(other.AsPoint());
            break;
        case HVGeometryShapeType::LineSegment:
            new (&geometry_.line_segment_) HVLineSegment(other.AsLineSegment());
            break;
        case HVGeometryShapeType::Rectangle:
            new (&geometry_.rect_) HVRect(other.AsRect());
            break;
        case HVGeometryShapeType::RotatedRectangle:
            new (&geometry_.rotated_rect_) HVRotatedRect(other.AsRotatedRect());
            break;
        case HVGeometryShapeType::Box:
            new (&geometry_.box_) HVBox(other.AsBox());
            break;
        case HVGeometryShapeType::RotatedBox:
            new (&geometry_.rotated_box_) HVRotatedBox(other.AsRotatedBox());
            break;
        default:
            new (&geometry_.point_) HVPoint();
            shape_type_ = HVGeometryShapeType::Point;
            break;
        }
    }
};
