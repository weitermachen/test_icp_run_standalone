#pragma once

#include <memory>
#include <vector>
#include <filesystem>

/**
 * 定义数据类型
 */
 // 基本数据类型
#define HV_INT                     0x0001
#define HV_LONG                    0x0002
#define HV_FLOAT                   0x0003
#define HV_DOUBLE                  0x0004
#define HV_BOOLEAN                 0x0005
#define HV_STRING                  0x0006

// 图像数据类型
#define HV_IMAGEDATAINFO2D         0x0101
#define HV_IMAGEDATAINFODEPTH      0x0102
#define HV_IMAGEROI                0x0103

// 3D点云相关
#define HV_POINT3D                 0x0201
#define HV_POINTCLOUD              0x0202

// 3D点
struct HVPoint3D
{
    double x;
    double y;
    double z;

    HVPoint3D()
        : x(0), y(0), z(0) {
    }
    HVPoint3D(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_) {
    }
};

/**
 * 定义数据结构体
 */
 // 2D图像数据
struct ImageDataInfo2D {
    size_t width;
    size_t height;
    unsigned char* image_data;
    int channels;

    // 构造函数
    ImageDataInfo2D()
        : width(0), height(0), image_data(nullptr), channels(0) {
    }

    ImageDataInfo2D(size_t w, size_t h, int ch)
        : width(w), height(h), channels(ch), image_data(nullptr) {
        allocate();
    }

    // 拷贝构造函数
    ImageDataInfo2D(const ImageDataInfo2D& other)
        : width(other.width), height(other.height), channels(other.channels), image_data(nullptr) {
        if (other.image_data) {
            allocate();
            std::memcpy(image_data, other.image_data, getDataSize());
        }
    }

    // 移动构造函数
    ImageDataInfo2D(ImageDataInfo2D&& other) noexcept
        : width(other.width), height(other.height),
        image_data(other.image_data), channels(other.channels) {
        other.image_data = nullptr;
        other.width = 0;
        other.height = 0;
        other.channels = 0;
    }

    // 拷贝赋值运算符
    ImageDataInfo2D& operator=(const ImageDataInfo2D& other) {
        if (this != &other) {
            release();
            width = other.width;
            height = other.height;
            channels = other.channels;
            if (other.image_data) {
                allocate();
                std::memcpy(image_data, other.image_data, getDataSize());
            }
        }
        return *this;
    }

    // 移动赋值运算符
    ImageDataInfo2D& operator=(ImageDataInfo2D&& other) noexcept {
        if (this != &other) {
            release();
            width = other.width;
            height = other.height;
            channels = other.channels;
            image_data = other.image_data;

            other.image_data = nullptr;
            other.width = 0;
            other.height = 0;
            other.channels = 0;
        }
        return *this;
    }

    // 析构函数
    ~ImageDataInfo2D() {
        release();
    }

    // 分配内存
    void allocate() {
        if (width > 0 && height > 0 && channels > 0) {
            size_t size = getDataSize();
            image_data = new unsigned char[size];
            std::memset(image_data, 0, size);
        }
    }

    // 释放内存
    void release() {
        if (image_data) {
            delete[] image_data;
            image_data = nullptr;
        }
    }

    // 重新分配内存
    void resize(size_t w, size_t h, int ch) {
        release();
        width = w;
        height = h;
        channels = ch;
        allocate();
    }

    // 获取数据大小（字节）
    size_t getDataSize() const {
        return width * height * channels;
    }

    // 获取单个像素大小（字节）
    size_t getPixelSize() const {
        return channels;
    }

    // 检查是否为空
    bool empty() const {
        return image_data == nullptr || width == 0 || height == 0 || channels == 0;
    }

    // 检查是否有效
    bool isValid() const {
        return !empty();
    }

    // 获取指定位置的像素指针
    unsigned char* at(size_t row, size_t col) {
        if (row >= height || col >= width) {
            throw std::out_of_range("Pixel position out of range");
        }
        return image_data + (row * width + col) * channels;
    }

    const unsigned char* at(size_t row, size_t col) const {
        if (row >= height || col >= width) {
            throw std::out_of_range("Pixel position out of range");
        }
        return image_data + (row * width + col) * channels;
    }

    // 访问指定位置的指定通道
    unsigned char& pixel(size_t row, size_t col, int channel) {
        if (row >= height || col >= width) {
            throw std::out_of_range("Pixel position out of range");
        }
        if (channel < 0 || channel >= channels) {
            throw std::out_of_range("Channel index out of range");
        }
        return image_data[(row * width + col) * channels + channel];
    }

    const unsigned char& pixel(size_t row, size_t col, int channel) const {
        if (row >= height || col >= width) {
            throw std::out_of_range("Pixel position out of range");
        }
        if (channel < 0 || channel >= channels) {
            throw std::out_of_range("Channel index out of range");
        }
        return image_data[(row * width + col) * channels + channel];
    }

    // 填充数据
    void fill(unsigned char value) {
        if (image_data) {
            std::memset(image_data, value, getDataSize());
        }
    }

    // 填充指定通道
    void fillChannel(int channel, unsigned char value) {
        if (channel < 0 || channel >= channels) {
            throw std::out_of_range("Channel index out of range");
        }
        if (!image_data) return;

        for (size_t i = 0; i < width * height; ++i) {
            image_data[i * channels + channel] = value;
        }
    }

    // 从外部数据复制
    void copyFrom(const unsigned char* data, size_t size) {
        if (!image_data) {
            throw std::runtime_error("Image data not allocated");
        }
        if (size != getDataSize()) {
            throw std::invalid_argument("Data size mismatch");
        }
        std::memcpy(image_data, data, size);
    }

    // 复制到外部缓冲区
    void copyTo(unsigned char* buffer, size_t buffer_size) const {
        if (!image_data) {
            throw std::runtime_error("Image data is empty");
        }
        size_t data_size = getDataSize();
        if (buffer_size < data_size) {
            throw std::invalid_argument("Buffer size too small");
        }
        std::memcpy(buffer, image_data, data_size);
    }

    // 克隆（深拷贝）
    ImageDataInfo2D clone() const {
        ImageDataInfo2D result(width, height, channels);
        if (image_data) {
            std::memcpy(result.image_data, image_data, getDataSize());
        }
        return result;
    }

    // 获取行指针
    unsigned char* row(size_t r) {
        if (r >= height) {
            throw std::out_of_range("Row index out of range");
        }
        return image_data + r * width * channels;
    }

    const unsigned char* row(size_t r) const {
        if (r >= height) {
            throw std::out_of_range("Row index out of range");
        }
        return image_data + r * width * channels;
    }

    // 获取步长（每行字节数）
    size_t step() const {
        return width * channels;
    }

    // 获取ROI（感兴趣区域）
    ImageDataInfo2D getROI(size_t x, size_t y, size_t roi_width, size_t roi_height) const {
        if (x + roi_width > width || y + roi_height > height) {
            throw std::out_of_range("ROI exceeds image boundaries");
        }

        ImageDataInfo2D roi(roi_width, roi_height, channels);

        for (size_t row = 0; row < roi_height; ++row) {
            const unsigned char* src = at(y + row, x);
            unsigned char* dst = roi.row(row);
            std::memcpy(dst, src, roi_width * channels);
        }

        return roi;
    }

    // 设置ROI
    void setROI(size_t x, size_t y, const ImageDataInfo2D& roi) {
        if (x + roi.width > width || y + roi.height > height) {
            throw std::out_of_range("ROI exceeds image boundaries");
        }
        if (roi.channels != channels) {
            throw std::invalid_argument("ROI channels mismatch");
        }

        for (size_t row = 0; row < roi.height; ++row) {
            const unsigned char* src = roi.row(row);
            unsigned char* dst = at(y + row, x);
            std::memcpy(dst, src, roi.width * channels);
        }
    }

    // 提取单个通道
    ImageDataInfo2D extractChannel(int channel) const {
        if (channel < 0 || channel >= channels) {
            throw std::out_of_range("Channel index out of range");
        }

        ImageDataInfo2D result(width, height, 1);

        for (size_t i = 0; i < width * height; ++i) {
            result.image_data[i] = image_data[i * channels + channel];
        }

        return result;
    }

    // 设置单个通道
    void setChannel(int channel, const ImageDataInfo2D& channel_data) {
        if (channel < 0 || channel >= channels) {
            throw std::out_of_range("Channel index out of range");
        }
        if (channel_data.width != width || channel_data.height != height) {
            throw std::invalid_argument("Channel data size mismatch");
        }
        if (channel_data.channels != 1) {
            throw std::invalid_argument("Channel data must be single channel");
        }

        for (size_t i = 0; i < width * height; ++i) {
            image_data[i * channels + channel] = channel_data.image_data[i];
        }
    }

    // 转换通道数
    ImageDataInfo2D convertChannels(int new_channels) const {
        if (new_channels <= 0) {
            throw std::invalid_argument("Invalid number of channels");
        }

        ImageDataInfo2D result(width, height, new_channels);

        for (size_t i = 0; i < width * height; ++i) {
            if (channels == 1 && new_channels == 3) {
                // 灰度 -> RGB (复制到所有通道)
                unsigned char gray = image_data[i];
                result.image_data[i * 3 + 0] = gray;
                result.image_data[i * 3 + 1] = gray;
                result.image_data[i * 3 + 2] = gray;
            }
            else if (channels == 3 && new_channels == 1) {
                // RGB -> 灰度 (简单平均)
                unsigned char r = image_data[i * 3 + 0];
                unsigned char g = image_data[i * 3 + 1];
                unsigned char b = image_data[i * 3 + 2];
                result.image_data[i] = static_cast<unsigned char>((r + g + b) / 3);
            }
            else if (channels == 3 && new_channels == 4) {
                // RGB -> RGBA (添加alpha通道)
                result.image_data[i * 4 + 0] = image_data[i * 3 + 0];
                result.image_data[i * 4 + 1] = image_data[i * 3 + 1];
                result.image_data[i * 4 + 2] = image_data[i * 3 + 2];
                result.image_data[i * 4 + 3] = 255;  // 完全不透明
            }
            else if (channels == 4 && new_channels == 3) {
                // RGBA -> RGB (丢弃alpha通道)
                result.image_data[i * 3 + 0] = image_data[i * 4 + 0];
                result.image_data[i * 3 + 1] = image_data[i * 4 + 1];
                result.image_data[i * 3 + 2] = image_data[i * 4 + 2];
            }
            else {
                // 其他情况：复制最小通道数
                int min_ch = std::min(channels, new_channels);
                for (int c = 0; c < min_ch; ++c) {
                    result.image_data[i * new_channels + c] = image_data[i * channels + c];
                }
                // 如果新通道更多，填充0
                for (int c = min_ch; c < new_channels; ++c) {
                    result.image_data[i * new_channels + c] = 0;
                }
            }
        }

        return result;
    };
};

// 3D点云
struct HVPointCloud
{
    std::vector<HVPoint3D> points;
};

// 图像 ROI
struct ImageRoi {
    size_t x; // 左上角x坐标
    size_t y; // 左上角y坐标
    size_t width;
    size_t height;
};

// 深度图数据类型
enum class DepthDataType {
    FLOAT32,   // 32位浮点型 (毫米为单位，真实深度值)
    UINT16     // 16位无符号整型 (0-65535原始值，需要z分辨率和偏移量转换)
};

//  深度图结构体
struct ImageDataInfoDepth {
    size_t width;
    size_t height;
    void* image_data;  // 使用void*存储不同类型的数据
    DepthDataType data_type;

    // UINT16类型的深度转换参数
    float z_resolution;  // z轴分辨率 (mm/unit)
    float z_offset;      // z轴偏移量 (mm)

    // 构造函数
    ImageDataInfoDepth()
        : width(0), height(0), image_data(nullptr),
        data_type(DepthDataType::FLOAT32), z_resolution(1.0f), z_offset(0.0f) {
    }

    ImageDataInfoDepth(size_t w, size_t h, DepthDataType type,
        float z_res, float z_off)
        : width(w), height(h), data_type(type),
        z_resolution(z_res), z_offset(z_off), image_data(nullptr) {
        allocate();
    }

    // 拷贝构造函数
    ImageDataInfoDepth(const ImageDataInfoDepth& other)
        : width(other.width), height(other.height), data_type(other.data_type),
        z_resolution(other.z_resolution), z_offset(other.z_offset), image_data(nullptr) {
        if (other.image_data) {
            allocate();
            std::memcpy(image_data, other.image_data, getDataSize());
        }
    }

    // 移动构造函数
    ImageDataInfoDepth(ImageDataInfoDepth&& other) noexcept
        : width(other.width), height(other.height),
        image_data(other.image_data), data_type(other.data_type),
        z_resolution(other.z_resolution), z_offset(other.z_offset) {
        other.image_data = nullptr;
        other.width = 0;
        other.height = 0;
    }

    // 拷贝赋值运算符
    ImageDataInfoDepth& operator=(const ImageDataInfoDepth& other) {
        if (this != &other) {
            release();
            width = other.width;
            height = other.height;
            data_type = other.data_type;
            z_resolution = other.z_resolution;
            z_offset = other.z_offset;
            if (other.image_data) {
                allocate();
                std::memcpy(image_data, other.image_data, getDataSize());
            }
        }
        return *this;
    }

    // 移动赋值运算符
    ImageDataInfoDepth& operator=(ImageDataInfoDepth&& other) noexcept {
        if (this != &other) {
            release();
            width = other.width;
            height = other.height;
            data_type = other.data_type;
            z_resolution = other.z_resolution;
            z_offset = other.z_offset;
            image_data = other.image_data;

            other.image_data = nullptr;
            other.width = 0;
            other.height = 0;
        }
        return *this;
    }

    // 析构函数
    ~ImageDataInfoDepth() {
        release();
    }

    // 分配内存
    void allocate() {
        if (width > 0 && height > 0) {
            size_t size = getDataSize();
            image_data = ::operator new(size);
            std::memset(image_data, 0, size);
        }
    }

    // 释放内存
    void release() {
        if (image_data) {
            ::operator delete(image_data);
            image_data = nullptr;
        }
    }

    // 重新分配内存
    void resize(size_t w, size_t h, DepthDataType type, float z_res, float z_off) {
        release();
        width = w;
        height = h;
        data_type = type;
        z_resolution = z_res;
        z_offset = z_off;
        allocate();
    }

    // 获取单个像素的字节数
    size_t getPixelSize() const {
        return data_type == DepthDataType::FLOAT32 ? sizeof(float) : sizeof(uint16_t);
    }

    // 获取数据大小（字节）
    size_t getDataSize() const {
        return width * height * getPixelSize();
    }

    // 检查是否为空
    bool empty() const {
        return image_data == nullptr || width == 0 || height == 0;
    }

    // 检查是否有效
    bool isValid() const {
        return !empty();
    }

    // 获取float*指针（仅当类型为FLOAT32时）
    float* asFloat() {
        if (data_type != DepthDataType::FLOAT32) {
            throw std::runtime_error("Data type is not FLOAT32");
        }
        return static_cast<float*>(image_data);
    }

    const float* asFloat() const {
        if (data_type != DepthDataType::FLOAT32) {
            throw std::runtime_error("Data type is not FLOAT32");
        }
        return static_cast<const float*>(image_data);
    }

    // 获取uint16_t*指针（仅当类型为UINT16时）
    uint16_t* asUInt16() {
        if (data_type != DepthDataType::UINT16) {
            throw std::runtime_error("Data type is not UINT16");
        }
        return static_cast<uint16_t*>(image_data);
    }

    const uint16_t* asUInt16() const {
        if (data_type != DepthDataType::UINT16) {
            throw std::runtime_error("Data type is not UINT16");
        }
        return static_cast<const uint16_t*>(image_data);
    }

    // 获取指定位置的深度值（返回真实深度，毫米）
    float getDepthAt(size_t row, size_t col) const {
        if (row >= height || col >= width) {
            throw std::out_of_range("Position out of range");
        }
        size_t idx = row * width + col;

        if (data_type == DepthDataType::FLOAT32) {
            return asFloat()[idx];
        }
        else {
            // UINT16: 原始值 -> 真实深度
            // depth_mm = raw_value * z_resolution + z_offset
            uint16_t raw_value = asUInt16()[idx];
            return raw_value * z_resolution + z_offset;
        }
    }

    // 设置指定位置的深度值（输入真实深度，毫米）
    void setDepthAt(size_t row, size_t col, float depth_mm) {
        if (row >= height || col >= width) {
            throw std::out_of_range("Position out of range");
        }
        size_t idx = row * width + col;

        if (data_type == DepthDataType::FLOAT32) {
            asFloat()[idx] = depth_mm;
        }
        else {
            // UINT16: 真实深度 -> 原始值
            // raw_value = (depth_mm - z_offset) / z_resolution
            float raw_value = (depth_mm - z_offset) / z_resolution;
            asUInt16()[idx] = static_cast<uint16_t>(
                std::min(std::max(raw_value, 0.0f), 65535.0f)
                );
        }
    }

    // 获取原始值（仅UINT16类型）
    uint16_t getRawValueAt(size_t row, size_t col) const {
        if (data_type != DepthDataType::UINT16) {
            throw std::runtime_error("getRawValueAt only works for UINT16 type");
        }
        if (row >= height || col >= width) {
            throw std::out_of_range("Position out of range");
        }
        return asUInt16()[row * width + col];
    }

    // 设置原始值（仅UINT16类型）
    void setRawValueAt(size_t row, size_t col, uint16_t raw_value) {
        if (data_type != DepthDataType::UINT16) {
            throw std::runtime_error("setRawValueAt only works for UINT16 type");
        }
        if (row >= height || col >= width) {
            throw std::out_of_range("Position out of range");
        }
        asUInt16()[row * width + col] = raw_value;
    }

    // 填充数据（输入真实深度，毫米）
    void fill(float depth_mm) {
        if (!image_data) return;

        if (data_type == DepthDataType::FLOAT32) {
            float* ptr = asFloat();
            for (size_t i = 0; i < width * height; ++i) {
                ptr[i] = depth_mm;
            }
        }
        else {
            // 真实深度 -> 原始值
            float raw_value = (depth_mm - z_offset) / z_resolution;
            uint16_t depth = static_cast<uint16_t>(
                std::min(std::max(raw_value, 0.0f), 65535.0f)
                );
            uint16_t* ptr = asUInt16();
            for (size_t i = 0; i < width * height; ++i) {
                ptr[i] = depth;
            }
        }
    }

    // 从外部数据复制
    void copyFrom(const void* data, size_t size) {
        if (!image_data) {
            throw std::runtime_error("Image data not allocated");
        }
        if (size != getDataSize()) {
            throw std::invalid_argument("Data size mismatch");
        }
        std::memcpy(image_data, data, size);
    }

    // 复制到外部缓冲区
    void copyTo(void* buffer, size_t buffer_size) const {
        if (!image_data) {
            throw std::runtime_error("Image data is empty");
        }
        size_t data_size = getDataSize();
        if (buffer_size < data_size) {
            throw std::invalid_argument("Buffer size too small");
        }
        std::memcpy(buffer, image_data, data_size);
    }

    // 类型转换：UINT16 -> FLOAT32
    ImageDataInfoDepth toFloat32() const {
        if (data_type == DepthDataType::FLOAT32) {
            return clone();
        }

        ImageDataInfoDepth result(width, height, DepthDataType::FLOAT32, z_resolution, z_offset);
        const uint16_t* src = asUInt16();
        float* dst = result.asFloat();

        for (size_t i = 0; i < width * height; ++i) {
            // 原始值转换为真实深度
            dst[i] = src[i] * z_resolution + z_offset;
        }

        return result;
    }

    // 类型转换：FLOAT32 -> UINT16
    ImageDataInfoDepth toUInt16(float z_res, float z_off) const {
        if (data_type == DepthDataType::UINT16) {
            return clone();
        }

        ImageDataInfoDepth result(width, height, DepthDataType::UINT16, z_res, z_off);
        const float* src = asFloat();
        uint16_t* dst = result.asUInt16();

        for (size_t i = 0; i < width * height; ++i) {
            // 真实深度转换为原始值
            float raw_value = (src[i] - z_off) / z_res;
            dst[i] = static_cast<uint16_t>(
                std::min(std::max(raw_value, 0.0f), 65535.0f)
                );
        }

        return result;
    }

    // 克隆（深拷贝）
    ImageDataInfoDepth clone() const {
        ImageDataInfoDepth result(width, height, data_type, z_resolution, z_offset);
        if (image_data) {
            std::memcpy(result.image_data, image_data, getDataSize());
        }
        return result;
    }

    // 获取行指针
    void* row(size_t r) {
        if (r >= height) {
            throw std::out_of_range("Row index out of range");
        }
        return static_cast<char*>(image_data) + r * width * getPixelSize();
    }

    const void* row(size_t r) const {
        if (r >= height) {
            throw std::out_of_range("Row index out of range");
        }
        return static_cast<const char*>(image_data) + r * width * getPixelSize();
    }

    // 获取步长（每行字节数）
    size_t step() const {
        return width * getPixelSize();
    }
};