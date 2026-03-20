#pragma once

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>

struct ICPParams {
    int method = 0; // 0: ICP, 1: GICP, 2: LP-ICP
    double voxel_size = 0.0;
    double max_correspondence_distance = 1.0;
    int max_iterations = 50;
    bool use_initial_guess = false;
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
};

struct ICPResult {
    bool success = false;
    bool converged = false;
    double fitness_score = -1.0;
    double rmse = -1.0;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_target;
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud;
    std::string error_message;
};

bool ParseTransformJson(const std::string& json_str, Eigen::Matrix4f& transform, std::string& error_message);

ICPResult RunPairRegistration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
                              const ICPParams& params);
